#include"../include/draw.h"
#include"internal/blend_utils.h"

using namespace pa2d::utils;
using namespace pa2d::utils::simd;

namespace pa2d {
    void circle(
        pa2d::Buffer& buffer,
        float centerX, float centerY, float radius,
        const pa2d::Color& fillColor,
        const pa2d::Color& strokeColor,
        float strokeWidth
    ) {
        if (!buffer.isValid() || radius <= 0) return;

        // --- 颜色和绘制模式预处理 ---
        const float finalFillOpacity = fillColor.a * (1.0f / 255.0f);
        const float finalStrokeOpacity = strokeColor.a * (1.0f / 255.0f);

        const bool drawFill = (finalFillOpacity > 0.0f);
        const bool drawStroke = (finalStrokeOpacity > 0.0f) && (strokeWidth > 0.0f);

        if (!drawFill && !drawStroke) return;

        // --- 宏观绘制模式判断 ---
        const bool mode_stroke_over_fill = drawStroke && drawFill;
        const bool mode_only_stroke = drawStroke && !drawFill;
        const bool mode_only_fill = drawFill && !drawStroke;

        // --- 边界计算和裁剪 ---
        const float antialiasRange = 1.0f;
        const float halfStrokeWidth = strokeWidth == 0 ? 0 : (strokeWidth + 1.0f) * 0.5f;

        // 浮点数安全余量和额外的索引安全余量
        const float EPSILON = 0.001f;
        const int INDEX_PADDING = 1;

        float maxDrawDist = radius;
        if (drawStroke) {
            maxDrawDist = std::max(maxDrawDist, radius + halfStrokeWidth);
        }
        const float outerRadius = maxDrawDist + antialiasRange;

        // minX/minY: 使用 floor 确定第一个像素索引
        const int minX = static_cast<int>(std::floor(centerX - outerRadius - EPSILON));
        const int minY = static_cast<int>(std::floor(centerY - outerRadius - EPSILON));

        // maxX/maxY: 使用 ceil 确定下一个整数坐标，然后减 1 得到索引，并加上 INDEX_PADDING 安全余量
        const int maxX_raw = static_cast<int>(std::ceil(centerX + outerRadius + EPSILON));
        const int maxY_raw = static_cast<int>(std::ceil(centerY + outerRadius + EPSILON));

        const int maxX = maxX_raw - 1 + INDEX_PADDING;
        const int maxY = maxY_raw - 1 + INDEX_PADDING;

        // 裁剪到缓冲区边界
        const int clampedMinX = std::max(0, minX);
        const int clampedMaxX = std::min(buffer.width - 1, maxX);
        const int clampedMinY = std::max(0, minY);
        const int clampedMaxY = std::min(buffer.height - 1, maxY);

        if (clampedMinX > clampedMaxX || clampedMinY > clampedMaxY) return;

        // --- AVX2 常量 ---
        const __m256 centerX_avx = _mm256_set1_ps(centerX);
        const __m256 centerY_avx = _mm256_set1_ps(centerY);
        const __m256 radius_avx = _mm256_set1_ps(radius);
        const __m256 halfStrokeWidth_avx = _mm256_set1_ps(halfStrokeWidth);
        const __m256 innerEdge_avx = _mm256_sub_ps(radius_avx, ANTIALIAS_RANGE_256);

        // 填充颜色 AVX 常量 (0-1)
        const __m256 fillR_avx = _mm256_set1_ps(fillColor.r * (1.0f / 255.0f));
        const __m256 fillG_avx = _mm256_set1_ps(fillColor.g * (1.0f / 255.0f));
        const __m256 fillB_avx = _mm256_set1_ps(fillColor.b * (1.0f / 255.0f));
        const __m256 strokeR_avx = _mm256_set1_ps(strokeColor.r * (1.0f / 255.0f));
        const __m256 strokeG_avx = _mm256_set1_ps(strokeColor.g * (1.0f / 255.0f));
        const __m256 strokeB_avx = _mm256_set1_ps(strokeColor.b * (1.0f / 255.0f));
        const __m256 strokeA_avx = _mm256_set1_ps(finalStrokeOpacity);
        const __m256 fillA_avx = _mm256_set1_ps(finalFillOpacity);

        // --- SSE 常量 ---
        const __m128 centerX_sse = _mm_set1_ps(centerX);
        const __m128 centerY_sse = _mm_set1_ps(centerY);
        const __m128 radius_sse = _mm_set1_ps(radius);
        const __m128 halfStrokeWidth_sse = _mm_set1_ps(halfStrokeWidth);
        const __m128 innerEdge_sse = _mm_sub_ps(radius_sse, ANTIALIAS_RANGE_128);

        const __m128 fillR_sse = _mm_set1_ps(fillColor.r * (1.0f / 255.0f));
        const __m128 fillG_sse = _mm_set1_ps(fillColor.g * (1.0f / 255.0f));
        const __m128 fillB_sse = _mm_set1_ps(fillColor.b * (1.0f / 255.0f));
        const __m128 fillA_sse = _mm_set1_ps(finalFillOpacity);

        const __m128 strokeR_sse = _mm_set1_ps(strokeColor.r * (1.0f / 255.0f));
        const __m128 strokeG_sse = _mm_set1_ps(strokeColor.g * (1.0f / 255.0f));
        const __m128 strokeB_sse = _mm_set1_ps(strokeColor.b * (1.0f / 255.0f));
        const __m128 strokeA_sse = _mm_set1_ps(finalStrokeOpacity);

        // --- 遍历像素 ---
        for (int y = clampedMinY; y <= clampedMaxY; ++y) {
            const float fy = static_cast<float>(y) + 0.5f;
            pa2d::Color* row = &buffer.at(0, y);

            const __m256 v_fy_avx = _mm256_set1_ps(fy);
            const __m128 v_fy_sse = _mm_set1_ps(fy);

            int x = clampedMinX;

            // --- AVX2 处理 (8像素) ---
            for (; x <= clampedMaxX - 7; x += 8) {
                __m256i xBase = _mm256_setr_epi32(x, x + 1, x + 2, x + 3, x + 4, x + 5, x + 6, x + 7);
                __m256 v_fx = _mm256_add_ps(_mm256_cvtepi32_ps(xBase), HALF_PIXEL_256);

                __m256 dx = _mm256_sub_ps(v_fx, centerX_avx);
                __m256 dy = _mm256_sub_ps(v_fy_avx, centerY_avx);
                __m256 distSq = _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy));
                __m256 dist = _mm256_sqrt_ps(distSq);

                // 1. 原始 Alpha 计算
                __m256 strokeAlpha_raw = ZERO_256;
                if (drawStroke) {
                    __m256 distToCircle = _mm256_sub_ps(dist, radius_avx);
                    __m256 absDistToCircle = _mm256_max_ps(_mm256_sub_ps(ZERO_256, distToCircle), distToCircle);
                    __m256 intensity = _mm256_div_ps(_mm256_sub_ps(halfStrokeWidth_avx, absDistToCircle), ANTIALIAS_RANGE_256);
                    strokeAlpha_raw = _mm256_max_ps(ZERO_256, _mm256_min_ps(ONE_256, intensity));
                    __m256 maxDist = _mm256_add_ps(halfStrokeWidth_avx, ANTIALIAS_RANGE_256);
                    __m256 inRange = _mm256_cmp_ps(absDistToCircle, maxDist, _CMP_LE_OQ);
                    strokeAlpha_raw = _mm256_and_ps(strokeAlpha_raw, inRange);
                }
                __m256 fillAlpha_raw = ZERO_256;
                if (drawFill) {
                    __m256 fillSolid = _mm256_cmp_ps(dist, innerEdge_avx, _CMP_LE_OQ);
                    __m256 fillAntialias = _mm256_and_ps(_mm256_cmp_ps(dist, innerEdge_avx, _CMP_GT_OQ), _mm256_cmp_ps(dist, radius_avx, _CMP_LE_OQ));
                    fillAlpha_raw = _mm256_blendv_ps(ZERO_256, ONE_256, fillSolid);
                    __m256 t = _mm256_div_ps(_mm256_sub_ps(dist, innerEdge_avx), ANTIALIAS_RANGE_256);
                    __m256 antialiasA = _mm256_sub_ps(ONE_256, t);
                    fillAlpha_raw = _mm256_blendv_ps(fillAlpha_raw, antialiasA, fillAntialias);
                }

                // 2. 应用全局不透明度
                __m256 effectiveStrokeAlpha = _mm256_mul_ps(strokeAlpha_raw, strokeA_avx);
                __m256 effectiveFillAlpha = _mm256_mul_ps(fillAlpha_raw, fillA_avx);

                __m256 finalAlpha;
                __m256 finalR, finalG, finalB;

                // --- 3. 核心混合逻辑  ---
                if (mode_stroke_over_fill) {
                    // Mode A: Stroke Over Fill (完整混合公式)
                    __m256 oneMinusEffectiveStrokeAlpha = _mm256_sub_ps(ONE_256, effectiveStrokeAlpha);
                    __m256 effectiveFillAlpha_modified = _mm256_mul_ps(effectiveFillAlpha, oneMinusEffectiveStrokeAlpha);

                    finalAlpha = _mm256_add_ps(effectiveStrokeAlpha, effectiveFillAlpha_modified);

                    finalR = _mm256_add_ps(_mm256_mul_ps(strokeR_avx, effectiveStrokeAlpha), _mm256_mul_ps(fillR_avx, effectiveFillAlpha_modified));
                    finalG = _mm256_add_ps(_mm256_mul_ps(strokeG_avx, effectiveStrokeAlpha), _mm256_mul_ps(fillG_avx, effectiveFillAlpha_modified));
                    finalB = _mm256_add_ps(_mm256_mul_ps(strokeB_avx, effectiveStrokeAlpha), _mm256_mul_ps(fillB_avx, effectiveFillAlpha_modified));

                    // 归一化颜色 (除以 finalAlpha)
                    __m256 invFinalAlpha = _mm256_div_ps(ONE_256, finalAlpha);
                    __m256 zero_mask = _mm256_cmp_ps(finalAlpha, ZERO_256, _CMP_EQ_OQ);
                    invFinalAlpha = _mm256_andnot_ps(zero_mask, invFinalAlpha);

                    finalR = _mm256_mul_ps(finalR, invFinalAlpha);
                    finalG = _mm256_mul_ps(finalG, invFinalAlpha);
                    finalB = _mm256_mul_ps(finalB, invFinalAlpha);
                }
                else if (mode_only_stroke) {
                    // Mode B: Only Stroke (跳过所有 Fill 计算)
                    finalAlpha = effectiveStrokeAlpha;
                    finalR = strokeR_avx;
                    finalG = strokeG_avx;
                    finalB = strokeB_avx;
                }
                else { // mode_only_fill
                    // Mode C: Only Fill (跳过所有 Stroke 计算)
                    finalAlpha = effectiveFillAlpha;
                    finalR = fillR_avx;
                    finalG = fillG_avx;
                    finalB = fillB_avx;
                }

                // 4. 写入目标缓冲区
                __m256 mask = _mm256_cmp_ps(finalAlpha, ZERO_256, _CMP_GT_OQ);
                if (_mm256_testz_ps(mask, mask)) continue;

                __m256i dest = _mm256_loadu_si256((__m256i*) & row[x]);
                __m256i rgba = blend_pixels_avx(
                    finalAlpha, dest,
                    finalR, finalG, finalB
                );

                rgba = _mm256_blendv_epi8(dest, rgba, _mm256_castps_si256(mask));
                _mm256_storeu_si256((__m256i*) & row[x], rgba);
            }

            // --- SSE 处理 (4像素) ---
            for (; x <= clampedMaxX - 3; x += 4) {
                __m128i xBase = _mm_setr_epi32(x, x + 1, x + 2, x + 3);
                __m128 v_fx = _mm_add_ps(_mm_cvtepi32_ps(xBase), HALF_PIXEL_128);

                __m128 dx = _mm_sub_ps(v_fx, centerX_sse);
                __m128 dy = _mm_sub_ps(v_fy_sse, centerY_sse);
                __m128 distSq = _mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy));
                __m128 dist = _mm_sqrt_ps(distSq);

                // 1. 原始 Alpha 计算
                __m128 strokeAlpha_raw = ZERO_128;
                if (drawStroke) {
                    __m128 distToCircle = _mm_sub_ps(dist, radius_sse);
                    __m128 absDistToCircle = _mm_max_ps(_mm_sub_ps(ZERO_128, distToCircle), distToCircle);
                    __m128 intensity = _mm_div_ps(_mm_sub_ps(halfStrokeWidth_sse, absDistToCircle), ANTIALIAS_RANGE_128);
                    strokeAlpha_raw = _mm_max_ps(ZERO_128, _mm_min_ps(ONE_128, intensity));
                    __m128 maxDist = _mm_add_ps(halfStrokeWidth_sse, ANTIALIAS_RANGE_128);
                    __m128 inRange = _mm_cmple_ps(absDistToCircle, maxDist);
                    strokeAlpha_raw = _mm_and_ps(strokeAlpha_raw, inRange);
                }
                __m128 fillAlpha_raw = ZERO_128;
                if (drawFill) {
                    __m128 fillSolid = _mm_cmple_ps(dist, innerEdge_sse);
                    __m128 fillAntialias = _mm_and_ps(_mm_cmpgt_ps(dist, innerEdge_sse), _mm_cmple_ps(dist, radius_sse));
                    fillAlpha_raw = _mm_blendv_ps(ZERO_128, ONE_128, fillSolid);
                    __m128 t = _mm_div_ps(_mm_sub_ps(dist, innerEdge_sse), ANTIALIAS_RANGE_128);
                    __m128 antialiasA = _mm_sub_ps(ONE_128, t);
                    fillAlpha_raw = _mm_blendv_ps(fillAlpha_raw, antialiasA, fillAntialias);
                }

                // 2. 应用全局不透明度
                __m128 effectiveStrokeAlpha = _mm_mul_ps(strokeAlpha_raw, strokeA_sse);
                __m128 effectiveFillAlpha = _mm_mul_ps(fillAlpha_raw, fillA_sse);

                __m128 finalAlpha;
                __m128 finalR, finalG, finalB;

                // --- 3. 核心混合逻辑 ---
                if (mode_stroke_over_fill) {
                    // Mode A: Stroke Over Fill
                    __m128 oneMinusEffectiveStrokeAlpha = _mm_sub_ps(ONE_128, effectiveStrokeAlpha);
                    __m128 effectiveFillAlpha_modified = _mm_mul_ps(effectiveFillAlpha, oneMinusEffectiveStrokeAlpha);

                    finalAlpha = _mm_add_ps(effectiveStrokeAlpha, effectiveFillAlpha_modified);

                    finalR = _mm_add_ps(_mm_mul_ps(strokeR_sse, effectiveStrokeAlpha), _mm_mul_ps(fillR_sse, effectiveFillAlpha_modified));
                    finalG = _mm_add_ps(_mm_mul_ps(strokeG_sse, effectiveStrokeAlpha), _mm_mul_ps(fillG_sse, effectiveFillAlpha_modified));
                    finalB = _mm_add_ps(_mm_mul_ps(strokeB_sse, effectiveStrokeAlpha), _mm_mul_ps(fillB_sse, effectiveFillAlpha_modified));

                    // 归一化颜色
                    __m128 invFinalAlpha = _mm_div_ps(ONE_128, finalAlpha);
                    invFinalAlpha = _mm_andnot_ps(_mm_cmpeq_ps(finalAlpha, ZERO_128), invFinalAlpha);

                    finalR = _mm_mul_ps(finalR, invFinalAlpha);
                    finalG = _mm_mul_ps(finalG, invFinalAlpha);
                    finalB = _mm_mul_ps(finalB, invFinalAlpha);
                }
                else if (mode_only_stroke) {
                    // Mode B: Only Stroke
                    finalAlpha = effectiveStrokeAlpha;
                    finalR = strokeR_sse;
                    finalG = strokeG_sse;
                    finalB = strokeB_sse;
                }
                else { // mode_only_fill
                    // Mode C: Only Fill
                    finalAlpha = effectiveFillAlpha;
                    finalR = fillR_sse;
                    finalG = fillG_sse;
                    finalB = fillB_sse;
                }

                // 4. 写入目标缓冲区
                __m128 mask = _mm_cmpgt_ps(finalAlpha, ZERO_128);
                if (_mm_movemask_ps(mask)) {
                    __m128i dest = _mm_loadu_si128((__m128i*) & row[x]);
                    __m128i rgba = blend_pixels_sse(
                        finalAlpha, dest,
                        finalR, finalG, finalB
                    );

                    rgba = _mm_blendv_epi8(dest, rgba, _mm_castps_si128(mask));
                    _mm_storeu_si128((__m128i*) & row[x], rgba);
                }
            }

            // --- 标量处理 (剩余像素) ---
            for (; x <= clampedMaxX; ++x) {
                const float fx = static_cast<float>(x) + 0.5f;
                const float dx = fx - centerX;
                const float dy = fy - centerY;
                const float dist = std::sqrt(dx * dx + dy * dy);

                // 1. 原始 Alpha 计算
                float strokeAlpha_raw = 0.0f;
                if (drawStroke) {
                    const float distToCircle = std::abs(dist - radius);
                    if (distToCircle <= halfStrokeWidth + antialiasRange) {
                        float intensity = (halfStrokeWidth - distToCircle) / antialiasRange;
                        strokeAlpha_raw = std::max(0.0f, std::min(1.0f, intensity));
                    }
                }
                float fillAlpha_raw = 0.0f;
                if (drawFill) {
                    const float innerEdge = radius - antialiasRange;
                    if (dist <= innerEdge) {
                        fillAlpha_raw = 1.0f;
                    }
                    else if (dist <= radius) {
                        float t = (dist - innerEdge) / antialiasRange;
                        fillAlpha_raw = 1.0f - t;
                    }
                }

                // 2. 应用全局不透明度
                const float effectiveStrokeAlpha = strokeAlpha_raw * finalStrokeOpacity;
                const float effectiveFillAlpha = fillAlpha_raw * finalFillOpacity;

                float finalAlpha_s;
                float R_src_pre, G_src_pre, B_src_pre;

                // --- 3. 核心混合逻辑 ---
                if (mode_stroke_over_fill) {
                    // Mode A: Stroke Over Fill
                    const float oneMinusEffectiveStrokeAlpha = 1.0f - effectiveStrokeAlpha;
                    const float effectiveFillAlpha_modified = effectiveFillAlpha * oneMinusEffectiveStrokeAlpha;

                    finalAlpha_s = effectiveStrokeAlpha + effectiveFillAlpha_modified;

                    R_src_pre = strokeColor.r * (effectiveStrokeAlpha / 255.0f) + fillColor.r * (effectiveFillAlpha_modified / 255.0f);
                    G_src_pre = strokeColor.g * (effectiveStrokeAlpha / 255.0f) + fillColor.g * (effectiveFillAlpha_modified / 255.0f);
                    B_src_pre = strokeColor.b * (effectiveStrokeAlpha / 255.0f) + fillColor.b * (effectiveFillAlpha_modified / 255.0f);
                }
                else if (mode_only_stroke) {
                    // Mode B: Only Stroke
                    finalAlpha_s = effectiveStrokeAlpha;
                    R_src_pre = strokeColor.r * (effectiveStrokeAlpha / 255.0f);
                    G_src_pre = strokeColor.g * (effectiveStrokeAlpha / 255.0f);
                    B_src_pre = strokeColor.b * (effectiveStrokeAlpha / 255.0f);
                }
                else { // mode_only_fill
                    // Mode C: Only Fill
                    finalAlpha_s = effectiveFillAlpha;
                    R_src_pre = fillColor.r * (effectiveFillAlpha / 255.0f);
                    G_src_pre = fillColor.g * (effectiveFillAlpha / 255.0f);
                    B_src_pre = fillColor.b * (effectiveFillAlpha / 255.0f);
                }

                // 4. 写入目标缓冲区
                if (finalAlpha_s > 0.0f) {
                    pa2d::Color srcColor;

                    // 完全不透明优化
                    if (finalAlpha_s >= 1.0f) {
                        srcColor.r = static_cast<uint8_t>(std::min(255.0f, R_src_pre * 255.0f));
                        srcColor.g = static_cast<uint8_t>(std::min(255.0f, G_src_pre * 255.0f));
                        srcColor.b = static_cast<uint8_t>(std::min(255.0f, B_src_pre * 255.0f));
                        srcColor.a = 255;
                    }
                    else {
                        // 半透明混合：需要归一化颜色
                        float invFinalAlpha = 1.0f / finalAlpha_s;
                        srcColor.r = static_cast<uint8_t>(std::min(255.0f, R_src_pre * invFinalAlpha * 255.0f));
                        srcColor.g = static_cast<uint8_t>(std::min(255.0f, G_src_pre * invFinalAlpha * 255.0f));
                        srcColor.b = static_cast<uint8_t>(std::min(255.0f, B_src_pre * invFinalAlpha * 255.0f));
                        srcColor.a = static_cast<uint8_t>(finalAlpha_s * 255.0f);
                    }

                    pa2d::Color& dest = row[x];
                    row[x] = Blend(srcColor, dest);
                }
            }
        }
    }
}