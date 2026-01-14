#include"../include/draw.h"
#include"internal/blend_utils.h"

using namespace pa2d::utils;
using namespace pa2d::utils::simd;

namespace pa2d {
    void roundRect(
        pa2d::Buffer& buffer,
        float x, float y, float width, float height,
        const pa2d::Color& fillColor,
        const pa2d::Color& strokeColor,
        float cornerRadius,
        float strokeWidth
    ) {
        if (!buffer.isValid() || width <= 0 || height <= 0) return;

        // --- 1. 颜色和绘制模式预处理 ---
        const float finalFillOpacity = fillColor.a * (1.0f / 255.0f);
        const float finalStrokeOpacity = strokeColor.a * (1.0f / 255.0f);

        const bool drawFill = (finalFillOpacity > 0.0f);
        const bool drawStroke = (finalStrokeOpacity > 0.0f) && (strokeWidth > 0.0f);

        if (!drawFill && !drawStroke) return;

        const bool mode_stroke_over_fill = drawStroke && drawFill;
        const bool mode_only_stroke = drawStroke && !drawFill;
        const bool mode_only_fill = drawFill && !drawStroke;

        // --- 2. 几何参数 ---
        cornerRadius = std::max(0.0f, std::min(cornerRadius, std::min(width, height) * 0.5f));
        const float centerX = x + width * 0.5f;
        const float centerY = y + height * 0.5f;
        const float halfWidth = width * 0.5f;
        const float halfHeight = height * 0.5f;
        const float halfStrokeWidth = strokeWidth == 0 ? 0 : (strokeWidth + 1.0f) * 0.5f;

        // --- 3. 边界计算和裁剪 (安全逻辑) ---
        const float antialiasRange = 1.0f;
        const float maxExt = std::max({ cornerRadius, halfWidth, halfHeight });
        const float maxRadius = maxExt + halfStrokeWidth + antialiasRange;

        const float EPSILON = 0.001f;
        const int INDEX_PADDING = 1;

        const int minX = static_cast<int>(std::floor(centerX - maxRadius - EPSILON));
        const int minY = static_cast<int>(std::floor(centerY - maxRadius - EPSILON));

        const int maxX_raw = static_cast<int>(std::ceil(centerX + maxRadius + EPSILON));
        const int maxY_raw = static_cast<int>(std::ceil(centerY + maxRadius + EPSILON));

        const int maxX = maxX_raw - 1 + INDEX_PADDING;
        const int maxY = maxY_raw - 1 + INDEX_PADDING;

        const int clampedMinX = std::max(0, minX);
        const int clampedMaxX = std::min(buffer.width - 1, maxX);
        const int clampedMinY = std::max(0, minY);
        const int clampedMaxY = std::min(buffer.height - 1, maxY);

        if (clampedMinX > clampedMaxX || clampedMinY > clampedMaxY) return;

        // --- 4. AVX2 常量 ---
        const __m256 centerX_v = _mm256_set1_ps(centerX);
        const __m256 centerY_v = _mm256_set1_ps(centerY);
        const __m256 halfWidth_v = _mm256_set1_ps(halfWidth);
        const __m256 halfHeight_v = _mm256_set1_ps(halfHeight);
        const __m256 cornerRadius_v = _mm256_set1_ps(cornerRadius);
        const __m256 halfStrokeWidth_v = _mm256_set1_ps(halfStrokeWidth);

        const __m256 fillR_v = _mm256_set1_ps(fillColor.r * (1.0f / 255.0f));
        const __m256 fillG_v = _mm256_set1_ps(fillColor.g * (1.0f / 255.0f));
        const __m256 fillB_v = _mm256_set1_ps(fillColor.b * (1.0f / 255.0f));
        const __m256 strokeR_v = _mm256_set1_ps(strokeColor.r * (1.0f / 255.0f));
        const __m256 strokeG_v = _mm256_set1_ps(strokeColor.g * (1.0f / 255.0f));
        const __m256 strokeB_v = _mm256_set1_ps(strokeColor.b * (1.0f / 255.0f));
        const __m256 fillA_v = _mm256_set1_ps(finalFillOpacity);
        const __m256 strokeA_v = _mm256_set1_ps(finalStrokeOpacity);

        // --- 5. SSE 常量 ---
        const __m128 centerX_sse = _mm_set1_ps(centerX);
        const __m128 centerY_sse = _mm_set1_ps(centerY);
        const __m128 halfWidth_sse = _mm_set1_ps(halfWidth);
        const __m128 halfHeight_sse = _mm_set1_ps(halfHeight);
        const __m128 cornerRadius_sse = _mm_set1_ps(cornerRadius);
        const __m128 halfStrokeWidth_sse = _mm_set1_ps(halfStrokeWidth);

        const __m128 fillR_sse = _mm_set1_ps(fillColor.r * (1.0f / 255.0f));
        const __m128 fillG_sse = _mm_set1_ps(fillColor.g * (1.0f / 255.0f));
        const __m128 fillB_sse = _mm_set1_ps(fillColor.b * (1.0f / 255.0f));
        const __m128 strokeR_sse = _mm_set1_ps(strokeColor.r * (1.0f / 255.0f));
        const __m128 strokeG_sse = _mm_set1_ps(strokeColor.g * (1.0f / 255.0f));
        const __m128 strokeB_sse = _mm_set1_ps(strokeColor.b * (1.0f / 255.0f));
        const __m128 fillA_sse = _mm_set1_ps(finalFillOpacity);
        const __m128 strokeA_sse = _mm_set1_ps(finalStrokeOpacity);

        auto RoundRectSDF_AVX2 = [](
            __m256 px_v, __m256 py_v,
            __m256 centerX_v, __m256 centerY_v,
            __m256 halfWidth_v, __m256 halfHeight_v,
            __m256 cornerRadius_v)
            {
                // P 相对 C 的绝对坐标
                __m256 dx_abs = _mm256_sub_ps(_mm256_max_ps(px_v, centerX_v), _mm256_min_ps(px_v, centerX_v));
                __m256 dy_abs = _mm256_sub_ps(_mm256_max_ps(py_v, centerY_v), _mm256_min_ps(py_v, centerY_v));

                // Q 向量：Q = abs(P) - B + R
                __m256 ax = _mm256_sub_ps(_mm256_sub_ps(dx_abs, halfWidth_v), _mm256_sub_ps(ZERO_256, cornerRadius_v));
                __m256 ay = _mm256_sub_ps(_mm256_sub_ps(dy_abs, halfHeight_v), _mm256_sub_ps(ZERO_256, cornerRadius_v));

                // 外部距离项 length(max(Q, 0.0)) 的两个部分 (cx, cy)
                __m256 cx = _mm256_max_ps(ax, ZERO_256);
                __m256 cy = _mm256_max_ps(ay, ZERO_256);

                __m256 distSq = _mm256_add_ps(_mm256_mul_ps(cx, cx), _mm256_mul_ps(cy, cy));
                __m256 dist = _mm256_sqrt_ps(distSq);

                // 最终 SDF = min(max(Q.x, Q.y), 0.0) + dist - R
                __m256 min_max_aa = _mm256_min_ps(_mm256_max_ps(ax, ay), ZERO_256);
                __m256 sdf = _mm256_sub_ps(_mm256_add_ps(min_max_aa, dist), cornerRadius_v);

                return sdf;
            };

        auto RoundRectSDF_SSE = [](
            __m128 px_v, __m128 py_v,
            __m128 centerX_v, __m128 centerY_v,
            __m128 halfWidth_v, __m128 halfHeight_v,
            __m128 cornerRadius_v)
            {
                // P 相对 C 的绝对坐标
                __m128 dx_abs = _mm_sub_ps(_mm_max_ps(px_v, centerX_v), _mm_min_ps(px_v, centerX_v));
                __m128 dy_abs = _mm_sub_ps(_mm_max_ps(py_v, centerY_v), _mm_min_ps(py_v, centerY_v));

                // Q 向量：Q = abs(P) - B + R
                __m128 ax = _mm_sub_ps(_mm_sub_ps(dx_abs, halfWidth_v), _mm_sub_ps(ZERO_128, cornerRadius_v));
                __m128 ay = _mm_sub_ps(_mm_sub_ps(dy_abs, halfHeight_v), _mm_sub_ps(ZERO_128, cornerRadius_v));

                // 外部距离项 length(max(Q, 0.0)) 的两个部分 (cx, cy)
                __m128 cx = _mm_max_ps(ax, ZERO_128);
                __m128 cy = _mm_max_ps(ay, ZERO_128);

                __m128 distSq = _mm_add_ps(_mm_mul_ps(cx, cx), _mm_mul_ps(cy, cy));
                __m128 dist = _mm_sqrt_ps(distSq);

                // 最终 SDF = min(max(Q.x, Q.y), 0.0) + dist - R
                __m128 min_max_aa = _mm_min_ps(_mm_max_ps(ax, ay), ZERO_128);
                __m128 sdf = _mm_sub_ps(_mm_add_ps(min_max_aa, dist), cornerRadius_v);

                return sdf;
            };

        // --- 6. 主绘制循环 ---
        for (int py = clampedMinY; py <= clampedMaxY; ++py) {
            const float fy = static_cast<float>(py) + 0.5f;
            pa2d::Color* row = &buffer.at(0, py);

            const __m256 py_v = _mm256_set1_ps(fy);
            const __m128 py_v_sse = _mm_set1_ps(fy);

            int px = clampedMinX;

            // --- AVX2处理 (8像素) ---
            for (; px <= clampedMaxX - 7; px += 8) {
                __m256i xBase = _mm256_setr_epi32(px, px + 1, px + 2, px + 3, px + 4, px + 5, px + 6, px + 7);
                __m256 px_v = _mm256_add_ps(_mm256_cvtepi32_ps(xBase), HALF_PIXEL_256);

                // A. 计算 SDF 距离
                __m256 sdf = RoundRectSDF_AVX2(
                    px_v, py_v, centerX_v, centerY_v,
                    halfWidth_v, halfHeight_v, cornerRadius_v
                );

                // B. Alpha 计算 (SDF -> Alpha)
                __m256 effectiveFillAlpha = ZERO_256;
                if (drawFill) {
                    __m256 t_fill = _mm256_div_ps(_mm256_sub_ps(sdf, ANTIALIAS_RANGE_256), ANTIALIAS_RANGE_256);
                    __m256 fillAlpha_raw = _mm256_sub_ps(ONE_256, t_fill);
                    effectiveFillAlpha = _mm256_mul_ps(_mm256_max_ps(ZERO_256, _mm256_min_ps(ONE_256, fillAlpha_raw)), fillA_v);
                }

                __m256 effectiveStrokeAlpha = ZERO_256;
                if (drawStroke) {
                    __m256 distToStrokeCenter = _mm256_max_ps(_mm256_sub_ps(ZERO_256, sdf), sdf); // |sdf|
                    __m256 halfWidthMinusDist = _mm256_sub_ps(halfStrokeWidth_v, distToStrokeCenter);
                    __m256 t_stroke = _mm256_div_ps(halfWidthMinusDist, ANTIALIAS_RANGE_256);

                    __m256 strokeAlpha_raw = _mm256_max_ps(ZERO_256, _mm256_min_ps(ONE_256, t_stroke));
                    effectiveStrokeAlpha = _mm256_mul_ps(strokeAlpha_raw, strokeA_v);
                }

                // C. 颜色混合和写入
                __m256 finalAlpha;
                __m256 finalR, finalG, finalB;

                if (mode_stroke_over_fill) {
                    __m256 oneMinusEffectiveStrokeAlpha = _mm256_sub_ps(ONE_256, effectiveStrokeAlpha);
                    __m256 effectiveFillAlpha_modified = _mm256_mul_ps(effectiveFillAlpha, oneMinusEffectiveStrokeAlpha);
                    finalAlpha = _mm256_add_ps(effectiveStrokeAlpha, effectiveFillAlpha_modified);

                    finalR = _mm256_add_ps(_mm256_mul_ps(strokeR_v, effectiveStrokeAlpha), _mm256_mul_ps(fillR_v, effectiveFillAlpha_modified));
                    finalG = _mm256_add_ps(_mm256_mul_ps(strokeG_v, effectiveStrokeAlpha), _mm256_mul_ps(fillG_v, effectiveFillAlpha_modified));
                    finalB = _mm256_add_ps(_mm256_mul_ps(strokeB_v, effectiveStrokeAlpha), _mm256_mul_ps(fillB_v, effectiveFillAlpha_modified));

                    __m256 invFinalAlpha = _mm256_div_ps(ONE_256, finalAlpha);
                    __m256 zero_mask = _mm256_cmp_ps(finalAlpha, ZERO_256, _CMP_EQ_OQ);
                    invFinalAlpha = _mm256_andnot_ps(zero_mask, invFinalAlpha);

                    finalR = _mm256_mul_ps(finalR, invFinalAlpha);
                    finalG = _mm256_mul_ps(finalG, invFinalAlpha);
                    finalB = _mm256_mul_ps(finalB, invFinalAlpha);
                }
                else if (mode_only_stroke) {
                    finalAlpha = effectiveStrokeAlpha;
                    finalR = strokeR_v; finalG = strokeG_v; finalB = strokeB_v;
                }
                else { // mode_only_fill
                    finalAlpha = effectiveFillAlpha;
                    finalR = fillR_v; finalG = fillG_v; finalB = fillB_v;
                }

                __m256 mask = _mm256_cmp_ps(finalAlpha, ZERO_256, _CMP_GT_OQ);
                if (_mm256_testz_ps(mask, mask)) continue;

                __m256i dest = _mm256_loadu_si256((__m256i*) & row[px]);
                __m256i rgba = blend_pixels_avx(finalAlpha, dest, finalR, finalG, finalB);
                rgba = _mm256_blendv_epi8(dest, rgba, _mm256_castps_si256(mask));
                _mm256_storeu_si256((__m256i*) & row[px], rgba);
            }

            // --- SSE处理 (4像素) ---
            for (; px <= clampedMaxX - 3; px += 4) {
                __m128i xBase = _mm_setr_epi32(px, px + 1, px + 2, px + 3);
                __m128 px_v_sse = _mm_add_ps(_mm_cvtepi32_ps(xBase), HALF_PIXEL_128);

                // A. 计算 SSE SDF 距离
                __m128 sdf = RoundRectSDF_SSE(
                    px_v_sse, py_v_sse, centerX_sse, centerY_sse,
                    halfWidth_sse, halfHeight_sse, cornerRadius_sse
                );

                // B. Alpha 计算 (SDF -> Alpha)
                __m128 effectiveFillAlpha = ZERO_128;
                if (drawFill) {
                    __m128 t_fill = _mm_div_ps(_mm_sub_ps(sdf, ANTIALIAS_RANGE_128), ANTIALIAS_RANGE_128);
                    __m128 fillAlpha_raw = _mm_sub_ps(ONE_128, t_fill);
                    effectiveFillAlpha = _mm_mul_ps(_mm_max_ps(ZERO_128, _mm_min_ps(ONE_128, fillAlpha_raw)), fillA_sse);
                }

                __m128 effectiveStrokeAlpha = ZERO_128;
                if (drawStroke) {
                    __m128 distToStrokeCenter = _mm_max_ps(_mm_sub_ps(ZERO_128, sdf), sdf); // |sdf|
                    __m128 halfWidthMinusDist = _mm_sub_ps(halfStrokeWidth_sse, distToStrokeCenter);
                    __m128 t_stroke = _mm_div_ps(halfWidthMinusDist, ANTIALIAS_RANGE_128);

                    __m128 strokeAlpha_raw = _mm_max_ps(ZERO_128, _mm_min_ps(ONE_128, t_stroke));
                    effectiveStrokeAlpha = _mm_mul_ps(strokeAlpha_raw, strokeA_sse);
                }

                // C. 颜色混合和写入
                __m128 finalAlpha;
                __m128 finalR, finalG, finalB;

                if (mode_stroke_over_fill) {
                    __m128 oneMinusEffectiveStrokeAlpha = _mm_sub_ps(ONE_128, effectiveStrokeAlpha);
                    __m128 effectiveFillAlpha_modified = _mm_mul_ps(effectiveFillAlpha, oneMinusEffectiveStrokeAlpha);
                    finalAlpha = _mm_add_ps(effectiveStrokeAlpha, effectiveFillAlpha_modified);

                    finalR = _mm_add_ps(_mm_mul_ps(strokeR_sse, effectiveStrokeAlpha), _mm_mul_ps(fillR_sse, effectiveFillAlpha_modified));
                    finalG = _mm_add_ps(_mm_mul_ps(strokeG_sse, effectiveStrokeAlpha), _mm_mul_ps(fillG_sse, effectiveFillAlpha_modified));
                    finalB = _mm_add_ps(_mm_mul_ps(strokeB_sse, effectiveStrokeAlpha), _mm_mul_ps(fillB_sse, effectiveFillAlpha_modified));

                    __m128 invFinalAlpha = _mm_div_ps(ONE_128, finalAlpha);
                    __m128 zero_mask = _mm_cmpeq_ps(finalAlpha, ZERO_128);
                    invFinalAlpha = _mm_andnot_ps(zero_mask, invFinalAlpha);

                    finalR = _mm_mul_ps(finalR, invFinalAlpha);
                    finalG = _mm_mul_ps(finalG, invFinalAlpha);
                    finalB = _mm_mul_ps(finalB, invFinalAlpha);
                }
                else if (mode_only_stroke) {
                    finalAlpha = effectiveStrokeAlpha;
                    finalR = strokeR_sse; finalG = strokeG_sse; finalB = strokeB_sse;
                }
                else { // mode_only_fill
                    finalAlpha = effectiveFillAlpha;
                    finalR = fillR_sse; finalG = fillG_sse; finalB = fillB_sse;
                }

                __m128 mask = _mm_cmpgt_ps(finalAlpha, ZERO_128);
                if (_mm_movemask_ps(mask)) {
                    __m128i dest = _mm_loadu_si128((__m128i*) & row[px]);
                    __m128i rgba = blend_pixels_sse(finalAlpha, dest, finalR, finalG, finalB);
                    rgba = _mm_blendv_epi8(dest, rgba, _mm_castps_si128(mask));
                    _mm_storeu_si128((__m128i*) & row[px], rgba);
                }
            }

            // --- 标量收尾 (1-3 像素) ---
            for (; px <= clampedMaxX; ++px) {
                const float fx = static_cast<float>(px) + 0.5f;

                // A. 标量 SDF 距离计算
                float dx_abs = std::abs(fx - centerX);
                float dy_abs = std::abs(fy - centerY);

                float ax = dx_abs - halfWidth + cornerRadius;
                float ay = dy_abs - halfHeight + cornerRadius;

                float cx = std::max(ax, 0.0f);
                float cy = std::max(ay, 0.0f);

                float sdf = std::min(std::max(ax, ay), 0.0f) + std::sqrt(cx * cx + cy * cy) - cornerRadius;

                // B. Alpha 计算
                float effectiveFillAlpha = 0.0f;
                if (drawFill) {
                    float t_fill = (sdf - antialiasRange) / antialiasRange;
                    float fillAlpha_raw = 1.0f - t_fill;
                    effectiveFillAlpha = std::max(0.0f, std::min(1.0f, fillAlpha_raw)) * finalFillOpacity;
                }

                float effectiveStrokeAlpha = 0.0f;
                if (drawStroke) {
                    float distToStrokeCenter = std::abs(sdf);
                    float halfWidthMinusDist = halfStrokeWidth - distToStrokeCenter;
                    float t_stroke = halfWidthMinusDist / antialiasRange;

                    float strokeAlpha_raw = std::max(0.0f, std::min(1.0f, t_stroke));
                    effectiveStrokeAlpha = strokeAlpha_raw * finalStrokeOpacity;
                }

                // C. 颜色混合和写入
                float finalAlpha;
                float R_src_pre, G_src_pre, B_src_pre;

                if (mode_stroke_over_fill) {
                    const float oneMinusEffectiveStrokeAlpha = 1.0f - effectiveStrokeAlpha;
                    const float effectiveFillAlpha_modified = effectiveFillAlpha * oneMinusEffectiveStrokeAlpha;
                    finalAlpha = effectiveStrokeAlpha + effectiveFillAlpha_modified;

                    R_src_pre = strokeColor.r * (effectiveStrokeAlpha / 255.0f) + fillColor.r * (effectiveFillAlpha_modified / 255.0f);
                    G_src_pre = strokeColor.g * (effectiveStrokeAlpha / 255.0f) + fillColor.g * (effectiveFillAlpha_modified / 255.0f);
                    B_src_pre = strokeColor.b * (effectiveStrokeAlpha / 255.0f) + fillColor.b * (effectiveFillAlpha_modified / 255.0f);
                }
                else if (mode_only_stroke) {
                    finalAlpha = effectiveStrokeAlpha;
                    R_src_pre = strokeColor.r * (effectiveStrokeAlpha / 255.0f);
                    G_src_pre = strokeColor.g * (effectiveStrokeAlpha / 255.0f);
                    B_src_pre = strokeColor.b * (effectiveStrokeAlpha / 255.0f);
                }
                else { // mode_only_fill
                    finalAlpha = effectiveFillAlpha;
                    R_src_pre = fillColor.r * (effectiveFillAlpha / 255.0f);
                    G_src_pre = fillColor.g * (effectiveFillAlpha / 255.0f);
                    B_src_pre = fillColor.b * (effectiveFillAlpha / 255.0f);
                }

                if (finalAlpha > 0.0f) {
                    pa2d::Color srcColor;
                    if (finalAlpha >= 1.0f) {
                        srcColor.r = static_cast<uint8_t>(std::min(255.0f, R_src_pre * 255.0f));
                        srcColor.g = static_cast<uint8_t>(std::min(255.0f, G_src_pre * 255.0f));
                        srcColor.b = static_cast<uint8_t>(std::min(255.0f, B_src_pre * 255.0f));
                        srcColor.a = 255;
                    }
                    else {
                        float invFinalAlpha = 1.0f / finalAlpha;
                        srcColor.r = static_cast<uint8_t>(std::min(255.0f, R_src_pre * invFinalAlpha * 255.0f));
                        srcColor.g = static_cast<uint8_t>(std::min(255.0f, G_src_pre * invFinalAlpha * 255.0f));
                        srcColor.b = static_cast<uint8_t>(std::min(255.0f, B_src_pre * invFinalAlpha * 255.0f));
                        srcColor.a = static_cast<uint8_t>(finalAlpha * 255.0f);
                    }
                    pa2d::Color& dest = row[px];
                    row[px] = Blend(srcColor, dest);
                }
            }
        }
    }


    void roundRect(
        pa2d::Buffer& buffer,
        float centerX, float centerY, float width, float height, float angle,
        const pa2d::Color& fillColor,
        const pa2d::Color& strokeColor,
        float cornerRadius,
        float strokeWidth
    ) {
        // 1. 参数校验和预处理
        if (!buffer.isValid() || width <= 0 || height <= 0) return;

        const float finalFillOpacity = fillColor.a * (1.0f / 255.0f);
        const float finalStrokeOpacity = strokeColor.a * (1.0f / 255.0f);

        const bool drawFill = (finalFillOpacity > 0.0f);
        const bool drawStroke = (finalStrokeOpacity > 0.0f) && (strokeWidth > 0.0f);

        if (!drawFill && !drawStroke) return;

        const bool mode_stroke_over_fill = drawStroke && drawFill;
        const bool mode_only_stroke = drawStroke && !drawFill;
        const bool mode_only_fill = drawFill && !drawStroke;

        // 2. 几何参数
        // 约束圆角半径
        cornerRadius = std::max(0.0f, std::min(cornerRadius, std::min(width, height) * 0.5f));
        const float halfWidth = width * 0.5f;
        const float halfHeight = height * 0.5f;
        const float halfStrokeWidth = strokeWidth == 0 ? 0 : (strokeWidth + 1.0f) * 0.5f;
        angle = angle * (GEOMETRY_PI / 180.0f);

        // 2.5 旋转和包围盒计算
        const float s = std::sin(angle);
        const float c = std::cos(angle);
        const float abs_s = std::abs(s);
        const float abs_c = std::abs(c);

        // 计算旋转后矩形的世界空间包围盒
        const float worldHalfWidth = abs_c * halfWidth + abs_s * halfHeight;
        const float worldHalfHeight = abs_s * halfWidth + abs_c * halfHeight;

        // 3. 边界计算和裁剪
        const float antialiasRange = 1.0f;  // 使用全局常量
        const float maxExtX = worldHalfWidth + halfStrokeWidth + antialiasRange;
        const float maxExtY = worldHalfHeight + halfStrokeWidth + antialiasRange;

        int minX = static_cast<int>(std::floor(centerX - maxExtX));
        int maxX = static_cast<int>(std::ceil(centerX + maxExtX));
        int minY = static_cast<int>(std::floor(centerY - maxExtY));
        int maxY = static_cast<int>(std::ceil(centerY + maxExtY));

        minX = std::max(0, minX);
        maxX = std::min(buffer.width - 1, maxX);
        minY = std::max(0, minY);
        maxY = std::min(buffer.height - 1, maxY);

        if (minX > maxX || minY > maxY) return;

        // 4. SIMD 常量准备
        // 使用全局常量
        const __m256 centerX_v = _mm256_set1_ps(centerX);
        const __m256 centerY_v = _mm256_set1_ps(centerY);

        // SDF 计算使用 halfWidth/Height 和 cornerRadius 的组合
        const __m256 sdf_w_v = _mm256_set1_ps(halfWidth - cornerRadius);
        const __m256 sdf_h_v = _mm256_set1_ps(halfHeight - cornerRadius);
        const __m256 cornerRadius_v = _mm256_set1_ps(cornerRadius);

        const __m256 halfStrokeWidth_v = _mm256_set1_ps(halfStrokeWidth);

        // 旋转常量
        const __m256 s_v = _mm256_set1_ps(s);
        const __m256 c_v = _mm256_set1_ps(c);

        // 颜色常量
        const __m256 fillR_v = _mm256_set1_ps(fillColor.r * (1.0f / 255.0f));
        const __m256 fillG_v = _mm256_set1_ps(fillColor.g * (1.0f / 255.0f));
        const __m256 fillB_v = _mm256_set1_ps(fillColor.b * (1.0f / 255.0f));
        const __m256 strokeR_v = _mm256_set1_ps(strokeColor.r * (1.0f / 255.0f));
        const __m256 strokeG_v = _mm256_set1_ps(strokeColor.g * (1.0f / 255.0f));
        const __m256 strokeB_v = _mm256_set1_ps(strokeColor.b * (1.0f / 255.0f));
        const __m256 fillA_v = _mm256_set1_ps(finalFillOpacity);
        const __m256 strokeA_v = _mm256_set1_ps(finalStrokeOpacity);

        // 5. SSE 常量
        const __m128 centerX_sse = _mm_set1_ps(centerX);
        const __m128 centerY_sse = _mm_set1_ps(centerY);

        const __m128 sdf_w_sse = _mm_set1_ps(halfWidth - cornerRadius);
        const __m128 sdf_h_sse = _mm_set1_ps(halfHeight - cornerRadius);
        const __m128 cornerRadius_sse = _mm_set1_ps(cornerRadius);

        const __m128 halfStrokeWidth_sse = _mm_set1_ps(halfStrokeWidth);

        const __m128 s_sse = _mm_set1_ps(s);
        const __m128 c_sse = _mm_set1_ps(c);

        const __m128 fillR_sse = _mm_set1_ps(fillColor.r * (1.0f / 255.0f));
        const __m128 fillG_sse = _mm_set1_ps(fillColor.g * (1.0f / 255.0f));
        const __m128 fillB_sse = _mm_set1_ps(fillColor.b * (1.0f / 255.0f));
        const __m128 strokeR_sse = _mm_set1_ps(strokeColor.r * (1.0f / 255.0f));
        const __m128 strokeG_sse = _mm_set1_ps(strokeColor.g * (1.0f / 255.0f));
        const __m128 strokeB_sse = _mm_set1_ps(strokeColor.b * (1.0f / 255.0f));
        const __m128 fillA_sse = _mm_set1_ps(finalFillOpacity);
        const __m128 strokeA_sse = _mm_set1_ps(finalStrokeOpacity);

        // 6. 主绘制循环
        for (int py = minY; py <= maxY; ++py) {
            const float fy = static_cast<float>(py) + 0.5f;  // 可以使用 HALF_128[0]
            pa2d::Color* row = &buffer.at(0, py);

            const __m256 py_v = _mm256_set1_ps(fy);
            const __m128 py_v_sse = _mm_set1_ps(fy);

            int px = minX;

            // --- AVX2处理 (8像素) ---
            for (; px <= maxX - 7; px += 8) {
                __m256i xBase = _mm256_setr_epi32(px, px + 1, px + 2, px + 3, px + 4, px + 5, px + 6, px + 7);
                __m256 px_v = _mm256_add_ps(_mm256_cvtepi32_ps(xBase), HALF_PIXEL_256);

                // A. 计算 SDF 距离
                // A.1. 逆向旋转像素
                __m256 dx_v = _mm256_sub_ps(px_v, centerX_v);
                __m256 dy_v = _mm256_sub_ps(py_v, centerY_v);
                __m256 x_local_v = _mm256_add_ps(_mm256_mul_ps(dx_v, c_v), _mm256_mul_ps(dy_v, s_v));
                __m256 y_local_v = _mm256_sub_ps(_mm256_mul_ps(dy_v, c_v), _mm256_mul_ps(dx_v, s_v));

                // A.2. 在局部坐标系中计算 AABB 圆角矩形 SDF
                __m256 dx_abs = _mm256_max_ps(_mm256_sub_ps(ZERO_256, x_local_v), x_local_v); // abs(x_local)
                __m256 dy_abs = _mm256_max_ps(_mm256_sub_ps(ZERO_256, y_local_v), y_local_v); // abs(y_local)

                __m256 ax = _mm256_sub_ps(dx_abs, sdf_w_v);
                __m256 ay = _mm256_sub_ps(dy_abs, sdf_h_v);

                __m256 cx = _mm256_max_ps(ax, ZERO_256);
                __m256 cy = _mm256_max_ps(ay, ZERO_256);

                __m256 dist_to_corner = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(cx, cx), _mm256_mul_ps(cy, cy)));

                __m256 sdf = _mm256_add_ps(
                    _mm256_min_ps(_mm256_max_ps(ax, ay), ZERO_256),
                    _mm256_sub_ps(dist_to_corner, cornerRadius_v)
                );

                // B. Alpha 计算
                __m256 effectiveFillAlpha = ZERO_256;
                if (drawFill) {
                    __m256 t_fill = _mm256_div_ps(_mm256_sub_ps(sdf, ANTIALIAS_RANGE_256), ANTIALIAS_RANGE_256);
                    __m256 fillAlpha_raw = _mm256_sub_ps(ONE_256, t_fill);
                    effectiveFillAlpha = _mm256_mul_ps(_mm256_max_ps(ZERO_256, _mm256_min_ps(ONE_256, fillAlpha_raw)), fillA_v);
                }

                __m256 effectiveStrokeAlpha = ZERO_256;
                if (drawStroke) {
                    __m256 distToStrokeCenter = _mm256_max_ps(_mm256_sub_ps(ZERO_256, sdf), sdf); // |sdf|
                    __m256 halfWidthMinusDist = _mm256_sub_ps(halfStrokeWidth_v, distToStrokeCenter);
                    __m256 t_stroke = _mm256_div_ps(halfWidthMinusDist, ANTIALIAS_RANGE_256);

                    __m256 strokeAlpha_raw = _mm256_max_ps(ZERO_256, _mm256_min_ps(ONE_256, t_stroke));
                    effectiveStrokeAlpha = _mm256_mul_ps(strokeAlpha_raw, strokeA_v);
                }

                // C. 颜色混合和写入
                __m256 finalAlpha;
                __m256 finalR, finalG, finalB;

                if (mode_stroke_over_fill) {
                    __m256 oneMinusEffectiveStrokeAlpha = _mm256_sub_ps(ONE_256, effectiveStrokeAlpha);
                    __m256 effectiveFillAlpha_modified = _mm256_mul_ps(effectiveFillAlpha, oneMinusEffectiveStrokeAlpha);
                    finalAlpha = _mm256_add_ps(effectiveStrokeAlpha, effectiveFillAlpha_modified);

                    finalR = _mm256_add_ps(_mm256_mul_ps(strokeR_v, effectiveStrokeAlpha), _mm256_mul_ps(fillR_v, effectiveFillAlpha_modified));
                    finalG = _mm256_add_ps(_mm256_mul_ps(strokeG_v, effectiveStrokeAlpha), _mm256_mul_ps(fillG_v, effectiveFillAlpha_modified));
                    finalB = _mm256_add_ps(_mm256_mul_ps(strokeB_v, effectiveStrokeAlpha), _mm256_mul_ps(fillB_v, effectiveFillAlpha_modified));

                    __m256 invFinalAlpha = _mm256_div_ps(ONE_256, finalAlpha);
                    __m256 zero_mask = _mm256_cmp_ps(finalAlpha, ZERO_256, _CMP_EQ_OQ);
                    invFinalAlpha = _mm256_andnot_ps(zero_mask, invFinalAlpha);

                    finalR = _mm256_mul_ps(finalR, invFinalAlpha);
                    finalG = _mm256_mul_ps(finalG, invFinalAlpha);
                    finalB = _mm256_mul_ps(finalB, invFinalAlpha);
                }
                else if (mode_only_stroke) {
                    finalAlpha = effectiveStrokeAlpha;
                    finalR = strokeR_v; finalG = strokeG_v; finalB = strokeB_v;
                }
                else { // mode_only_fill
                    finalAlpha = effectiveFillAlpha;
                    finalR = fillR_v; finalG = fillG_v; finalB = fillB_v;
                }

                __m256 mask = _mm256_cmp_ps(finalAlpha, ZERO_256, _CMP_GT_OQ);
                if (_mm256_testz_ps(mask, mask)) continue;

                __m256i dest = _mm256_loadu_si256((__m256i*) & row[px]);
                __m256i rgba = blend_pixels_avx(finalAlpha, dest, finalR, finalG, finalB);
                rgba = _mm256_blendv_epi8(dest, rgba, _mm256_castps_si256(mask));
                _mm256_storeu_si256((__m256i*) & row[px], rgba);
            }

            // --- SSE处理 (4像素) ---
            for (; px <= maxX - 3; px += 4) {
                __m128i xBase = _mm_setr_epi32(px, px + 1, px + 2, px + 3);
                __m128 px_v_sse = _mm_add_ps(_mm_cvtepi32_ps(xBase), HALF_PIXEL_128);

                // A. 计算 SSE SDF 距离
                __m128 dx_v = _mm_sub_ps(px_v_sse, centerX_sse);
                __m128 dy_v = _mm_sub_ps(py_v_sse, centerY_sse);
                __m128 x_local_v = _mm_add_ps(_mm_mul_ps(dx_v, c_sse), _mm_mul_ps(dy_v, s_sse));
                __m128 y_local_v = _mm_sub_ps(_mm_mul_ps(dy_v, c_sse), _mm_mul_ps(dx_v, s_sse));

                __m128 dx_abs = _mm_max_ps(_mm_sub_ps(ZERO_128, x_local_v), x_local_v);
                __m128 dy_abs = _mm_max_ps(_mm_sub_ps(ZERO_128, y_local_v), y_local_v);
                __m128 ax = _mm_sub_ps(dx_abs, sdf_w_sse);
                __m128 ay = _mm_sub_ps(dy_abs, sdf_h_sse);
                __m128 cx = _mm_max_ps(ax, ZERO_128);
                __m128 cy = _mm_max_ps(ay, ZERO_128);
                __m128 dist_to_corner = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(cx, cx), _mm_mul_ps(cy, cy)));
                __m128 sdf = _mm_add_ps(
                    _mm_min_ps(_mm_max_ps(ax, ay), ZERO_128),
                    _mm_sub_ps(dist_to_corner, cornerRadius_sse)
                );

                // B. Alpha 计算
                __m128 effectiveFillAlpha = ZERO_128;
                if (drawFill) {
                    __m128 t_fill = _mm_div_ps(_mm_sub_ps(sdf, ANTIALIAS_RANGE_128), ANTIALIAS_RANGE_128);
                    __m128 fillAlpha_raw = _mm_sub_ps(ONE_128, t_fill);
                    effectiveFillAlpha = _mm_mul_ps(_mm_max_ps(ZERO_128, _mm_min_ps(ONE_128, fillAlpha_raw)), fillA_sse);
                }

                __m128 effectiveStrokeAlpha = ZERO_128;
                if (drawStroke) {
                    __m128 distToStrokeCenter = _mm_max_ps(_mm_sub_ps(ZERO_128, sdf), sdf);
                    __m128 halfWidthMinusDist = _mm_sub_ps(halfStrokeWidth_sse, distToStrokeCenter);
                    __m128 t_stroke = _mm_div_ps(halfWidthMinusDist, ANTIALIAS_RANGE_128);
                    __m128 strokeAlpha_raw = _mm_max_ps(ZERO_128, _mm_min_ps(ONE_128, t_stroke));
                    effectiveStrokeAlpha = _mm_mul_ps(strokeAlpha_raw, strokeA_sse);
                }

                // C. 颜色混合和写入
                __m128 finalAlpha;
                __m128 finalR, finalG, finalB;

                if (mode_stroke_over_fill) {
                    __m128 oneMinusEffectiveStrokeAlpha = _mm_sub_ps(ONE_128, effectiveStrokeAlpha);
                    __m128 effectiveFillAlpha_modified = _mm_mul_ps(effectiveFillAlpha, oneMinusEffectiveStrokeAlpha);
                    finalAlpha = _mm_add_ps(effectiveStrokeAlpha, effectiveFillAlpha_modified);

                    finalR = _mm_add_ps(_mm_mul_ps(strokeR_sse, effectiveStrokeAlpha), _mm_mul_ps(fillR_sse, effectiveFillAlpha_modified));
                    finalG = _mm_add_ps(_mm_mul_ps(strokeG_sse, effectiveStrokeAlpha), _mm_mul_ps(fillG_sse, effectiveFillAlpha_modified));
                    finalB = _mm_add_ps(_mm_mul_ps(strokeB_sse, effectiveStrokeAlpha), _mm_mul_ps(fillB_sse, effectiveFillAlpha_modified));

                    __m128 invFinalAlpha = _mm_div_ps(ONE_128, finalAlpha);
                    __m128 zero_mask = _mm_cmpeq_ps(finalAlpha, ZERO_128);
                    invFinalAlpha = _mm_andnot_ps(zero_mask, invFinalAlpha);

                    finalR = _mm_mul_ps(finalR, invFinalAlpha);
                    finalG = _mm_mul_ps(finalG, invFinalAlpha);
                    finalB = _mm_mul_ps(finalB, invFinalAlpha);
                }
                else if (mode_only_stroke) {
                    finalAlpha = effectiveStrokeAlpha;
                    finalR = strokeR_sse; finalG = strokeG_sse; finalB = strokeB_sse;
                }
                else { // mode_only_fill
                    finalAlpha = effectiveFillAlpha;
                    finalR = fillR_sse; finalG = fillG_sse; finalB = fillB_sse;
                }

                __m128 mask = _mm_cmpgt_ps(finalAlpha, ZERO_128);
                if (_mm_movemask_ps(mask)) {
                    __m128i dest = _mm_loadu_si128((__m128i*) & row[px]);
                    __m128i rgba = blend_pixels_sse(finalAlpha, dest, finalR, finalG, finalB);
                    rgba = _mm_blendv_epi8(dest, rgba, _mm_castps_si128(mask));
                    _mm_storeu_si128((__m128i*) & row[px], rgba);
                }
            }

            // --- 标量收尾 (1-3 像素) ---
            for (; px <= maxX; ++px) {
                const float fx = static_cast<float>(px) + 0.5f;

                // A. 标量 SDF 距离计算
                const float dx = fx - centerX;
                const float dy = fy - centerY;
                const float x_local = dx * c + dy * s;
                const float y_local = -dx * s + dy * c;

                float dx_abs = std::abs(x_local);
                float dy_abs = std::abs(y_local);
                float ax = dx_abs - (halfWidth - cornerRadius);
                float ay = dy_abs - (halfHeight - cornerRadius);
                float cx = std::max(ax, 0.0f);
                float cy = std::max(ay, 0.0f);
                float sdf = std::min(std::max(ax, ay), 0.0f) + std::sqrt(cx * cx + cy * cy) - cornerRadius;

                // B. Alpha 计算
                float effectiveFillAlpha = 0.0f;
                if (drawFill) {
                    float t_fill = (sdf - antialiasRange) / antialiasRange;
                    float fillAlpha_raw = 1.0f - t_fill;
                    effectiveFillAlpha = std::max(0.0f, std::min(1.0f, fillAlpha_raw)) * finalFillOpacity;
                }

                float effectiveStrokeAlpha = 0.0f;
                if (drawStroke) {
                    float distToStrokeCenter = std::abs(sdf);
                    float halfWidthMinusDist = halfStrokeWidth - distToStrokeCenter;
                    float t_stroke = halfWidthMinusDist / antialiasRange;
                    float strokeAlpha_raw = std::max(0.0f, std::min(1.0f, t_stroke));
                    effectiveStrokeAlpha = strokeAlpha_raw * finalStrokeOpacity;
                }

                // C. 颜色混合和写入
                float finalAlpha;
                float R_src_pre, G_src_pre, B_src_pre;

                if (mode_stroke_over_fill) {
                    const float oneMinusEffectiveStrokeAlpha = 1.0f - effectiveStrokeAlpha;
                    const float effectiveFillAlpha_modified = effectiveFillAlpha * oneMinusEffectiveStrokeAlpha;
                    finalAlpha = effectiveStrokeAlpha + effectiveFillAlpha_modified;
                    R_src_pre = strokeColor.r * (effectiveStrokeAlpha / 255.0f) + fillColor.r * (effectiveFillAlpha_modified / 255.0f);
                    G_src_pre = strokeColor.g * (effectiveStrokeAlpha / 255.0f) + fillColor.g * (effectiveFillAlpha_modified / 255.0f);
                    B_src_pre = strokeColor.b * (effectiveStrokeAlpha / 255.0f) + fillColor.b * (effectiveFillAlpha_modified / 255.0f);
                }
                else if (mode_only_stroke) {
                    finalAlpha = effectiveStrokeAlpha;
                    R_src_pre = strokeColor.r * (effectiveStrokeAlpha / 255.0f);
                    G_src_pre = strokeColor.g * (effectiveStrokeAlpha / 255.0f);
                    B_src_pre = strokeColor.b * (effectiveStrokeAlpha / 255.0f);
                }
                else { // mode_only_fill
                    finalAlpha = effectiveFillAlpha;
                    R_src_pre = fillColor.r * (effectiveFillAlpha / 255.0f);
                    G_src_pre = fillColor.g * (effectiveFillAlpha / 255.0f);
                    B_src_pre = fillColor.b * (effectiveFillAlpha / 255.0f);
                }

                if (finalAlpha > 0.0f) {
                    pa2d::Color srcColor;
                    if (finalAlpha >= 1.0f) {
                        srcColor.r = static_cast<uint8_t>(std::min(255.0f, R_src_pre * 255.0f));
                        srcColor.g = static_cast<uint8_t>(std::min(255.0f, G_src_pre * 255.0f));
                        srcColor.b = static_cast<uint8_t>(std::min(255.0f, B_src_pre * 255.0f));
                        srcColor.a = 255;
                    }
                    else {
                        float invFinalAlpha = 1.0f / finalAlpha;
                        srcColor.r = static_cast<uint8_t>(std::min(255.0f, R_src_pre * invFinalAlpha * 255.0f));
                        srcColor.g = static_cast<uint8_t>(std::min(255.0f, G_src_pre * invFinalAlpha * 255.0f));
                        srcColor.b = static_cast<uint8_t>(std::min(255.0f, B_src_pre * invFinalAlpha * 255.0f));
                        srcColor.a = static_cast<uint8_t>(finalAlpha * 255.0f);
                    }
                    pa2d::Color& dest = row[px];
                    row[px] = Blend(srcColor, dest);
                }
            }
        }
    }
}