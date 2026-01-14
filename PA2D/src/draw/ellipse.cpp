#include"../include/draw.h"
#include"internal/blend_utils.h"

using namespace pa2d::utils;
using namespace pa2d::utils::simd;

namespace pa2d {
    void ellipse(
        pa2d::Buffer& buffer,
        float cx, float cy, float width, float height,
        const pa2d::Color& fillColor,
        const pa2d::Color& strokeColor,
        float strokeWidth
    ) {
        // --- 1. 参数校验和预处理 ---
        if (!buffer.isValid() || width <= 0 || height <= 0) return;

        const float finalFillOpacity = fillColor.a * (1.0f / 255.0f);
        const float finalStrokeOpacity = strokeColor.a * (1.0f / 255.0f);

        const bool drawFill = (finalFillOpacity > 0.0f);
        const bool drawStroke = (finalStrokeOpacity > 0.0f) && (strokeWidth > 0.0f);

        if (!drawFill && !drawStroke) return;

        const bool mode_stroke_over_fill = drawStroke && drawFill;
        const bool mode_only_stroke = drawStroke && !drawFill;
        const bool mode_only_fill = drawFill && !drawStroke;

        // 椭圆几何参数 (半轴)
        const float halfWidth = width * 0.5f;
        const float halfHeight = height * 0.5f;
        const float halfStrokeWidth = strokeWidth == 0 ? 0 : (strokeWidth + 1.0) * 0.5f;

        // --- 2. 边界计算和裁剪 ---
        const float maxExtX = halfWidth + halfStrokeWidth + 1.0f;
        const float maxExtY = halfHeight + halfStrokeWidth + 1.0f;

        // 基于中心点和最大扩展范围计算边界
        int minX = static_cast<int>(std::floor(cx - maxExtX));
        int maxX = static_cast<int>(std::ceil(cx + maxExtX));
        int minY = static_cast<int>(std::floor(cy - maxExtY));
        int maxY = static_cast<int>(std::ceil(cy + maxExtY));

        minX = std::max(0, minX);
        maxX = std::min(buffer.width - 1, maxX);
        minY = std::max(0, minY);
        maxY = std::min(buffer.height - 1, maxY);

        if (minX > maxX || minY > maxY) return;

        // --- 3. SIMD 常量准备 (AVX2) ---
        // 几何常量 (用于 SDF)
        const __m256 cx_v = _mm256_set1_ps(cx);
        const __m256 cy_v = _mm256_set1_ps(cy);
        const __m256 halfWidth_v = _mm256_set1_ps(halfWidth);
        const __m256 halfHeight_v = _mm256_set1_ps(halfHeight);
        const __m256 A2_v = _mm256_mul_ps(halfWidth_v, halfWidth_v);
        const __m256 B2_v = _mm256_mul_ps(halfHeight_v, halfHeight_v);
        const __m256 A4_v = _mm256_mul_ps(A2_v, A2_v);
        const __m256 B4_v = _mm256_mul_ps(B2_v, B2_v);

        const __m256 A2_inv_v = _mm256_div_ps(ONE_256, A2_v);
        const __m256 B2_inv_v = _mm256_div_ps(ONE_256, B2_v);
        const __m256 A4_inv_v = _mm256_div_ps(ONE_256, A4_v);
        const __m256 B4_inv_v = _mm256_div_ps(ONE_256, B4_v);

        const __m256 halfStrokeWidth_v = _mm256_set1_ps(halfStrokeWidth);

        // 颜色常量 (AVX2)
        const __m256 fillA_v = _mm256_set1_ps(finalFillOpacity);
        const __m256 strokeA_v = _mm256_set1_ps(finalStrokeOpacity);
        const __m256 fillR_v = _mm256_set1_ps(fillColor.r * (1.0f / 255.0f));
        const __m256 fillG_v = _mm256_set1_ps(fillColor.g * (1.0f / 255.0f));
        const __m256 fillB_v = _mm256_set1_ps(fillColor.b * (1.0f / 255.0f));
        const __m256 strokeR_v = _mm256_set1_ps(strokeColor.r * (1.0f / 255.0f));
        const __m256 strokeG_v = _mm256_set1_ps(strokeColor.g * (1.0f / 255.0f));
        const __m256 strokeB_v = _mm256_set1_ps(strokeColor.b * (1.0f / 255.0f));

        // --- 4. SIMD 常量准备 (SSE) ---
        const __m128 cx_sse = _mm_set1_ps(cx);
        const __m128 cy_sse = _mm_set1_ps(cy);
        const __m128 halfWidth_sse = _mm_set1_ps(halfWidth);
        const __m128 halfHeight_sse = _mm_set1_ps(halfHeight);
        const __m128 A2_sse = _mm_mul_ps(halfWidth_sse, halfWidth_sse);
        const __m128 B2_sse = _mm_mul_ps(halfHeight_sse, halfHeight_sse);
        const __m128 A4_sse = _mm_mul_ps(A2_sse, A2_sse);
        const __m128 B4_sse = _mm_mul_ps(B2_sse, B2_sse);

        const __m128 A2_inv_sse = _mm_div_ps(ONE_128, A2_sse);
        const __m128 B2_inv_sse = _mm_div_ps(ONE_128, B2_sse);
        const __m128 A4_inv_sse = _mm_div_ps(ONE_128, A4_sse);
        const __m128 B4_inv_sse = _mm_div_ps(ONE_128, B4_sse);

        const __m128 halfStrokeWidth_sse = _mm_set1_ps(halfStrokeWidth);

        const __m128 fillA_sse = _mm_set1_ps(finalFillOpacity);
        const __m128 strokeA_sse = _mm_set1_ps(finalStrokeOpacity);
        const __m128 fillR_sse = _mm_set1_ps(fillColor.r * (1.0f / 255.0f));
        const __m128 fillG_sse = _mm_set1_ps(fillColor.g * (1.0f / 255.0f));
        const __m128 fillB_sse = _mm_set1_ps(fillColor.b * (1.0f / 255.0f));
        const __m128 strokeR_sse = _mm_set1_ps(strokeColor.r * (1.0f / 255.0f));
        const __m128 strokeG_sse = _mm_set1_ps(strokeColor.g * (1.0f / 255.0f));
        const __m128 strokeB_sse = _mm_set1_ps(strokeColor.b * (1.0f / 255.0f));

        // --- 5. 主绘制循环 ---
        for (int py = minY; py <= maxY; ++py) {
            const float fy = static_cast<float>(py) + 0.5f;
            pa2d::Color* row = &buffer.at(0, py);

            const __m256 py_v = _mm256_set1_ps(fy);
            const __m128 py_v_sse = _mm_set1_ps(fy);

            int px = minX;

            // --- AVX2处理 (8像素) ---
            for (; px <= maxX - 7; px += 8) {
                __m256i xBase = _mm256_setr_epi32(px, px + 1, px + 2, px + 3, px + 4, px + 5, px + 6, px + 7);
                __m256 px_v = _mm256_add_ps(_mm256_cvtepi32_ps(xBase), HALF_PIXEL_256);

                // A. 计算 SDF 距离
                __m256 dx = _mm256_sub_ps(px_v, cx_v);
                __m256 dy = _mm256_sub_ps(py_v, cy_v);
                __m256 dx2 = _mm256_mul_ps(dx, dx);
                __m256 dy2 = _mm256_mul_ps(dy, dy);

                // 椭圆函数 F(x, y) = x^2/A^2 + y^2/B^2 - 1
                __m256 F = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(dx2, A2_inv_v), _mm256_mul_ps(dy2, B2_inv_v)), ONE_256);

                // 梯度 Gx = 2x/A^2, Gy = 2y/B^2. 长度 |grad F| = sqrt( (2x/A^2)^2 + (2y/B^2)^2 )
                // 使用简化形式 |grad F| = 2 * sqrt( x^2/A^4 + y^2/B^4 )
                __m256 grad_sq = _mm256_add_ps(_mm256_mul_ps(dx2, A4_inv_v), _mm256_mul_ps(dy2, B4_inv_v));
                __m256 grad_length = _mm256_mul_ps(_mm256_set1_ps(2.0f), _mm256_sqrt_ps(grad_sq));

                // SDF 近似公式: F / |grad F|
                // 避免除以零 (如果 grad_length 接近 0)
                __m256 safe_grad_length = _mm256_max_ps(grad_length, _mm256_set1_ps(GEOMETRY_EPSILON));
                __m256 sdf = _mm256_div_ps(F, safe_grad_length);

                // B. Alpha 计算 (SDF -> Alpha)
                __m256 effectiveFillAlpha = ZERO_256;
                if (drawFill) {
                    // Fill Alpha: sdf 越小 (内部) alpha 越大
                    __m256 t_fill = _mm256_div_ps(_mm256_sub_ps(sdf, ANTIALIAS_RANGE_256), ANTIALIAS_RANGE_256);
                    __m256 fillAlpha_raw = _mm256_sub_ps(ONE_256, t_fill);
                    effectiveFillAlpha = _mm256_mul_ps(_mm256_max_ps(ZERO_256, _mm256_min_ps(ONE_256, fillAlpha_raw)), fillA_v);
                }

                __m256 effectiveStrokeAlpha = ZERO_256;
                if (drawStroke) {
                    // Stroke Alpha: |sdf| 越接近 halfStrokeWidth 越实心
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
                __m128 dx = _mm_sub_ps(px_v_sse, cx_sse);
                __m128 dy = _mm_sub_ps(py_v_sse, cy_sse);
                __m128 dx2 = _mm_mul_ps(dx, dx);
                __m128 dy2 = _mm_mul_ps(dy, dy);

                __m128 F = _mm_sub_ps(_mm_add_ps(_mm_mul_ps(dx2, A2_inv_sse), _mm_mul_ps(dy2, B2_inv_sse)), ONE_128);

                __m128 grad_sq = _mm_add_ps(_mm_mul_ps(dx2, A4_inv_sse), _mm_mul_ps(dy2, B4_inv_sse));
                __m128 grad_length = _mm_mul_ps(_mm_set1_ps(2.0f), _mm_sqrt_ps(grad_sq));

                __m128 safe_grad_length = _mm_max_ps(grad_length, _mm_set1_ps(GEOMETRY_EPSILON));
                __m128 sdf = _mm_div_ps(F, safe_grad_length);

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
                    __m128 zero_mask_eq = _mm_cmpeq_ps(finalAlpha, ZERO_128);
                    invFinalAlpha = _mm_andnot_ps(zero_mask_eq, invFinalAlpha);

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

            // --- 标量收尾 ---
            for (; px <= maxX; ++px) {
                const float fx = static_cast<float>(px) + 0.5f;

                // A. 标量 SDF 距离计算
                float dx = fx - cx;
                float dy = fy - cy;

                float A = halfWidth;
                float B = halfHeight;
                float A2 = A * A;
                float B2 = B * B;

                // F(x, y) = x^2/A^2 + y^2/B^2 - 1
                float F = (dx * dx / A2) + (dy * dy / B2) - 1.0f;

                // |grad F| = 2 * sqrt( x^2/A^4 + y^2/B^4 )
                float grad_sq = (dx * dx / (A2 * A2)) + (dy * dy / (B2 * B2));
                float grad_length = 2.0f * std::sqrt(grad_sq);

                // SDF 近似公式: F / |grad F|
                float safe_grad_length = std::max(grad_length, GEOMETRY_EPSILON);
                float sdf = F / safe_grad_length;

                // B. Alpha 计算
                float effectiveFillAlpha = 0.0f;
                if (drawFill) {
                    float t_fill = (sdf - 1.0f) / 1.0f;
                    float fillAlpha_raw = 1.0f - t_fill;
                    effectiveFillAlpha = std::max(0.0f, std::min(1.0f, fillAlpha_raw)) * finalFillOpacity;
                }

                float effectiveStrokeAlpha = 0.0f;
                if (drawStroke) {
                    float distToStrokeCenter = std::abs(sdf);
                    float halfWidthMinusDist = halfStrokeWidth - distToStrokeCenter;
                    float t_stroke = halfWidthMinusDist / 1.0f;

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


    void ellipse(
        pa2d::Buffer& buffer,
        float cx, float cy, float width, float height, float angle,
        const pa2d::Color& fillColor,
        const pa2d::Color& strokeColor,
        float strokeWidth
    ) {
        const float EPSILON = GEOMETRY_EPSILON;
        if (std::abs(width - height) < EPSILON) {
            circle(buffer, cx, cy, width * 0.5f, fillColor, strokeColor, strokeWidth);
            return;
        }
        // --- 1. 参数校验和预处理 ---
        if (!buffer.isValid() || width <= 0 || height <= 0) return;

        const float finalFillOpacity = fillColor.a * (1.0f / 255.0f);
        const float finalStrokeOpacity = strokeColor.a * (1.0f / 255.0f);

        const bool drawFill = (finalFillOpacity > 0.0f);
        const bool drawStroke = (finalStrokeOpacity > 0.0f) && (strokeWidth > 0.0f);

        if (!drawFill && !drawStroke) return;

        const bool mode_stroke_over_fill = drawStroke && drawFill;
        const bool mode_only_stroke = drawStroke && !drawFill;
        const bool mode_only_fill = drawFill && !drawStroke;

        // 椭圆几何参数 (半轴)
        const float halfWidth = width * 0.5f;
        const float halfHeight = height * 0.5f;
        const float halfStrokeWidth = strokeWidth == 0 ? 0 : (strokeWidth + 1.0) * 0.5f;
        angle = angle * (GEOMETRY_PI / 180.0f);

        // --- 2. 旋转和包围盒 (Bounding Box) 计算 ---
        const float s = std::sin(angle);
        const float c = std::cos(angle);
        const float abs_s = std::abs(s);
        const float abs_c = std::abs(c);

        // 计算包含描边和 AA 的安全半轴
        const float safeHalfWidth = halfWidth + halfStrokeWidth + 1.0f;
        const float safeHalfHeight = halfHeight + halfStrokeWidth + 1.0f;

        // 计算旋转后椭圆的世界空间包围盒 (AABB)
        const float maxExtX = abs_c * safeHalfWidth + abs_s * safeHalfHeight;
        const float maxExtY = abs_s * safeHalfWidth + abs_c * safeHalfHeight;

        // 基于中心点和最大扩展范围计算边界
        int minX = static_cast<int>(std::floor(cx - maxExtX));
        int maxX = static_cast<int>(std::ceil(cx + maxExtX));
        int minY = static_cast<int>(std::floor(cy - maxExtY));
        int maxY = static_cast<int>(std::ceil(cy + maxExtY));

        minX = std::max(0, minX);
        maxX = std::min(buffer.width - 1, maxX);
        minY = std::max(0, minY);
        maxY = std::min(buffer.height - 1, maxY);

        if (minX > maxX || minY > maxY) return;

        // --- 3. SIMD 常量准备 (AVX2) ---
        const __m256 cx_v = _mm256_set1_ps(cx);
        const __m256 cy_v = _mm256_set1_ps(cy);
        const __m256 halfWidth_v = _mm256_set1_ps(halfWidth);
        const __m256 halfHeight_v = _mm256_set1_ps(halfHeight);
        const __m256 A2_v = _mm256_mul_ps(halfWidth_v, halfWidth_v);
        const __m256 B2_v = _mm256_mul_ps(halfHeight_v, halfHeight_v);
        const __m256 A4_v = _mm256_mul_ps(A2_v, A2_v);
        const __m256 B4_v = _mm256_mul_ps(B2_v, B2_v);

        const __m256 A2_inv_v = _mm256_div_ps(ONE_256, A2_v);
        const __m256 B2_inv_v = _mm256_div_ps(ONE_256, B2_v);
        const __m256 A4_inv_v = _mm256_div_ps(ONE_256, A4_v);
        const __m256 B4_inv_v = _mm256_div_ps(ONE_256, B4_v);

        const __m256 halfStrokeWidth_v = _mm256_set1_ps(halfStrokeWidth);

        // [新增] 旋转常量
        const __m256 s_v = _mm256_set1_ps(s);
        const __m256 c_v = _mm256_set1_ps(c);

        // 颜色常量 (AVX2)
        const __m256 fillA_v = _mm256_set1_ps(finalFillOpacity);
        const __m256 strokeA_v = _mm256_set1_ps(finalStrokeOpacity);
        const __m256 fillR_v = _mm256_set1_ps(fillColor.r * (1.0f / 255.0f));
        const __m256 fillG_v = _mm256_set1_ps(fillColor.g * (1.0f / 255.0f));
        const __m256 fillB_v = _mm256_set1_ps(fillColor.b * (1.0f / 255.0f));
        const __m256 strokeR_v = _mm256_set1_ps(strokeColor.r * (1.0f / 255.0f));
        const __m256 strokeG_v = _mm256_set1_ps(strokeColor.g * (1.0f / 255.0f));
        const __m256 strokeB_v = _mm256_set1_ps(strokeColor.b * (1.0f / 255.0f));

        // --- 4. SIMD 常量准备 (SSE) ---
        const __m128 cx_sse = _mm_set1_ps(cx);
        const __m128 cy_sse = _mm_set1_ps(cy);
        const __m128 halfWidth_sse = _mm_set1_ps(halfWidth);
        const __m128 halfHeight_sse = _mm_set1_ps(halfHeight);
        const __m128 A2_sse = _mm_mul_ps(halfWidth_sse, halfWidth_sse);
        const __m128 B2_sse = _mm_mul_ps(halfHeight_sse, halfHeight_sse);
        const __m128 A4_sse = _mm_mul_ps(A2_sse, A2_sse);
        const __m128 B4_sse = _mm_mul_ps(B2_sse, B2_sse);

        const __m128 A2_inv_sse = _mm_div_ps(ONE_128, A2_sse);
        const __m128 B2_inv_sse = _mm_div_ps(ONE_128, B2_sse);
        const __m128 A4_inv_sse = _mm_div_ps(ONE_128, A4_sse);
        const __m128 B4_inv_sse = _mm_div_ps(ONE_128, B4_sse);

        const __m128 halfStrokeWidth_sse = _mm_set1_ps(halfStrokeWidth);

        // [新增] 旋转常量
        const __m128 s_sse = _mm_set1_ps(s);
        const __m128 c_sse = _mm_set1_ps(c);

        // 颜色常量 (SSE)
        const __m128 fillA_sse = _mm_set1_ps(finalFillOpacity);
        const __m128 strokeA_sse = _mm_set1_ps(finalStrokeOpacity);
        const __m128 fillR_sse = _mm_set1_ps(fillColor.r * (1.0f / 255.0f));
        const __m128 fillG_sse = _mm_set1_ps(fillColor.g * (1.0f / 255.0f));
        const __m128 fillB_sse = _mm_set1_ps(fillColor.b * (1.0f / 255.0f));
        const __m128 strokeR_sse = _mm_set1_ps(strokeColor.r * (1.0f / 255.0f));
        const __m128 strokeG_sse = _mm_set1_ps(strokeColor.g * (1.0f / 255.0f));
        const __m128 strokeB_sse = _mm_set1_ps(strokeColor.b * (1.0f / 255.0f));

        // --- 5. 主绘制循环 ---
        for (int py = minY; py <= maxY; ++py) {
            const float fy = static_cast<float>(py) + 0.5f;
            pa2d::Color* row = &buffer.at(0, py);

            const __m256 py_v = _mm256_set1_ps(fy);
            const __m128 py_v_sse = _mm_set1_ps(fy);

            int px = minX;

            // --- AVX2处理 (8像素) ---
            for (; px <= maxX - 7; px += 8) {
                __m256i xBase = _mm256_setr_epi32(px, px + 1, px + 2, px + 3, px + 4, px + 5, px + 6, px + 7);
                __m256 px_v = _mm256_add_ps(_mm256_cvtepi32_ps(xBase), HALF_PIXEL_256);

                // A. 计算 SDF 距离
                __m256 dx = _mm256_sub_ps(px_v, cx_v);
                __m256 dy = _mm256_sub_ps(py_v, cy_v);

                // 逆向旋转像素
                __m256 x_local_v = _mm256_add_ps(_mm256_mul_ps(dx, c_v), _mm256_mul_ps(dy, s_v));
                __m256 y_local_v = _mm256_sub_ps(_mm256_mul_ps(dy, c_v), _mm256_mul_ps(dx, s_v));

                // 在局部坐标系中计算轴对齐椭圆 SDF
                __m256 dx2 = _mm256_mul_ps(x_local_v, x_local_v);
                __m256 dy2 = _mm256_mul_ps(y_local_v, y_local_v);

                __m256 F = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(dx2, A2_inv_v), _mm256_mul_ps(dy2, B2_inv_v)), ONE_256);

                __m256 grad_sq = _mm256_add_ps(_mm256_mul_ps(dx2, A4_inv_v), _mm256_mul_ps(dy2, B4_inv_v));
                __m256 grad_length = _mm256_mul_ps(_mm256_set1_ps(2.0f), _mm256_sqrt_ps(grad_sq));

                __m256 safe_grad_length = _mm256_max_ps(grad_length, _mm256_set1_ps(GEOMETRY_EPSILON));
                __m256 sdf = _mm256_div_ps(F, safe_grad_length);

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
            for (; px <= maxX - 3; px += 4) {
                __m128i xBase = _mm_setr_epi32(px, px + 1, px + 2, px + 3);
                __m128 px_v_sse = _mm_add_ps(_mm_cvtepi32_ps(xBase), HALF_PIXEL_128);

                // A. 计算 SSE SDF 距离
                __m128 dx = _mm_sub_ps(px_v_sse, cx_sse);
                __m128 dy = _mm_sub_ps(py_v_sse, cy_sse);

                // 逆向旋转
                __m128 x_local_v = _mm_add_ps(_mm_mul_ps(dx, c_sse), _mm_mul_ps(dy, s_sse));
                __m128 y_local_v = _mm_sub_ps(_mm_mul_ps(dy, c_sse), _mm_mul_ps(dx, s_sse));

                // 局部 SDF
                __m128 dx2 = _mm_mul_ps(x_local_v, x_local_v);
                __m128 dy2 = _mm_mul_ps(y_local_v, y_local_v);

                __m128 F = _mm_sub_ps(_mm_add_ps(_mm_mul_ps(dx2, A2_inv_sse), _mm_mul_ps(dy2, B2_inv_sse)), ONE_128);

                __m128 grad_sq = _mm_add_ps(_mm_mul_ps(dx2, A4_inv_sse), _mm_mul_ps(dy2, B4_inv_sse));
                __m128 grad_length = _mm_mul_ps(_mm_set1_ps(2.0f), _mm_sqrt_ps(grad_sq));

                __m128 safe_grad_length = _mm_max_ps(grad_length, _mm_set1_ps(GEOMETRY_EPSILON));
                __m128 sdf = _mm_div_ps(F, safe_grad_length);

                // B. Alpha 计算
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
                    __m128 zero_mask_eq = _mm_cmpeq_ps(finalAlpha, ZERO_128);
                    invFinalAlpha = _mm_andnot_ps(zero_mask_eq, invFinalAlpha);

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

            // --- 标量收尾 ---
            for (; px <= maxX; ++px) {
                const float fx = static_cast<float>(px) + 0.5f;

                // A. 标量 SDF 距离计算
                float dx = fx - cx;
                float dy = fy - cy;
                float x_local = dx * c + dy * s;
                float y_local = -dx * s + dy * c;

                // A.2. 局部 SDF
                float A = halfWidth;
                float B = halfHeight;
                float A2 = A * A;
                float B2 = B * B;

                float F = (x_local * x_local / A2) + (y_local * y_local / B2) - 1.0f;
                float grad_sq = (x_local * x_local / (A2 * A2)) + (y_local * y_local / (B2 * B2));
                float grad_length = 2.0f * std::sqrt(grad_sq);
                float safe_grad_length = std::max(grad_length, GEOMETRY_EPSILON);
                float sdf = F / safe_grad_length;

                // B. Alpha 计算
                float effectiveFillAlpha = 0.0f;
                if (drawFill) {
                    float t_fill = (sdf - 1.0f) / 1.0f;
                    float fillAlpha_raw = 1.0f - t_fill;
                    effectiveFillAlpha = std::max(0.0f, std::min(1.0f, fillAlpha_raw)) * finalFillOpacity;
                }

                float effectiveStrokeAlpha = 0.0f;
                if (drawStroke) {
                    float distToStrokeCenter = std::abs(sdf);
                    float halfWidthMinusDist = halfStrokeWidth - distToStrokeCenter;
                    float t_stroke = halfWidthMinusDist / 1.0f;
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