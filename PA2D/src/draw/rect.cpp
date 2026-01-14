#include"../include/draw.h"
#include"internal/blend_utils.h"

using namespace pa2d::utils;
using namespace pa2d::utils::simd;

namespace pa2d {

    void rect(
        pa2d::Buffer& buffer,
        float x, float y, float width, float height,
        const pa2d::Color& fillColor,
        const pa2d::Color& strokeColor,
        float strokeWidth
    ) {
        if (!buffer.isValid() || width <= 0 || height <= 0) return;

        // --- 1. 预处理与模式判断 ---
        const float finalFillOpacity = fillColor.a * (1.0f / 255.0f);
        const float finalStrokeOpacity = strokeColor.a * (1.0f / 255.0f);
        const bool drawFill = (finalFillOpacity > 0.0f);
        const bool drawStroke = (finalStrokeOpacity > 0.0f) && (strokeWidth > 0.0f);

        if (!drawFill && !drawStroke) return;

        const bool mode_stroke_over_fill = drawStroke && drawFill;
        const bool mode_only_stroke = drawStroke && !drawFill;

        // --- 2. 几何参数 ---
        const float centerX = x + width * 0.5f;
        const float centerY = y + height * 0.5f;
        const float halfWidth = width * 0.5f;
        const float halfHeight = height * 0.5f;

        // 使用全局常量
        float antialiasRange = 1.0f;

        const float halfStrokeWidth = strokeWidth == 0 ? 0 : (strokeWidth + 1.0f) * 0.5f;

        const float invAntialiasRange = 1.0f / antialiasRange;

        // --- 3. 边界裁剪 ---
        const float maxExtX = halfWidth + halfStrokeWidth + antialiasRange;
        const float maxExtY = halfHeight + halfStrokeWidth + antialiasRange;

        int minX = std::max(0, static_cast<int>(std::floor(centerX - maxExtX)));
        int maxX = std::min(buffer.width - 1, static_cast<int>(std::ceil(centerX + maxExtX)));
        int minY = std::max(0, static_cast<int>(std::floor(centerY - maxExtY)));
        int maxY = std::min(buffer.height - 1, static_cast<int>(std::ceil(centerY + maxExtY)));

        if (minX > maxX || minY > maxY) return;

        // --- 4. SIMD 常量 ---
        const __m256 centerX_v = _mm256_set1_ps(centerX);
        const __m256 centerY_v = _mm256_set1_ps(centerY);
        const __m256 halfWidth_v = _mm256_set1_ps(halfWidth);
        const __m256 halfHeight_v = _mm256_set1_ps(halfHeight);
        const __m256 halfStrokeWidth_v = _mm256_set1_ps(halfStrokeWidth);
        const __m256 invAARange_v = _mm256_set1_ps(invAntialiasRange);

        // 预乘颜色的 Alpha (0.0 - 1.0)
        const __m256 fillA_v = _mm256_set1_ps(finalFillOpacity);
        const __m256 strokeA_v = _mm256_set1_ps(finalStrokeOpacity);

        // 预先计算好 RGB 分量 (归一化 0-1)
        const __m256 fillR_v = _mm256_set1_ps(fillColor.r * (1.0f / 255.0f));
        const __m256 fillG_v = _mm256_set1_ps(fillColor.g * (1.0f / 255.0f));
        const __m256 fillB_v = _mm256_set1_ps(fillColor.b * (1.0f / 255.0f));
        const __m256 strokeR_v = _mm256_set1_ps(strokeColor.r * (1.0f / 255.0f));
        const __m256 strokeG_v = _mm256_set1_ps(strokeColor.g * (1.0f / 255.0f));
        const __m256 strokeB_v = _mm256_set1_ps(strokeColor.b * (1.0f / 255.0f));

        // SSE 常量
        const __m128 centerX_sse = _mm_set1_ps(centerX);
        const __m128 centerY_sse = _mm_set1_ps(centerY);
        const __m128 halfWidth_sse = _mm_set1_ps(halfWidth);
        const __m128 halfHeight_sse = _mm_set1_ps(halfHeight);
        const __m128 halfStrokeWidth_sse = _mm_set1_ps(halfStrokeWidth);
        const __m128 invAARange_sse = _mm_set1_ps(invAntialiasRange);
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

            // >>> AVX2 处理 (8像素) <<<
            for (; px <= maxX - 7; px += 8) {
                __m256i xBase = _mm256_setr_epi32(px, px + 1, px + 2, px + 3, px + 4, px + 5, px + 6, px + 7);
                __m256 px_v = _mm256_add_ps(_mm256_cvtepi32_ps(xBase), HALF_PIXEL_256);

                // A. SDF 计算
                __m256 dx_abs = _mm256_sub_ps(_mm256_max_ps(px_v, centerX_v), _mm256_min_ps(px_v, centerX_v));
                __m256 dy_abs = _mm256_sub_ps(_mm256_max_ps(py_v, centerY_v), _mm256_min_ps(py_v, centerY_v));

                // 矩形有符号距离 (Signed Distance)
                __m256 d_x = _mm256_sub_ps(dx_abs, halfWidth_v);
                __m256 d_y = _mm256_sub_ps(dy_abs, halfHeight_v);
                // 这里的 SDF：外部 > 0，内部 < 0
                __m256 sdf = _mm256_max_ps(d_x, d_y);

                // B. 计算 Alpha
                __m256 effectiveStrokeAlpha = ZERO_256;
                __m256 effectiveFillAlpha = ZERO_256;

                if (drawStroke) {
                    // Stroke SDF: 距离矩形边缘的绝对距离
                    // abs(sdf) 越小，说明越靠近边缘
                    __m256 distToEdge = _mm256_max_ps(sdf, _mm256_sub_ps(ZERO_256, sdf)); // abs(sdf)

                    // 线性衰减：(halfStrokeWidth - dist) / aaRange
                    // 正数表示在描边内，负数表示在描边外
                    __m256 rawAlpha = _mm256_sub_ps(halfStrokeWidth_v, distToEdge);

                    // 使用全局常量
                    effectiveStrokeAlpha = _mm256_mul_ps(_mm256_max_ps(ZERO_256, _mm256_min_ps(ONE_256, _mm256_mul_ps(rawAlpha, invAARange_v))), strokeA_v);
                }

                if (drawFill) {
                    // Fill SDF: sdf 越小越内部
                    // 线性衰减：(0 - sdf) / aaRange -> -sdf / aaRange
                    __m256 rawAlpha = _mm256_sub_ps(ZERO_256, sdf);
                    effectiveFillAlpha = _mm256_mul_ps(_mm256_max_ps(ZERO_256, _mm256_min_ps(ONE_256, _mm256_mul_ps(rawAlpha, invAARange_v))), fillA_v);
                }

                // C. 混合逻辑
                __m256 finalAlpha, finalR, finalG, finalB;

                if (mode_stroke_over_fill) {
                    // 标准混合公式：Out = Stroke + Fill * (1 - StrokeAlpha)
                    __m256 invStrokeA = _mm256_sub_ps(ONE_256, effectiveStrokeAlpha);
                    __m256 modFillA = _mm256_mul_ps(effectiveFillAlpha, invStrokeA);

                    finalAlpha = _mm256_add_ps(effectiveStrokeAlpha, modFillA);
                    finalR = _mm256_add_ps(_mm256_mul_ps(strokeR_v, effectiveStrokeAlpha), _mm256_mul_ps(fillR_v, modFillA));
                    finalG = _mm256_add_ps(_mm256_mul_ps(strokeG_v, effectiveStrokeAlpha), _mm256_mul_ps(fillG_v, modFillA));
                    finalB = _mm256_add_ps(_mm256_mul_ps(strokeB_v, effectiveStrokeAlpha), _mm256_mul_ps(fillB_v, modFillA));

                    // 反预乘：颜色 / Alpha (避免除0)
                    __m256 maskPos = _mm256_cmp_ps(finalAlpha, ZERO_256, _CMP_GT_OQ);
                    __m256 rcpAlpha = _mm256_div_ps(ONE_256, finalAlpha);
                    // 如果 Alpha 极小，保持原值或设为0 (这里通过掩码处理)
                    finalR = _mm256_mul_ps(finalR, rcpAlpha);
                    finalG = _mm256_mul_ps(finalG, rcpAlpha);
                    finalB = _mm256_mul_ps(finalB, rcpAlpha);
                }
                else if (mode_only_stroke) {
                    finalAlpha = effectiveStrokeAlpha;
                    finalR = strokeR_v; finalG = strokeG_v; finalB = strokeB_v;
                }
                else {
                    finalAlpha = effectiveFillAlpha;
                    finalR = fillR_v; finalG = fillG_v; finalB = fillB_v;
                }

                // D. 写入内存
                __m256 mask = _mm256_cmp_ps(finalAlpha, ZERO_256, _CMP_GT_OQ);
                if (!_mm256_testz_ps(mask, mask)) {
                    __m256i dest = _mm256_loadu_si256((__m256i*) & row[px]);
                    __m256i rgba = blend_pixels_avx(finalAlpha, dest, finalR, finalG, finalB);
                    rgba = _mm256_blendv_epi8(dest, rgba, _mm256_castps_si256(mask));
                    _mm256_storeu_si256((__m256i*) & row[px], rgba);
                }
            }

            // >>> SSE 处理 (4像素) <<<
            for (; px <= maxX - 3; px += 4) {
                __m128i xBase = _mm_setr_epi32(px, px + 1, px + 2, px + 3);
                __m128 px_v = _mm_add_ps(_mm_cvtepi32_ps(xBase), HALF_PIXEL_128);

                __m128 dx_abs = _mm_sub_ps(_mm_max_ps(px_v, centerX_sse), _mm_min_ps(px_v, centerX_sse));
                __m128 dy_abs = _mm_sub_ps(_mm_max_ps(py_v_sse, centerY_sse), _mm_min_ps(py_v_sse, centerY_sse));
                __m128 sdf = _mm_max_ps(_mm_sub_ps(dx_abs, halfWidth_sse), _mm_sub_ps(dy_abs, halfHeight_sse));

                __m128 effectiveStrokeAlpha = ZERO_128;
                __m128 effectiveFillAlpha = ZERO_128;

                if (drawStroke) {
                    __m128 distToEdge = _mm_max_ps(sdf, _mm_sub_ps(ZERO_128, sdf));
                    __m128 rawAlpha = _mm_sub_ps(halfStrokeWidth_sse, distToEdge);
                    effectiveStrokeAlpha = _mm_mul_ps(_mm_max_ps(ZERO_128, _mm_min_ps(ONE_128, _mm_mul_ps(rawAlpha, invAARange_sse))), strokeA_sse);
                }
                if (drawFill) {
                    __m128 rawAlpha = _mm_sub_ps(ZERO_128, sdf);
                    effectiveFillAlpha = _mm_mul_ps(_mm_max_ps(ZERO_128, _mm_min_ps(ONE_128, _mm_mul_ps(rawAlpha, invAARange_sse))), fillA_sse);
                }

                __m128 finalAlpha, finalR, finalG, finalB;
                if (mode_stroke_over_fill) {
                    __m128 invStrokeA = _mm_sub_ps(ONE_128, effectiveStrokeAlpha);
                    __m128 modFillA = _mm_mul_ps(effectiveFillAlpha, invStrokeA);
                    finalAlpha = _mm_add_ps(effectiveStrokeAlpha, modFillA);
                    finalR = _mm_add_ps(_mm_mul_ps(strokeR_sse, effectiveStrokeAlpha), _mm_mul_ps(fillR_sse, modFillA));
                    finalG = _mm_add_ps(_mm_mul_ps(strokeG_sse, effectiveStrokeAlpha), _mm_mul_ps(fillG_sse, modFillA));
                    finalB = _mm_add_ps(_mm_mul_ps(strokeB_sse, effectiveStrokeAlpha), _mm_mul_ps(fillB_sse, modFillA));

                    __m128 maskPos = _mm_cmpgt_ps(finalAlpha, ZERO_128);
                    __m128 rcpAlpha = _mm_div_ps(ONE_128, finalAlpha);
                    // 仅对 valid alpha 做除法，虽然 SIMD 并行可能需要全部算，但掩码最后会遮住
                    finalR = _mm_mul_ps(finalR, rcpAlpha);
                    finalG = _mm_mul_ps(finalG, rcpAlpha);
                    finalB = _mm_mul_ps(finalB, rcpAlpha);
                }
                else if (mode_only_stroke) {
                    finalAlpha = effectiveStrokeAlpha;
                    finalR = strokeR_sse; finalG = strokeG_sse; finalB = strokeB_sse;
                }
                else {
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

            // >>> 标量收尾 (Scalar) <<<
            for (; px <= maxX; ++px) {
                float fx = static_cast<float>(px) + 0.5f;
                float dx = std::abs(fx - centerX) - halfWidth;
                float dy = std::abs(fy - centerY) - halfHeight;
                float sdf = std::max(dx, dy);

                float sAlpha = 0.0f, fAlpha = 0.0f;
                if (drawStroke) {
                    float distToEdge = std::abs(sdf);
                    float raw = (halfStrokeWidth - distToEdge) * invAntialiasRange;
                    sAlpha = std::max(0.0f, std::min(1.0f, raw)) * finalStrokeOpacity;
                }
                if (drawFill) {
                    float raw = -sdf * invAntialiasRange;
                    fAlpha = std::max(0.0f, std::min(1.0f, raw)) * finalFillOpacity;
                }

                float finA, finR, finG, finB;
                if (mode_stroke_over_fill) {
                    float invS = 1.0f - sAlpha;
                    float modF = fAlpha * invS;
                    finA = sAlpha + modF;
                    finR = strokeColor.r * (1.0f / 255.0f) * sAlpha + fillColor.r * (1.0f / 255.0f) * modF;
                    finG = strokeColor.g * (1.0f / 255.0f) * sAlpha + fillColor.g * (1.0f / 255.0f) * modF;
                    finB = strokeColor.b * (1.0f / 255.0f) * sAlpha + fillColor.b * (1.0f / 255.0f) * modF;
                    if (finA > 0.001f) {
                        float invA = 1.0f / finA;
                        finR *= invA; finG *= invA; finB *= invA;
                    }
                }
                else if (mode_only_stroke) {
                    finA = sAlpha;
                    finR = strokeColor.r * (1.0f / 255.0f);
                    finG = strokeColor.g * (1.0f / 255.0f);
                    finB = strokeColor.b * (1.0f / 255.0f);
                }
                else {
                    finA = fAlpha;
                    finR = fillColor.r * (1.0f / 255.0f);
                    finG = fillColor.g * (1.0f / 255.0f);
                    finB = fillColor.b * (1.0f / 255.0f);
                }

                if (finA > 0.0f) {
                    pa2d::Color src;
                    src.r = static_cast<uint8_t>(std::min(255.0f, finR * 255.0f));
                    src.g = static_cast<uint8_t>(std::min(255.0f, finG * 255.0f));
                    src.b = static_cast<uint8_t>(std::min(255.0f, finB * 255.0f));
                    src.a = static_cast<uint8_t>(finA * 255.0f);
                    row[px] = Blend(src, row[px]);
                }
            }
        }
    }


    void rect(
        pa2d::Buffer& buffer,
        float centerX, float centerY, float width, float height, float angle,
        const pa2d::Color& fillColor,
        const pa2d::Color& strokeColor,
        float strokeWidth
    ) {
        // 1. 参数校验和预处理
        if (!buffer.isValid() || width <= 0 || height <= 0) return;
        if (angle == 0) return rect(buffer, centerX - width * 0.5f, centerY - height * 0.5f, width, height, fillColor, strokeColor, strokeWidth);

        const float finalFillOpacity = fillColor.a * (1.0f / 255.0f);
        const float finalStrokeOpacity = strokeColor.a * (1.0f / 255.0f);

        const bool drawFill = (finalFillOpacity > 0.0f);
        const bool drawStroke = (finalStrokeOpacity > 0.0f) && (strokeWidth > 0.0f);

        if (!drawFill && !drawStroke) return;

        // 宏观绘制模式判断
        const bool mode_stroke_over_fill = drawStroke && drawFill;
        const bool mode_only_stroke = drawStroke && !drawFill;
        const bool mode_only_fill = drawFill && !drawStroke;

        // 2. 几何参数
        const float antialiasRange = 1.0f;
        const float halfWidth = width * 0.5f;
        const float halfHeight = height * 0.5f;
        const float halfStrokeWidth = strokeWidth == 0 ? 0 : (strokeWidth + 1.0f) * 0.5f;
        angle = angle * (GEOMETRY_PI / 180.0f);

        // 2.5 旋转和包围盒 (Bounding Box) 计算
        const float s = std::sin(angle);
        const float c = std::cos(angle);
        const float abs_s = std::abs(s);
        const float abs_c = std::abs(c);

        // 计算旋转后矩形的世界空间包围盒 (AABB)
        const float worldHalfWidth = abs_c * halfWidth + abs_s * halfHeight;
        const float worldHalfHeight = abs_s * halfWidth + abs_c * halfHeight;

        // 3. 边界计算和裁剪
        const float maxExtX = worldHalfWidth + halfStrokeWidth + antialiasRange;
        const float maxExtY = worldHalfHeight + halfStrokeWidth + antialiasRange;

        // 基于中心点和最大扩展范围计算边界
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
        const __m256 halfWidth_v = _mm256_set1_ps(halfWidth);
        const __m256 halfHeight_v = _mm256_set1_ps(halfHeight);
        const __m256 halfStrokeWidth_v = _mm256_set1_ps(halfStrokeWidth);
        const __m256 antialiasRange_v = ANTIALIAS_RANGE_256;

        // 旋转常量
        const __m256 s_v = _mm256_set1_ps(s);
        const __m256 c_v = _mm256_set1_ps(c);

        // 颜色常量 (AVX2)
        const __m256 fillR_v = _mm256_set1_ps(fillColor.r * (1.0f / 255.0f));
        const __m256 fillG_v = _mm256_set1_ps(fillColor.g * (1.0f / 255.0f));
        const __m256 fillB_v = _mm256_set1_ps(fillColor.b * (1.0f / 255.0f));
        const __m256 strokeR_v = _mm256_set1_ps(strokeColor.r * (1.0f / 255.0f));
        const __m256 strokeG_v = _mm256_set1_ps(strokeColor.g * (1.0f / 255.0f));
        const __m256 strokeB_v = _mm256_set1_ps(strokeColor.b * (1.0f / 255.0f));
        const __m256 fillA_v = _mm256_set1_ps(finalFillOpacity);
        const __m256 strokeA_v = _mm256_set1_ps(finalStrokeOpacity);

        // SSE 常量
        const __m128 centerX_sse = _mm_set1_ps(centerX);
        const __m128 centerY_sse = _mm_set1_ps(centerY);
        const __m128 halfWidth_sse = _mm_set1_ps(halfWidth);
        const __m128 halfHeight_sse = _mm_set1_ps(halfHeight);
        const __m128 halfStrokeWidth_sse = _mm_set1_ps(halfStrokeWidth);
        const __m128 antialiasRange_sse = ANTIALIAS_RANGE_128;

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

        // 5. 主绘制循环
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

                // A. 旋转矩形 SDF
                __m256 dx_v = _mm256_sub_ps(px_v, centerX_v);
                __m256 dy_v = _mm256_sub_ps(py_v, centerY_v);

                // A.2. 逆向旋转像素 (到矩形的局部坐标系)
                __m256 x_local_v = _mm256_add_ps(_mm256_mul_ps(dx_v, c_v), _mm256_mul_ps(dy_v, s_v));
                __m256 y_local_v = _mm256_sub_ps(_mm256_mul_ps(dy_v, c_v), _mm256_mul_ps(dx_v, s_v));

                // A.3. 在局部坐标系中计算 AABB SDF
                __m256 dx_abs_local = _mm256_max_ps(_mm256_sub_ps(ZERO_256, x_local_v), x_local_v); // abs(x_local_v)
                __m256 dy_abs_local = _mm256_max_ps(_mm256_sub_ps(ZERO_256, y_local_v), y_local_v); // abs(y_local_v)

                __m256 d_x = _mm256_sub_ps(dx_abs_local, halfWidth_v);
                __m256 d_y = _mm256_sub_ps(dy_abs_local, halfHeight_v);

                __m256 sdf = _mm256_max_ps(d_x, d_y); // AABB SDF (正=外部, 负=内部)

                // B. Alpha 计算 (SDF -> Alpha)
                __m256 effectiveFillAlpha = ZERO_256;
                if (drawFill) {
                    __m256 t_fill = _mm256_div_ps(_mm256_sub_ps(sdf, antialiasRange_v), antialiasRange_v);
                    __m256 fillAlpha_raw = _mm256_sub_ps(ONE_256, t_fill);
                    effectiveFillAlpha = _mm256_mul_ps(_mm256_max_ps(ZERO_256, _mm256_min_ps(ONE_256, fillAlpha_raw)), fillA_v);
                }

                __m256 effectiveStrokeAlpha = ZERO_256;
                if (drawStroke) {
                    __m256 distToStrokeCenter = _mm256_max_ps(_mm256_sub_ps(ZERO_256, sdf), sdf); // |sdf|
                    __m256 halfWidthMinusDist = _mm256_sub_ps(halfStrokeWidth_v, distToStrokeCenter);
                    __m256 t_stroke = _mm256_div_ps(halfWidthMinusDist, antialiasRange_v);

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

                // A. 旋转矩形 SDF (SSE)
                __m128 dx_v = _mm_sub_ps(px_v_sse, centerX_sse);
                __m128 dy_v = _mm_sub_ps(py_v_sse, centerY_sse);

                __m128 x_local_v = _mm_add_ps(_mm_mul_ps(dx_v, c_sse), _mm_mul_ps(dy_v, s_sse));
                __m128 y_local_v = _mm_sub_ps(_mm_mul_ps(dy_v, c_sse), _mm_mul_ps(dx_v, s_sse));

                __m128 dx_abs_local = _mm_max_ps(_mm_sub_ps(ZERO_128, x_local_v), x_local_v);
                __m128 dy_abs_local = _mm_max_ps(_mm_sub_ps(ZERO_128, y_local_v), y_local_v);

                __m128 d_x = _mm_sub_ps(dx_abs_local, halfWidth_sse);
                __m128 d_y = _mm_sub_ps(dy_abs_local, halfHeight_sse);

                __m128 sdf = _mm_max_ps(d_x, d_y);

                // B. Alpha 计算
                __m128 effectiveFillAlpha = ZERO_128;
                if (drawFill) {
                    __m128 t_fill = _mm_div_ps(_mm_sub_ps(sdf, antialiasRange_sse), antialiasRange_sse);
                    __m128 fillAlpha_raw = _mm_sub_ps(ONE_128, t_fill);
                    effectiveFillAlpha = _mm_mul_ps(_mm_max_ps(ZERO_128, _mm_min_ps(ONE_128, fillAlpha_raw)), fillA_sse);
                }

                __m128 effectiveStrokeAlpha = ZERO_128;
                if (drawStroke) {
                    __m128 distToStrokeCenter = _mm_max_ps(_mm_sub_ps(ZERO_128, sdf), sdf);
                    __m128 halfWidthMinusDist = _mm_sub_ps(halfStrokeWidth_sse, distToStrokeCenter);
                    __m128 t_stroke = _mm_div_ps(halfWidthMinusDist, antialiasRange_sse);
                    __m128 strokeAlpha_raw = _mm_max_ps(ZERO_128, _mm_min_ps(ONE_128, t_stroke));
                    effectiveStrokeAlpha = _mm_mul_ps(strokeAlpha_raw, strokeA_sse);
                }

                // C. 颜色混合和写入 (SSE)
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

                const float d_x = std::abs(x_local) - halfWidth;
                const float d_y = std::abs(y_local) - halfHeight;

                const float sdf = std::max(d_x, d_y); // AABB SDF

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