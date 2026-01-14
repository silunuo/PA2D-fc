#include"../include/draw.h"
#include"internal/blend_utils.h"

using namespace pa2d::utils;
using namespace pa2d::utils::simd;

namespace pa2d {
    void triangle(
        pa2d::Buffer& buffer,
        float ax, float ay, float bx, float by, float cx, float cy,
        const pa2d::Color& fillColor,
        const pa2d::Color& strokeColor,
        float strokeWidth
    ) {
        if (!buffer.isValid()) return;

        // --- 1. 参数预处理 ---
        const float finalFillOpacity = fillColor.a * (1.0f / 255.0f);
        const float finalStrokeOpacity = strokeColor.a * (1.0f / 255.0f);

        const bool drawFill = (finalFillOpacity > 0.0f);
        const bool drawStroke = (finalStrokeOpacity > 0.0f) && (strokeWidth > 0.0f);

        if (!drawFill && !drawStroke) return;

        const bool mode_stroke_over_fill = drawStroke && drawFill;
        const bool mode_only_stroke = drawStroke && !drawFill;
        const bool mode_only_fill = drawFill && !drawStroke;

        const float halfStrokeWidth = strokeWidth == 0 ? 0 : (strokeWidth + 1.0f) * 0.5f;
        const float antialiasRange = 1.0f;  // 使用全局常量

        // --- 2. 边界计算和裁剪 ---
        const float minX_tri = std::min({ ax, bx, cx });
        const float maxX_tri = std::max({ ax, bx, cx });
        const float minY_tri = std::min({ ay, by, cy });
        const float maxY_tri = std::max({ ay, by, cy });

        const float maxExt = halfStrokeWidth + antialiasRange;

        int minX = static_cast<int>(std::floor(minX_tri - maxExt));
        int maxX = static_cast<int>(std::ceil(maxX_tri + maxExt));
        int minY = static_cast<int>(std::floor(minY_tri - maxExt));
        int maxY = static_cast<int>(std::ceil(maxY_tri + maxExt));

        minX = std::max(0, minX);
        maxX = std::min(buffer.width - 1, maxX);
        minY = std::max(0, minY);
        maxY = std::min(buffer.height - 1, maxY);

        if (minX > maxX || minY > maxY) return;

        // --- 确定三角形缠绕方向 ---
        const float area_cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);

        // 如果面积接近零，视为退化三角形
        if (std::abs(area_cross) < 1e-6f && drawFill) return;

        // 方向符号：逆时针为正，顺时针为负
        const float orientation_sign = (area_cross < 0.0f) ? -1.0f : 1.0f;

        // --- 3. SIMD 常量准备 ---
        // AVX2 常量
        const __m256 ax_v = _mm256_set1_ps(ax), ay_v = _mm256_set1_ps(ay);
        const __m256 bx_v = _mm256_set1_ps(bx), by_v = _mm256_set1_ps(by);
        const __m256 cx_v = _mm256_set1_ps(cx), cy_v = _mm256_set1_ps(cy);
        const __m256 halfStrokeWidth_v = _mm256_set1_ps(halfStrokeWidth);

        // 修复常量
        const __m256 orientation_sign_v = _mm256_set1_ps(orientation_sign);

        const __m256 fillR_v = _mm256_set1_ps(fillColor.r * (1.0f / 255.0f));
        const __m256 fillG_v = _mm256_set1_ps(fillColor.g * (1.0f / 255.0f));
        const __m256 fillB_v = _mm256_set1_ps(fillColor.b * (1.0f / 255.0f));
        const __m256 strokeR_v = _mm256_set1_ps(strokeColor.r * (1.0f / 255.0f));
        const __m256 strokeG_v = _mm256_set1_ps(strokeColor.g * (1.0f / 255.0f));
        const __m256 strokeB_v = _mm256_set1_ps(strokeColor.b * (1.0f / 255.0f));
        const __m256 fillA_v = _mm256_set1_ps(finalFillOpacity);
        const __m256 strokeA_v = _mm256_set1_ps(finalStrokeOpacity);

        // SSE 常量
        const __m128 ax_sse = _mm_set1_ps(ax), ay_sse = _mm_set1_ps(ay);
        const __m128 bx_sse = _mm_set1_ps(bx), by_sse = _mm_set1_ps(by);
        const __m128 cx_sse = _mm_set1_ps(cx), cy_sse = _mm_set1_ps(cy);
        const __m128 halfStrokeWidth_sse = _mm_set1_ps(halfStrokeWidth);

        // 修复常量
        const __m128 orientation_sign_sse = _mm_set1_ps(orientation_sign);

        const __m128 fillR_sse = _mm_set1_ps(fillColor.r * (1.0f / 255.0f));
        const __m128 fillG_sse = _mm_set1_ps(fillColor.g * (1.0f / 255.0f));
        const __m128 fillB_sse = _mm_set1_ps(fillColor.b * (1.0f / 255.0f));
        const __m128 strokeR_sse = _mm_set1_ps(strokeColor.r * (1.0f / 255.0f));
        const __m128 strokeG_sse = _mm_set1_ps(strokeColor.g * (1.0f / 255.0f));
        const __m128 strokeB_sse = _mm_set1_ps(strokeColor.b * (1.0f / 255.0f));
        const __m128 fillA_sse = _mm_set1_ps(finalFillOpacity);
        const __m128 strokeA_sse = _mm_set1_ps(finalStrokeOpacity);

        auto DistToSegment_AVX2 = [](
            __m256 Px, __m256 Py,
            __m256 Ax, __m256 Ay,
            __m256 Bx, __m256 By)
            {
                // 1. 向量计算: BA = A-B, PB = P-B
                __m256 BAx = _mm256_sub_ps(Ax, Bx);
                __m256 BAy = _mm256_sub_ps(Ay, By);
                __m256 PBx = _mm256_sub_ps(Px, Bx);
                __m256 PBy = _mm256_sub_ps(Py, By);

                // 2. 投影参数 t = clamp((PB . BA) / (BA . BA), 0.0, 1.0)
                __m256 dot_PB_BA = _mm256_add_ps(_mm256_mul_ps(PBx, BAx), _mm256_mul_ps(PBy, BAy));
                __m256 dot_BA_BA = _mm256_add_ps(_mm256_mul_ps(BAx, BAx), _mm256_mul_ps(BAy, BAy));

                // 避免除以零
                __m256 dot_BA_BA_inv = _mm256_div_ps(ONE_256, dot_BA_BA);
                __m256 zero_mask_eq = _mm256_cmp_ps(dot_BA_BA, ZERO_256, _CMP_EQ_OQ);
                dot_BA_BA_inv = _mm256_andnot_ps(zero_mask_eq, dot_BA_BA_inv);

                __m256 t = _mm256_mul_ps(dot_PB_BA, dot_BA_BA_inv);
                t = _mm256_max_ps(ZERO_256, _mm256_min_ps(ONE_256, t)); // clamp(t, 0, 1)

                // 3. 最近点 K = B + t * BA
                __m256 Kx = _mm256_add_ps(Bx, _mm256_mul_ps(t, BAx));
                __m256 Ky = _mm256_add_ps(By, _mm256_mul_ps(t, BAy));

                // 4. 距离 length(P - K)
                __m256 Qx = _mm256_sub_ps(Px, Kx);
                __m256 Qy = _mm256_sub_ps(Py, Ky);

                __m256 distSq = _mm256_add_ps(_mm256_mul_ps(Qx, Qx), _mm256_mul_ps(Qy, Qy));
                return _mm256_sqrt_ps(distSq);
            };

        auto DistToSegment_SSE = [](
            __m128 Px, __m128 Py,
            __m128 Ax, __m128 Ay,
            __m128 Bx, __m128 By)
            {
                // 1. 向量计算: BA = A-B, PB = P-B
                __m128 BAx = _mm_sub_ps(Ax, Bx);
                __m128 BAy = _mm_sub_ps(Ay, By);
                __m128 PBx = _mm_sub_ps(Px, Bx);
                __m128 PBy = _mm_sub_ps(Py, By);

                // 2. 投影参数 t
                __m128 dot_PB_BA = _mm_add_ps(_mm_mul_ps(PBx, BAx), _mm_mul_ps(PBy, BAy));
                __m128 dot_BA_BA = _mm_add_ps(_mm_mul_ps(BAx, BAx), _mm_mul_ps(BAy, BAy));

                __m128 dot_BA_BA_inv = _mm_div_ps(ONE_128, dot_BA_BA);
                __m128 zero_mask_eq = _mm_cmpeq_ps(dot_BA_BA, ZERO_128);
                dot_BA_BA_inv = _mm_andnot_ps(zero_mask_eq, dot_BA_BA_inv);

                __m128 t = _mm_mul_ps(dot_PB_BA, dot_BA_BA_inv);
                t = _mm_max_ps(ZERO_128, _mm_min_ps(ONE_128, t)); // clamp(t, 0, 1)

                // 3. 最近点 K = B + t * BA
                __m128 Kx = _mm_add_ps(Bx, _mm_mul_ps(t, BAx));
                __m128 Ky = _mm_add_ps(By, _mm_mul_ps(t, BAy));

                // 4. 距离 length(P - K)
                __m128 Qx = _mm_sub_ps(Px, Kx);
                __m128 Qy = _mm_sub_ps(Py, Ky);

                __m128 distSq = _mm_add_ps(_mm_mul_ps(Qx, Qx), _mm_mul_ps(Qy, Qy));
                return _mm_sqrt_ps(distSq);
            };

        // 4. 主绘制循环
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
                __m256 dist_AB = DistToSegment_AVX2(px_v, py_v, ax_v, ay_v, bx_v, by_v);
                __m256 dist_BC = DistToSegment_AVX2(px_v, py_v, bx_v, by_v, cx_v, cy_v);
                __m256 dist_CA = DistToSegment_AVX2(px_v, py_v, cx_v, cy_v, ax_v, ay_v);

                __m256 unsigned_dist = _mm256_min_ps(dist_AB, _mm256_min_ps(dist_BC, dist_CA));

                // 内外判断 (叉积/边缘函数)
                __m256 cross_AB = _mm256_sub_ps(_mm256_mul_ps(_mm256_sub_ps(bx_v, ax_v), _mm256_sub_ps(py_v, ay_v)),
                    _mm256_mul_ps(_mm256_sub_ps(by_v, ay_v), _mm256_sub_ps(px_v, ax_v)));
                __m256 cross_BC = _mm256_sub_ps(_mm256_mul_ps(_mm256_sub_ps(cx_v, bx_v), _mm256_sub_ps(py_v, by_v)),
                    _mm256_mul_ps(_mm256_sub_ps(cy_v, by_v), _mm256_sub_ps(px_v, bx_v)));
                __m256 cross_CA = _mm256_sub_ps(_mm256_mul_ps(_mm256_sub_ps(ax_v, cx_v), _mm256_sub_ps(py_v, cy_v)),
                    _mm256_mul_ps(_mm256_sub_ps(ay_v, cy_v), _mm256_sub_ps(px_v, cx_v)));

                // 修复点：将边缘函数乘以方向符号
                __m256 adj_cross_AB = _mm256_mul_ps(cross_AB, orientation_sign_v);
                __m256 adj_cross_BC = _mm256_mul_ps(cross_BC, orientation_sign_v);
                __m256 adj_cross_CA = _mm256_mul_ps(cross_CA, orientation_sign_v);

                // 假设调整后的叉积为正表示在三角形内部
                __m256 inside_mask_AB = _mm256_cmp_ps(adj_cross_AB, ZERO_256, _CMP_GE_OQ);
                __m256 inside_mask_BC = _mm256_cmp_ps(adj_cross_BC, ZERO_256, _CMP_GE_OQ);
                __m256 inside_mask_CA = _mm256_cmp_ps(adj_cross_CA, ZERO_256, _CMP_GE_OQ);

                __m256 inside_mask = _mm256_and_ps(_mm256_and_ps(inside_mask_AB, inside_mask_BC), inside_mask_CA);

                // 最终 SDF = unsigned_dist * sign_check (内部为负，外部为正)
                __m256 neg_one = _mm256_sub_ps(ZERO_256, ONE_256);
                __m256 sign = _mm256_blendv_ps(ONE_256, neg_one, inside_mask);
                __m256 sdf = _mm256_mul_ps(unsigned_dist, sign);

                // B. Alpha 计算 (SDF -> Alpha)
                __m256 effectiveFillAlpha = ZERO_256;
                if (drawFill) {
                    // Fill Alpha: 使用 sdf
                    __m256 t_fill = _mm256_div_ps(_mm256_sub_ps(sdf, ANTIALIAS_RANGE_256), ANTIALIAS_RANGE_256);
                    __m256 fillAlpha_raw = _mm256_sub_ps(ONE_256, t_fill);
                    __m256 final_fill_alpha = _mm256_mul_ps(_mm256_max_ps(ZERO_256, _mm256_min_ps(ONE_256, fillAlpha_raw)), fillA_v);
                    effectiveFillAlpha = _mm256_and_ps(final_fill_alpha, inside_mask); // 仅在内部绘制填充
                }

                __m256 effectiveStrokeAlpha = ZERO_256;
                if (drawStroke) {
                    // Stroke Alpha: 使用 |sdf|
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
                __m128 dist_AB = DistToSegment_SSE(px_v_sse, py_v_sse, ax_sse, ay_sse, bx_sse, by_sse);
                __m128 dist_BC = DistToSegment_SSE(px_v_sse, py_v_sse, bx_sse, by_sse, cx_sse, cy_sse);
                __m128 dist_CA = DistToSegment_SSE(px_v_sse, py_v_sse, cx_sse, cy_sse, ax_sse, ay_sse);

                __m128 unsigned_dist = _mm_min_ps(dist_AB, _mm_min_ps(dist_BC, dist_CA));

                // 内外判断 (SSE)
                __m128 cross_AB = _mm_sub_ps(_mm_mul_ps(_mm_sub_ps(bx_sse, ax_sse), _mm_sub_ps(py_v_sse, ay_sse)),
                    _mm_mul_ps(_mm_sub_ps(by_sse, ay_sse), _mm_sub_ps(px_v_sse, ax_sse)));
                __m128 cross_BC = _mm_sub_ps(_mm_mul_ps(_mm_sub_ps(cx_sse, bx_sse), _mm_sub_ps(py_v_sse, by_sse)),
                    _mm_mul_ps(_mm_sub_ps(cy_sse, by_sse), _mm_sub_ps(px_v_sse, bx_sse)));
                __m128 cross_CA = _mm_sub_ps(_mm_mul_ps(_mm_sub_ps(ax_sse, cx_sse), _mm_sub_ps(py_v_sse, cy_sse)),
                    _mm_mul_ps(_mm_sub_ps(ay_sse, cy_sse), _mm_sub_ps(px_v_sse, cx_sse)));

                // 修复点：将边缘函数乘以方向符号
                __m128 adj_cross_AB = _mm_mul_ps(cross_AB, orientation_sign_sse);
                __m128 adj_cross_BC = _mm_mul_ps(cross_BC, orientation_sign_sse);
                __m128 adj_cross_CA = _mm_mul_ps(cross_CA, orientation_sign_sse);

                __m128 inside_mask_AB = _mm_cmpge_ps(adj_cross_AB, ZERO_128);
                __m128 inside_mask_BC = _mm_cmpge_ps(adj_cross_BC, ZERO_128);
                __m128 inside_mask_CA = _mm_cmpge_ps(adj_cross_CA, ZERO_128);

                __m128 inside_mask = _mm_and_ps(_mm_and_ps(inside_mask_AB, inside_mask_BC), inside_mask_CA);

                __m128 neg_one_sse = _mm_sub_ps(ZERO_128, ONE_128);
                __m128 sign = _mm_or_ps(_mm_andnot_ps(inside_mask, ONE_128), _mm_and_ps(inside_mask, neg_one_sse));
                __m128 sdf = _mm_mul_ps(unsigned_dist, sign);

                // B. Alpha 计算 (SDF -> Alpha)
                __m128 effectiveFillAlpha = ZERO_128;
                if (drawFill) {
                    __m128 t_fill = _mm_div_ps(_mm_sub_ps(sdf, ANTIALIAS_RANGE_128), ANTIALIAS_RANGE_128);
                    __m128 fillAlpha_raw = _mm_sub_ps(ONE_128, t_fill);
                    __m128 final_fill_alpha = _mm_mul_ps(_mm_max_ps(ZERO_128, _mm_min_ps(ONE_128, fillAlpha_raw)), fillA_sse);
                    effectiveFillAlpha = _mm_and_ps(final_fill_alpha, inside_mask);
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
                auto distToSegment = [&](float Px, float Py, float Ax, float Ay, float Bx, float By) -> float {
                    float BAx = Ax - Bx;
                    float BAy = Ay - By;
                    float PBx = Px - Bx;
                    float PBy = Py - By;
                    float dot_PB_BA = PBx * BAx + PBy * BAy;
                    float dot_BA_BA = BAx * BAx + BAy * BAy;

                    float t = 0.0f;
                    if (dot_BA_BA > 1e-6) {
                        t = std::max(0.0f, std::min(1.0f, dot_PB_BA / dot_BA_BA));
                    }

                    float Kx = Bx + t * BAx;
                    float Ky = By + t * BAy;
                    return std::sqrt(std::pow(Px - Kx, 2) + std::pow(Py - Ky, 2));
                    };

                float dist_AB = distToSegment(fx, fy, ax, ay, bx, by);
                float dist_BC = distToSegment(fx, fy, bx, by, cx, cy);
                float dist_CA = distToSegment(fx, fy, cx, cy, ax, ay);

                float unsigned_dist = std::min({ dist_AB, dist_BC, dist_CA });

                // 内外判断 (边缘函数/叉积)
                auto cross = [&](float Px, float Py, float Ax, float Ay, float Bx, float By) {
                    return (Bx - Ax) * (Py - Ay) - (By - Ay) * (Px - Ax);
                    };

                float cross_AB = cross(fx, fy, ax, ay, bx, by);
                float cross_BC = cross(fx, fy, bx, by, cx, cy);
                float cross_CA = cross(fx, fy, cx, cy, ax, ay);

                // 修复点：将叉积乘以方向符号
                float adj_cross_AB = cross_AB * orientation_sign;
                float adj_cross_BC = cross_BC * orientation_sign;
                float adj_cross_CA = cross_CA * orientation_sign;

                bool is_inside = (adj_cross_AB >= 0.0f) && (adj_cross_BC >= 0.0f) && (adj_cross_CA >= 0.0f);

                // 最终 SDF
                float sdf = is_inside ? -unsigned_dist : unsigned_dist;

                // B. Alpha 计算
                float effectiveFillAlpha = 0.0f;
                if (drawFill) {
                    if (is_inside) {
                        float t_fill = (sdf - antialiasRange) / antialiasRange;
                        float fillAlpha_raw = 1.0f - t_fill;
                        effectiveFillAlpha = std::max(0.0f, std::min(1.0f, fillAlpha_raw)) * finalFillOpacity;
                    }
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