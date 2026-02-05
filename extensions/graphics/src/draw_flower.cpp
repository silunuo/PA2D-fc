// draw_flower.cpp - 花朵渲染函数实现
#include "../include/draw_flower.h"
#include "../include/blend_utils.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <immintrin.h> // AVX2
#include <emmintrin.h> // SSE

// 辅助常量：五角折叠向量
// k1 = { sin(72/2), cos(72/2) } -> { 0.587785, 0.809017 } 
// 调整坐标系以适配屏幕坐标 (Y向下) 和 0度角朝上
// 这里使用 Inigo Quilez 的 SDF 参数变体适配屏幕坐标系
namespace star_constants {
    const float SIN_36 = 0.58778525229f;
    const float COS_36 = 0.80901699437f;

    // 折叠向量 K1 (对应 72度)
    const float K1_X = 0.809016994f;
    const float K1_Y = -0.587785252f;

    // 折叠向量 K2 (对应 144度)
    const float K2_X = -0.809016994f;
    const float K2_Y = -0.587785252f;
}

// 核心通用渲染函数实现
void flower_impl(
    pa2d::Buffer& buffer,
    float cx, float cy,           // 中心坐标
    float radius,                 // 外半径
    float innerRadius,            // 内半径
    float angle,               // 旋转角度 (弧度)
    const pa2d::Color& fillColor,
    const pa2d::Color& strokeColor,
    float strokeWidth
) {
    using namespace simd;
    if (!buffer) return;

    using namespace star_constants;

    // --- 1. 参数预处理 ---
    const float finalFillOpacity = fillColor.a * (1.0f / 255.0f);
    const float finalStrokeOpacity = strokeColor.a * (1.0f / 255.0f);

    const bool drawFill = (finalFillOpacity > 0.0f);
    const bool drawStroke = (finalStrokeOpacity > 0.0f) && (strokeWidth > 0.0f);

    if (!drawFill && !drawStroke) return;

    const bool mode_stroke_over_fill = drawStroke && drawFill;
    const bool mode_only_stroke = drawStroke && !drawFill;
    const float halfStrokeWidth = strokeWidth * 0.5f;
    const float max_radius = radius + halfStrokeWidth + 1.0f;
    const float rotation = angle * (GEOMETRY_PI / 180.0f);
    // 旋转矩阵预计算
    const float cos_r = std::cos(rotation);
    const float sin_r = std::sin(rotation);
    const float inv_radius_range = 1.0f / (radius - innerRadius);

    // --- 2. 边界计算和裁剪 ---
    int minX = static_cast<int>(std::floor(cx - max_radius));
    int maxX = static_cast<int>(std::ceil(cx + max_radius));
    int minY = static_cast<int>(std::floor(cy - max_radius));
    int maxY = static_cast<int>(std::ceil(cy + max_radius));

    minX = std::max(0, minX);
    maxX = std::min(buffer.width - 1, maxX);
    minY = std::max(0, minY);
    maxY = std::min(buffer.height - 1, maxY);

    if (minX > maxX || minY > maxY) return;

    // --- 3. SIMD常量预计算 ---
    // AVX2常量
    const __m256 cx_v = _mm256_set1_ps(cx);
    const __m256 cy_v = _mm256_set1_ps(cy);
    const __m256 cos_v = _mm256_set1_ps(cos_r);
    const __m256 sin_v = _mm256_set1_ps(sin_r);
    const __m256 radius_v = _mm256_set1_ps(radius);
    const __m256 innerRadius_v = _mm256_set1_ps(innerRadius);
    const __m256 inv_radius_range_v = _mm256_set1_ps(inv_radius_range);
    const __m256 halfStrokeWidth_v = _mm256_set1_ps(halfStrokeWidth);
    const __m256 aa_range_v = _mm256_set1_ps(1.0f); // ANTIALIAS_RANGE = 1.0

    const __m256 pi_2_5_v = _mm256_set1_ps(2.0f * GEOMETRY_PI / 5.0f);
    const __m256 half_sector_v = _mm256_mul_ps(pi_2_5_v, _mm256_set1_ps(0.5f));
    const __m256 inv_sector_v = _mm256_set1_ps(5.0f / (2.0f * GEOMETRY_PI));
    const __m256 half_pi_2_5_v = _mm256_set1_ps(GEOMETRY_PI / 5.0f);

    // 颜色常量
    const __m256 fillR_v = _mm256_set1_ps(fillColor.r * (1.0f / 255.0f));
    const __m256 fillG_v = _mm256_set1_ps(fillColor.g * (1.0f / 255.0f));
    const __m256 fillB_v = _mm256_set1_ps(fillColor.b * (1.0f / 255.0f));
    const __m256 fillA_v = _mm256_set1_ps(finalFillOpacity);
    const __m256 strokeR_v = _mm256_set1_ps(strokeColor.r * (1.0f / 255.0f));
    const __m256 strokeG_v = _mm256_set1_ps(strokeColor.g * (1.0f / 255.0f));
    const __m256 strokeB_v = _mm256_set1_ps(strokeColor.b * (1.0f / 255.0f));
    const __m256 strokeA_v = _mm256_set1_ps(finalStrokeOpacity);

    // SSE常量
    const __m128 cx_sse = _mm_set1_ps(cx);
    const __m128 cy_sse = _mm_set1_ps(cy);
    const __m128 cos_sse = _mm_set1_ps(cos_r);
    const __m128 sin_sse = _mm_set1_ps(sin_r);
    const __m128 radius_sse = _mm_set1_ps(radius);
    const __m128 innerRadius_sse = _mm_set1_ps(innerRadius);
    const __m128 inv_radius_range_sse = _mm_set1_ps(inv_radius_range);
    const __m128 halfStrokeWidth_sse = _mm_set1_ps(halfStrokeWidth);
    const __m128 aa_range_sse = _mm_set1_ps(1.0f);

    const __m128 pi_2_5_sse = _mm_set1_ps(2.0f * GEOMETRY_PI / 5.0f);
    const __m128 half_sector_sse = _mm_mul_ps(pi_2_5_sse, _mm_set1_ps(0.5f));
    const __m128 inv_sector_sse = _mm_set1_ps(5.0f / (2.0f * GEOMETRY_PI));
    const __m128 half_pi_2_5_sse = _mm_set1_ps(GEOMETRY_PI / 5.0f);

    // 颜色SSE常量
    const __m128 fillR_sse = _mm_set1_ps(fillColor.r * (1.0f / 255.0f));
    const __m128 fillG_sse = _mm_set1_ps(fillColor.g * (1.0f / 255.0f));
    const __m128 fillB_sse = _mm_set1_ps(fillColor.b * (1.0f / 255.0f));
    const __m128 fillA_sse = _mm_set1_ps(finalFillOpacity);
    const __m128 strokeR_sse = _mm_set1_ps(strokeColor.r * (1.0f / 255.0f));
    const __m128 strokeG_sse = _mm_set1_ps(strokeColor.g * (1.0f / 255.0f));
    const __m128 strokeB_sse = _mm_set1_ps(strokeColor.b * (1.0f / 255.0f));
    const __m128 strokeA_sse = _mm_set1_ps(finalStrokeOpacity);

    // --- 4. SIMD辅助函数 ---
    // AVX2: 计算点到线段的距离
    auto avx2_distance_to_segment = [](__m256 px, __m256 py,
        __m256 x1, __m256 y1,
        __m256 x2, __m256 y2) -> __m256 {
        __m256 dx = _mm256_sub_ps(x2, x1);
        __m256 dy = _mm256_sub_ps(y2, y1);
        __m256 segment_length_sq = _mm256_add_ps(_mm256_mul_ps(dx, dx),
            _mm256_mul_ps(dy, dy));

        // 避免除零
        __m256 eps = _mm256_set1_ps(1e-6f);
        __m256 safe_segment_length_sq = _mm256_max_ps(segment_length_sq, eps);

        // 计算投影参数 t
        __m256 px_minus_x1 = _mm256_sub_ps(px, x1);
        __m256 py_minus_y1 = _mm256_sub_ps(py, y1);
        __m256 dot = _mm256_add_ps(_mm256_mul_ps(px_minus_x1, dx),
            _mm256_mul_ps(py_minus_y1, dy));
        __m256 t = _mm256_div_ps(dot, safe_segment_length_sq);
        t = _mm256_min_ps(_mm256_max_ps(t, simd::ZERO_256), simd::ONE_256);

        // 计算投影点
        __m256 proj_x = _mm256_add_ps(x1, _mm256_mul_ps(t, dx));
        __m256 proj_y = _mm256_add_ps(y1, _mm256_mul_ps(t, dy));

        // 计算距离
        __m256 diff_x = _mm256_sub_ps(px, proj_x);
        __m256 diff_y = _mm256_sub_ps(py, proj_y);
        return _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(diff_x, diff_x),
            _mm256_mul_ps(diff_y, diff_y)));
    };

    // SSE: 计算点到线段的距离
    auto sse_distance_to_segment = [](__m128 px, __m128 py,
        __m128 x1, __m128 y1,
        __m128 x2, __m128 y2) -> __m128 {
        __m128 dx = _mm_sub_ps(x2, x1);
        __m128 dy = _mm_sub_ps(y2, y1);
        __m128 segment_length_sq = _mm_add_ps(_mm_mul_ps(dx, dx),
            _mm_mul_ps(dy, dy));

        __m128 eps = _mm_set1_ps(1e-6f);
        __m128 safe_segment_length_sq = _mm_max_ps(segment_length_sq, eps);

        __m128 px_minus_x1 = _mm_sub_ps(px, x1);
        __m128 py_minus_y1 = _mm_sub_ps(py, y1);
        __m128 dot = _mm_add_ps(_mm_mul_ps(px_minus_x1, dx),
            _mm_mul_ps(py_minus_y1, dy));
        __m128 t = _mm_div_ps(dot, safe_segment_length_sq);
        t = _mm_min_ps(_mm_max_ps(t, ZERO_128), ONE_128);

        __m128 proj_x = _mm_add_ps(x1, _mm_mul_ps(t, dx));
        __m128 proj_y = _mm_add_ps(y1, _mm_mul_ps(t, dy));

        __m128 diff_x = _mm_sub_ps(px, proj_x);
        __m128 diff_y = _mm_sub_ps(py, proj_y);
        return _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(diff_x, diff_x),
            _mm_mul_ps(diff_y, diff_y)));
    };

    // --- 5. 主渲染循环 ---
    for (int py = minY; py <= maxY; ++py) {
        const float fy = static_cast<float>(py) + 0.5f;
        pa2d::Color* row = &buffer.at(0, py);

        const __m256 py_v = _mm256_set1_ps(fy);
        const __m128 py_v_sse = _mm_set1_ps(fy);

        int px = minX;

        // === AVX2循环 (8像素) ===
        for (; px <= maxX - 7; px += 8) {
            __m256i xBase = _mm256_setr_epi32(px, px + 1, px + 2, px + 3, px + 4, px + 5, px + 6, px + 7);
            __m256 px_v = _mm256_add_ps(_mm256_cvtepi32_ps(xBase), HALF_PIXEL_256);

            // 1. 变换到本地坐标系
            __m256 dx = _mm256_sub_ps(px_v, cx_v);
            __m256 dy = _mm256_sub_ps(py_v, cy_v);

            // 应用旋转
            __m256 x = _mm256_sub_ps(_mm256_mul_ps(dx, cos_v), _mm256_mul_ps(dy, sin_v));
            __m256 y = _mm256_add_ps(_mm256_mul_ps(dx, sin_v), _mm256_mul_ps(dy, cos_v));

            // 2. 计算极坐标
            __m256 length_sq = _mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y));
            __m256 length = _mm256_sqrt_ps(length_sq);

            // 计算角度 (atan2)
            __m256 angle = avx2_atan2(y, x);

            // 3. 扇区映射
            // 计算扇区索引
            __m256 sector = _mm256_floor_ps(_mm256_mul_ps(angle, inv_sector_v));
            __m256 angle_in_sector = _mm256_sub_ps(angle, _mm256_mul_ps(sector, pi_2_5_v));

            // 归一化到[-half_sector, half_sector]
            __m256 mask_gt_half = _mm256_cmp_ps(angle_in_sector, half_sector_v, _CMP_GT_OQ);
            __m256 adjust = _mm256_and_ps(mask_gt_half, pi_2_5_v);
            angle_in_sector = _mm256_sub_ps(angle_in_sector, adjust);

            // 4. 计算参考半径
            __m256 angle_abs = _mm256_and_ps(angle_in_sector, abs_mask_256);
            __m256 t_radius = _mm256_div_ps(angle_abs, half_sector_v);
            t_radius = _mm256_min_ps(_mm256_max_ps(t_radius, ZERO_256), ONE_256);

            __m256 reference_radius = _mm256_add_ps(radius_v,
                _mm256_mul_ps(_mm256_sub_ps(innerRadius_v, radius_v), t_radius));

            // 5. 径向SDF
            __m256 sdf_radial = _mm256_sub_ps(length, reference_radius);

            // 6. 计算到边的距离
            // 预计算扇区顶点
            __m256 sector_0 = _mm256_mul_ps(sector, pi_2_5_v);
            __m256 sector_1 = _mm256_add_ps(sector_0, pi_2_5_v);
            __m256 sector_half = _mm256_add_ps(sector_0, half_pi_2_5_v);

            __m256 cos_s0 = avx2_cos(sector_0);
            __m256 sin_s0 = avx2_sin(sector_0);
            __m256 cos_s1 = avx2_cos(sector_1);
            __m256 sin_s1 = avx2_sin(sector_1);
            __m256 cos_sh = avx2_cos(sector_half);
            __m256 sin_sh = avx2_sin(sector_half);

            __m256 outer_x1 = _mm256_mul_ps(cos_s0, radius_v);
            __m256 outer_y1 = _mm256_mul_ps(sin_s0, radius_v);
            __m256 outer_x2 = _mm256_mul_ps(cos_s1, radius_v);
            __m256 outer_y2 = _mm256_mul_ps(sin_s1, radius_v);
            __m256 inner_x = _mm256_mul_ps(cos_sh, innerRadius_v);
            __m256 inner_y = _mm256_mul_ps(sin_sh, innerRadius_v);

            // 计算到两条边的距离
            __m256 dist_edge1 = avx2_distance_to_segment(x, y, outer_x1, outer_y1, inner_x, inner_y);
            __m256 dist_edge2 = avx2_distance_to_segment(x, y, inner_x, inner_y, outer_x2, outer_y2);

            __m256 edge_dist = _mm256_min_ps(dist_edge1, dist_edge2);

            // 7. 最终SDF
            __m256 sdf = _mm256_min_ps(sdf_radial, edge_dist);

            // 8. 计算填充和描边的alpha
            __m256 fill_alpha = ZERO_256;
            if (drawFill) {
                // 只在内部有值
                __m256 inside_mask = _mm256_cmp_ps(sdf, ZERO_256, _CMP_LE_OQ);
                __m256 neg_sdf = _mm256_sub_ps(ZERO_256, sdf);
                __m256 t_fill = _mm256_div_ps(neg_sdf, aa_range_v);
                __m256 fill_alpha_raw = _mm256_min_ps(ONE_256, t_fill);
                fill_alpha = _mm256_mul_ps(fill_alpha_raw, fillA_v);
                fill_alpha = _mm256_and_ps(fill_alpha, inside_mask);
            }

            __m256 stroke_alpha = ZERO_256;
            if (drawStroke) {
                __m256 sdf_abs = _mm256_and_ps(sdf, abs_mask_256);
                __m256 dist_to_edge = _mm256_sub_ps(halfStrokeWidth_v, sdf_abs);
                __m256 t_stroke = _mm256_div_ps(dist_to_edge, aa_range_v);
                __m256 stroke_alpha_raw = _mm256_max_ps(ZERO_256,
                    _mm256_min_ps(ONE_256, t_stroke));
                stroke_alpha = _mm256_mul_ps(stroke_alpha_raw, strokeA_v);
            }

            // 9. 颜色混合
            __m256 final_alpha, final_r, final_g, final_b;
            if (mode_stroke_over_fill) {
                __m256 one_minus_stroke = _mm256_sub_ps(ONE_256, stroke_alpha);
                __m256 fill_mod = _mm256_mul_ps(fill_alpha, one_minus_stroke);
                final_alpha = _mm256_add_ps(stroke_alpha, fill_mod);

                __m256 inv_alpha = _mm256_div_ps(ONE_256, final_alpha);
                inv_alpha = _mm256_andnot_ps(_mm256_cmp_ps(final_alpha, ZERO_256, _CMP_EQ_OQ), inv_alpha);

                final_r = _mm256_mul_ps(_mm256_add_ps(
                    _mm256_mul_ps(strokeR_v, stroke_alpha),
                    _mm256_mul_ps(fillR_v, fill_mod)), inv_alpha);
                final_g = _mm256_mul_ps(_mm256_add_ps(
                    _mm256_mul_ps(strokeG_v, stroke_alpha),
                    _mm256_mul_ps(fillG_v, fill_mod)), inv_alpha);
                final_b = _mm256_mul_ps(_mm256_add_ps(
                    _mm256_mul_ps(strokeB_v, stroke_alpha),
                    _mm256_mul_ps(fillB_v, fill_mod)), inv_alpha);
            }
            else if (mode_only_stroke) {
                final_alpha = stroke_alpha;
                final_r = strokeR_v;
                final_g = strokeG_v;
                final_b = strokeB_v;
            }
            else {
                final_alpha = fill_alpha;
                final_r = fillR_v;
                final_g = fillG_v;
                final_b = fillB_v;
            }

            // 10. 混合到目标缓冲区
            __m256 alpha_mask = _mm256_cmp_ps(final_alpha, ZERO_256, _CMP_GT_OQ);
            if (!_mm256_testz_ps(alpha_mask, alpha_mask)) {
                // 转换颜色到8位并混合
                __m256i dest = _mm256_loadu_si256((__m256i*) & row[px]);
                __m256i rgba = blend_pixels_avx(final_alpha, dest, final_r, final_g, final_b);

                // 只更新alpha > 0的像素
                rgba = _mm256_blendv_epi8(dest, rgba, _mm256_castps_si256(alpha_mask));
                _mm256_storeu_si256((__m256i*) & row[px], rgba);
            }
        }

        // === SSE循环 (4像素) ===
        for (; px <= maxX - 3; px += 4) {
            __m128i xBase = _mm_setr_epi32(px, px + 1, px + 2, px + 3);
            __m128 px_v = _mm_add_ps(_mm_cvtepi32_ps(xBase), HALF_PIXEL_128);

            // 变换到本地坐标系
            __m128 dx = _mm_sub_ps(px_v, cx_sse);
            __m128 dy = _mm_sub_ps(py_v_sse, cy_sse);

            __m128 x = _mm_sub_ps(_mm_mul_ps(dx, cos_sse), _mm_mul_ps(dy, sin_sse));
            __m128 y = _mm_add_ps(_mm_mul_ps(dx, sin_sse), _mm_mul_ps(dy, cos_sse));

            // 计算极坐标
            __m128 length_sq = _mm_add_ps(_mm_mul_ps(x, x), _mm_mul_ps(y, y));
            __m128 length = _mm_sqrt_ps(length_sq);

            // 计算角度 (近似atan2)
            __m128 angle = sse_atan2(y, x);

            // 扇区映射
            __m128 sector = sse_floor(_mm_mul_ps(angle, inv_sector_sse));
            __m128 angle_in_sector = _mm_sub_ps(angle, _mm_mul_ps(sector, pi_2_5_sse));

            __m128 mask_gt_half = _mm_cmpgt_ps(angle_in_sector, half_sector_sse);
            __m128 adjust = _mm_and_ps(mask_gt_half, pi_2_5_sse);
            angle_in_sector = _mm_sub_ps(angle_in_sector, adjust);

            // 计算参考半径
            __m128 angle_abs = _mm_and_ps(angle_in_sector, abs_mask_128);
            __m128 t_radius = _mm_div_ps(angle_abs, half_sector_sse);
            t_radius = _mm_min_ps(_mm_max_ps(t_radius, ZERO_128), ONE_128);

            __m128 reference_radius = _mm_add_ps(radius_sse,
                _mm_mul_ps(_mm_sub_ps(innerRadius_sse, radius_sse), t_radius));

            __m128 sdf_radial = _mm_sub_ps(length, reference_radius);

            // 计算到边的距离
            __m128 sector_0 = _mm_mul_ps(sector, pi_2_5_sse);
            __m128 sector_1 = _mm_add_ps(sector_0, pi_2_5_sse);
            __m128 sector_half = _mm_add_ps(sector_0, half_pi_2_5_sse);

            __m128 cos_s0 = sse_cos(sector_0);
            __m128 sin_s0 = sse_sin(sector_0);
            __m128 cos_s1 = sse_cos(sector_1);
            __m128 sin_s1 = sse_sin(sector_1);
            __m128 cos_sh = sse_cos(sector_half);
            __m128 sin_sh = sse_sin(sector_half);

            __m128 outer_x1 = _mm_mul_ps(cos_s0, radius_sse);
            __m128 outer_y1 = _mm_mul_ps(sin_s0, radius_sse);
            __m128 outer_x2 = _mm_mul_ps(cos_s1, radius_sse);
            __m128 outer_y2 = _mm_mul_ps(sin_s1, radius_sse);
            __m128 inner_x = _mm_mul_ps(cos_sh, innerRadius_sse);
            __m128 inner_y = _mm_mul_ps(sin_sh, innerRadius_sse);

            __m128 dist_edge1 = sse_distance_to_segment(x, y, outer_x1, outer_y1, inner_x, inner_y);
            __m128 dist_edge2 = sse_distance_to_segment(x, y, inner_x, inner_y, outer_x2, outer_y2);

            __m128 edge_dist = _mm_min_ps(dist_edge1, dist_edge2);
            __m128 sdf = _mm_min_ps(sdf_radial, edge_dist);

            // 计算alpha
            __m128 fill_alpha = ZERO_128;
            if (drawFill) {
                __m128 inside_mask = _mm_cmple_ps(sdf, ZERO_128);
                __m128 neg_sdf = _mm_sub_ps(ZERO_128, sdf);
                __m128 t_fill = _mm_div_ps(neg_sdf, aa_range_sse);
                __m128 fill_alpha_raw = _mm_min_ps(ONE_128, t_fill);
                fill_alpha = _mm_mul_ps(fill_alpha_raw, fillA_sse);
                fill_alpha = _mm_and_ps(fill_alpha, inside_mask);
            }

            __m128 stroke_alpha = ZERO_128;
            if (drawStroke) {
                __m128 sdf_abs = _mm_and_ps(sdf, abs_mask_128);
                __m128 dist_to_edge = _mm_sub_ps(halfStrokeWidth_sse, sdf_abs);
                __m128 t_stroke = _mm_div_ps(dist_to_edge, aa_range_sse);
                __m128 stroke_alpha_raw = _mm_max_ps(ZERO_128, _mm_min_ps(ONE_128, t_stroke));
                stroke_alpha = _mm_mul_ps(stroke_alpha_raw, strokeA_sse);
            }

            // 颜色混合
            __m128 final_alpha, final_r, final_g, final_b;
            if (mode_stroke_over_fill) {
                __m128 one_minus_stroke = _mm_sub_ps(ONE_128, stroke_alpha);
                __m128 fill_mod = _mm_mul_ps(fill_alpha, one_minus_stroke);
                final_alpha = _mm_add_ps(stroke_alpha, fill_mod);

                __m128 inv_alpha = _mm_div_ps(ONE_128, final_alpha);
                inv_alpha = _mm_andnot_ps(_mm_cmpeq_ps(final_alpha, ZERO_128), inv_alpha);

                final_r = _mm_mul_ps(_mm_add_ps(
                    _mm_mul_ps(strokeR_sse, stroke_alpha),
                    _mm_mul_ps(fillR_sse, fill_mod)), inv_alpha);
                final_g = _mm_mul_ps(_mm_add_ps(
                    _mm_mul_ps(strokeG_sse, stroke_alpha),
                    _mm_mul_ps(fillG_sse, fill_mod)), inv_alpha);
                final_b = _mm_mul_ps(_mm_add_ps(
                    _mm_mul_ps(strokeB_sse, stroke_alpha),
                    _mm_mul_ps(fillB_sse, fill_mod)), inv_alpha);
            }
            else if (mode_only_stroke) {
                final_alpha = stroke_alpha;
                final_r = strokeR_sse;
                final_g = strokeG_sse;
                final_b = strokeB_sse;
            }
            else {
                final_alpha = fill_alpha;
                final_r = fillR_sse;
                final_g = fillG_sse;
                final_b = fillB_sse;
            }

            // 混合到目标缓冲区
            __m128 alpha_mask = _mm_cmpgt_ps(final_alpha, ZERO_128);
            if (_mm_movemask_ps(alpha_mask)) {
                __m128i dest = _mm_loadu_si128((__m128i*) & row[px]);
                __m128i rgba = blend_pixels_sse(final_alpha, dest, final_r, final_g, final_b);

                rgba = _mm_blendv_epi8(dest, rgba, _mm_castps_si128(alpha_mask));
                _mm_storeu_si128((__m128i*) & row[px], rgba);
            }
        }

        // === 标量循环 (剩余像素) ===
        for (; px <= maxX; ++px) {
            float fx = static_cast<float>(px) + 0.5f;
            float fy_ = static_cast<float>(py) + 0.5f;

            // 使用原始标量版本的compute_star_sdf
            float dx = fx - cx;
            float dy = fy_ - cy;
            float x = dx * cos_r - dy * sin_r;
            float y = dx * sin_r + dy * cos_r;

            float angle = std::atan2(y, x);
            float length = std::sqrt(x * x + y * y);

            const float sector_angle = 2.0f * GEOMETRY_PI / 5.0f;
            const float half_sector = sector_angle * 0.5f;

            float sector = std::floor(angle / sector_angle);
            angle -= sector * sector_angle;

            if (angle > half_sector) {
                angle -= sector_angle;
            }

            float angle_offset = std::abs(angle);
            float t = angle_offset / half_sector;
            float reference_radius = radius + (innerRadius - radius) * t;

            float sdf_radial = length - reference_radius;

            // 计算到边的距离
            auto distance_to_segment = [](float px, float py, float x1, float y1, float x2, float y2) -> float {
                float dx = x2 - x1;
                float dy = y2 - y1;
                float segment_length_sq = dx * dx + dy * dy;

                if (segment_length_sq < 1e-6f) {
                    return std::sqrt((px - x1) * (px - x1) + (py - y1) * (py - y1));
                }

                float t_ = ((px - x1) * dx + (py - y1) * dy) / segment_length_sq;
                t_ = std::max(0.0f, std::min(1.0f, t_));

                float proj_x = x1 + t_ * dx;
                float proj_y = y1 + t_ * dy;

                return std::sqrt((px - proj_x) * (px - proj_x) + (py - proj_y) * (py - proj_y));
                };

            float outer_angle1 = sector * sector_angle;
            float outer_angle2 = (sector + 1) * sector_angle;
            float inner_angle = (sector + 0.5f) * sector_angle;

            float outer_x1 = std::cos(outer_angle1) * radius;
            float outer_y1 = std::sin(outer_angle1) * radius;
            float outer_x2 = std::cos(outer_angle2) * radius;
            float outer_y2 = std::sin(outer_angle2) * radius;
            float inner_x_ = std::cos(inner_angle) * innerRadius;
            float inner_y_ = std::sin(inner_angle) * innerRadius;

            float dist_edge1 = distance_to_segment(x, y, outer_x1, outer_y1, inner_x_, inner_y_);
            float dist_edge2 = distance_to_segment(x, y, inner_x_, inner_y_, outer_x2, outer_y2);
            float edge_dist = std::min(dist_edge1, dist_edge2);

            float sdf = std::min(sdf_radial, edge_dist);

            // 计算alpha
            float fill_alpha = 0.0f;
            if (drawFill && sdf <= 0.0f) {
                float t_fill = (-sdf) / 1.0f;
                fill_alpha = std::min(1.0f, t_fill) * finalFillOpacity;
            }

            float stroke_alpha = 0.0f;
            if (drawStroke) {
                float sdf_abs = std::abs(sdf);
                if (sdf_abs <= halfStrokeWidth + 1.0f) {
                    float t_stroke = (halfStrokeWidth + 1.0f - sdf_abs) / 1.0f;
                    stroke_alpha = std::max(0.0f, std::min(1.0f, t_stroke)) * finalStrokeOpacity;
                }
            }

            // 混合逻辑
            float final_alpha = 0.0f;
            float final_r = 0.0f, final_g = 0.0f, final_b = 0.0f;

            if (mode_stroke_over_fill) {
                if (stroke_alpha > 0.0f) {
                    final_alpha = stroke_alpha + fill_alpha * (1.0f - stroke_alpha);
                    if (final_alpha > 0.0f) {
                        final_r = (strokeColor.r * stroke_alpha + fillColor.r * fill_alpha * (1.0f - stroke_alpha)) / 255.0f;
                        final_g = (strokeColor.g * stroke_alpha + fillColor.g * fill_alpha * (1.0f - stroke_alpha)) / 255.0f;
                        final_b = (strokeColor.b * stroke_alpha + fillColor.b * fill_alpha * (1.0f - stroke_alpha)) / 255.0f;
                    }
                }
                else {
                    final_alpha = fill_alpha;
                    final_r = fillColor.r / 255.0f;
                    final_g = fillColor.g / 255.0f;
                    final_b = fillColor.b / 255.0f;
                }
            }
            else if (mode_only_stroke) {
                final_alpha = stroke_alpha;
                final_r = strokeColor.r / 255.0f;
                final_g = strokeColor.g / 255.0f;
                final_b = strokeColor.b / 255.0f;
            }
            else {
                final_alpha = fill_alpha;
                final_r = fillColor.r / 255.0f;
                final_g = fillColor.g / 255.0f;
                final_b = fillColor.b / 255.0f;
            }

            // 应用混合
            if (final_alpha > 0.0f) {
                pa2d::Color src;
                src.r = static_cast<uint8_t>(std::min(255.0f, final_r * 255.0f));
                src.g = static_cast<uint8_t>(std::min(255.0f, final_g * 255.0f));
                src.b = static_cast<uint8_t>(std::min(255.0f, final_b * 255.0f));
                src.a = static_cast<uint8_t>(std::min(255.0f, final_alpha * 255.0f));

                row[px] = Blend(src, row[px]);
            }
        }
    }
}

/**
 * @brief 渲染花朵到画布上
 */
void drawFlower(
    pa2d::Canvas& src,
    float cx, float cy,           // 中心坐标
    float radius,                 // 半径
    float angle,               // 旋转角度 (弧度)
    const pa2d::Color& fillColor,
    const pa2d::Color& strokeColor,
    float strokeWidth
) {
    flower_impl(
        src.getBuffer(),
        cx, cy,
        radius,
        radius / 2,
        angle,
        fillColor,
        strokeColor,
        strokeWidth
    );
}

/**
 * @brief 使用样式对象渲染花朵
 */
void drawFlower(
    pa2d::Canvas& src,
    float cx, float cy,
    float radius,
    float angle,
    const pa2d::Style style
) {
    flower_impl(
        src.getBuffer(),
        cx, cy,
        radius,
        radius / 2,  // 内半径
        angle,       // 角度（弧度）
        style.fill_,
        style.stroke_,
        style.width_
    );
}