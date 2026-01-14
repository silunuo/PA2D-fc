#include"../include/draw.h"
#include"internal/blend_utils.h"

using namespace pa2d::utils;
using namespace pa2d::utils::simd;

namespace pa2d {
    void sector(
        pa2d::Buffer& buffer,
        float cx, float cy, float radius,
        float startAngleDeg, float endAngleDeg,
        const pa2d::Color& fillColor,
        const pa2d::Color& strokeColor,
        float strokeWidth,
        bool drawArc,
        bool drawRadialEdges
    ) {
        // --- 1. 参数校验和预处理 ---
        if (!buffer.isValid() || radius <= 0) return;

        const float finalFillOpacity = fillColor.a * (1.0f / 255.0f);
        const float finalStrokeOpacity = strokeColor.a * (1.0f / 255.0f);

        const bool drawFill = (finalFillOpacity > 0.0f);
        const bool drawStroke = (finalStrokeOpacity > 0.0f) && (strokeWidth > 0.0f);

        if (!drawFill && !drawStroke) return;

        const bool mode_stroke_over_fill = drawStroke && drawFill;
        const bool mode_only_stroke = drawStroke && !drawFill;

        const float arcStrokeWidth = strokeWidth == 0 ? 0 : strokeWidth;
        const float edgeStrokeWidth = strokeWidth == 0 ? 0 : strokeWidth;

        const float halfArcStrokeWidth = arcStrokeWidth * 0.5f;
        const float halfEdgeStrokeWidth = edgeStrokeWidth * 0.5f;
        const float EPSILON = GEOMETRY_EPSILON;

        // 角度处理：规范化
        float startAngle = ((std::fmod(startAngleDeg, 360.0f)) * GEOMETRY_PI / 180.0f);
        float endAngle = ((std::fmod(endAngleDeg, 360.0f)) * GEOMETRY_PI / 180.0f);

        // 处理endAngle可能小于startAngle的情况
        if (endAngleDeg < startAngleDeg) {
            // 如果结束角度小于开始角度，加上360度
            endAngle += 2.0f * GEOMETRY_PI;
        }

        // 确保角度范围合理
        startAngle = std::fmod(startAngle, 2.0f * GEOMETRY_PI);
        if (startAngle < 0.0f) startAngle += 2.0f * GEOMETRY_PI;

        float angleDiff = endAngle - startAngle;

        if (angleDiff > 2.0f * GEOMETRY_PI - EPSILON) {
            // 如果角度差接近2π，设为完整圆
            angleDiff = 2.0f * GEOMETRY_PI;
        }

        const bool isFullCircle = (angleDiff >= 2.0f * GEOMETRY_PI - EPSILON);
        if (isFullCircle) {
            angleDiff = 2.0f * GEOMETRY_PI;
        }

        // 计算结束角度
        const float actualEndAngle = startAngle + angleDiff;

        // --- 2. 边界计算和裁剪 ---
        const float maxExt = radius + halfArcStrokeWidth + 1.0f;

        int minX = static_cast<int>(std::floor(cx - maxExt));
        int maxX = static_cast<int>(std::ceil(cx + maxExt));
        int minY = static_cast<int>(std::floor(cy - maxExt));
        int maxY = static_cast<int>(std::ceil(cy + maxExt));

        minX = std::max(0, minX);
        maxX = std::min(buffer.width - 1, maxX);
        minY = std::max(0, minY);
        maxY = std::min(buffer.height - 1, maxY);

        if (minX > maxX || minY > maxY) return;

        // --- 3. SIMD 常量准备 ---
        const __m256 cx_v = _mm256_set1_ps(cx);
        const __m256 cy_v = _mm256_set1_ps(cy);
        const __m256 radius_v = _mm256_set1_ps(radius);

        const __m256 halfArcStrokeWidth_v = _mm256_set1_ps(halfArcStrokeWidth);

        const float edge_sdf_offset_f = halfArcStrokeWidth * 0.5f;
        const __m256 edge_sdf_offset_v = _mm256_set1_ps(edge_sdf_offset_f);

        // 角度常量
        const __m256 start_cos_v = _mm256_set1_ps(std::cos(startAngle));
        const __m256 start_sin_v = _mm256_set1_ps(std::sin(startAngle));
        const __m256 end_cos_v = _mm256_set1_ps(std::cos(actualEndAngle));
        const __m256 end_sin_v = _mm256_set1_ps(std::sin(actualEndAngle));

        // 颜色常量
        const __m256 fillA_v = _mm256_set1_ps(finalFillOpacity);
        const __m256 strokeA_v = _mm256_set1_ps(finalStrokeOpacity);
        const __m256 fillR_v = _mm256_set1_ps(fillColor.r * (1.0f / 255.0f));
        const __m256 fillG_v = _mm256_set1_ps(fillColor.g * (1.0f / 255.0f));
        const __m256 fillB_v = _mm256_set1_ps(fillColor.b * (1.0f / 255.0f));
        const __m256 strokeR_v = _mm256_set1_ps(strokeColor.r * (1.0f / 255.0f));
        const __m256 strokeG_v = _mm256_set1_ps(strokeColor.g * (1.0f / 255.0f));
        const __m256 strokeB_v = _mm256_set1_ps(strokeColor.b * (1.0f / 255.0f));

        // 光线SDF所需的角度差常量
        const __m256 angleDiff_v = _mm256_set1_ps(angleDiff);
        const __m256 angleDiff_cos_v = _mm256_set1_ps(std::cos(angleDiff));
        const __m256 angleDiff_sin_v = _mm256_set1_ps(std::sin(angleDiff));

        // SSE常量
        const __m128 cx_sse = _mm_set1_ps(cx);
        const __m128 cy_sse = _mm_set1_ps(cy);
        const __m128 radius_sse = _mm_set1_ps(radius);
        const __m128 halfArcStrokeWidth_sse = _mm_set1_ps(halfArcStrokeWidth);
        const __m128 edge_sdf_offset_sse = _mm_set1_ps(edge_sdf_offset_f);

        // SSE 角度常量
        const __m128 start_cos_sse = _mm_set1_ps(std::cos(startAngle));
        const __m128 start_sin_sse = _mm_set1_ps(std::sin(startAngle));
        const __m128 end_cos_sse = _mm_set1_ps(std::cos(actualEndAngle));
        const __m128 end_sin_sse = _mm_set1_ps(std::sin(actualEndAngle));

        // SSE 颜色常量
        const __m128 fillA_sse = _mm_set1_ps(finalFillOpacity);
        const __m128 strokeA_sse = _mm_set1_ps(finalStrokeOpacity);
        const __m128 fillR_sse = _mm_set1_ps(fillColor.r * (1.0f / 255.0f));
        const __m128 fillG_sse = _mm_set1_ps(fillColor.g * (1.0f / 255.0f));
        const __m128 fillB_sse = _mm_set1_ps(fillColor.b * (1.0f / 255.0f));
        const __m128 strokeR_sse = _mm_set1_ps(strokeColor.r * (1.0f / 255.0f));
        const __m128 strokeG_sse = _mm_set1_ps(strokeColor.g * (1.0f / 255.0f));
        const __m128 strokeB_sse = _mm_set1_ps(strokeColor.b * (1.0f / 255.0f));

        // 光线SDF所需的角度差常量
        const __m128 angleDiff_sse = _mm_set1_ps(angleDiff);
        const __m128 angleDiff_cos_sse = _mm_set1_ps(std::cos(angleDiff));
        const __m128 angleDiff_sin_sse = _mm_set1_ps(std::sin(angleDiff));

        // --- 4. 自定义绝对值函数 ---
        auto mm256_abs_ps = [](__m256 x) -> __m256 {
            __m256 sign_mask = _mm256_set1_ps(-0.0f);
            return _mm256_andnot_ps(sign_mask, x);
        };

        auto mm_abs_ps = [](__m128 x) -> __m128 {
            __m128 sign_mask = _mm_set1_ps(-0.0f);
            return _mm_andnot_ps(sign_mask, x);
        };

        // --- 5. 修正的 SDF 函数 ---

        // ** 填充 SDF (AVX2) **
        auto sector_sdf_avx2 = [&](__m256 dx, __m256 dy) -> __m256 {
            __m256 dist_sq = _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy));
            __m256 dist = _mm256_sqrt_ps(dist_sq);
            __m256 circle_sdf = _mm256_sub_ps(dist, radius_v);

            if (isFullCircle) return circle_sdf;

            // 旋转坐标到起始角度为0
            __m256 px_rot = _mm256_add_ps(_mm256_mul_ps(dx, start_cos_v), _mm256_mul_ps(dy, start_sin_v));
            __m256 py_rot = _mm256_sub_ps(_mm256_mul_ps(dy, start_cos_v), _mm256_mul_ps(dx, start_sin_v));

            // 计算相对角度
            __m256 angle = _mm256_atan2_ps(py_rot, px_rot);

            // 规范化角度到 [0, 2π)
            __m256 mask_neg = _mm256_cmp_ps(angle, ZERO_256, _CMP_LT_OQ);
            __m256 two_pi = _mm256_set1_ps(2.0f * GEOMETRY_PI);
            angle = _mm256_add_ps(angle, _mm256_and_ps(two_pi, mask_neg));

            // 检查是否在角度范围内 [0, angleDiff]
            __m256 in_range = _mm256_and_ps(
                _mm256_cmp_ps(angle, ZERO_256, _CMP_GE_OQ),
                _mm256_cmp_ps(angle, _mm256_set1_ps(angleDiff), _CMP_LE_OQ)
            );

            // 计算到边缘的距离 (用于扇形外部)
            __m256 dist_to_start = _mm256_mul_ps(dist, mm256_abs_ps(angle));
            __m256 dist_to_end = _mm256_mul_ps(dist, mm256_abs_ps(_mm256_sub_ps(angle, _mm256_set1_ps(angleDiff))));
            __m256 dist_to_edge = _mm256_min_ps(dist_to_start, dist_to_end);

            __m256 outside_sdf = _mm256_max_ps(circle_sdf, dist_to_edge);

            return _mm256_blendv_ps(outside_sdf, circle_sdf, in_range);
        };

        // ** 弧线 SDF (AVX2) **
        auto arc_sdf_avx2 = [&](__m256 dx, __m256 dy) -> __m256 {
            if (!drawArc || isFullCircle) return _mm256_set1_ps(1000.0f);

            __m256 dist_sq = _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy));
            __m256 dist = _mm256_sqrt_ps(dist_sq);

            // 圆弧的SDF是到圆环的距离
            __m256 arc_sdf = mm256_abs_ps(_mm256_sub_ps(dist, radius_v));

            // 使用与填充SDF相同的旋转方法来检查角度
            __m256 px_rot = _mm256_add_ps(_mm256_mul_ps(dx, start_cos_v), _mm256_mul_ps(dy, start_sin_v));
            __m256 py_rot = _mm256_sub_ps(_mm256_mul_ps(dy, start_cos_v), _mm256_mul_ps(dx, start_sin_v));
            __m256 angle_rel = _mm256_atan2_ps(py_rot, px_rot);

            __m256 mask_neg = _mm256_cmp_ps(angle_rel, ZERO_256, _CMP_LT_OQ);
            __m256 two_pi = _mm256_set1_ps(2.0f * GEOMETRY_PI);
            angle_rel = _mm256_add_ps(angle_rel, _mm256_and_ps(two_pi, mask_neg));

            // 检查是否在角度范围内 [0, angleDiff]
            __m256 in_angle_range = _mm256_and_ps(
                _mm256_cmp_ps(angle_rel, ZERO_256, _CMP_GE_OQ),
                _mm256_cmp_ps(angle_rel, _mm256_set1_ps(angleDiff), _CMP_LE_OQ)
            );

            // 对于角度范围外的点，返回大值
            return _mm256_blendv_ps(_mm256_set1_ps(1000.0f), arc_sdf, in_angle_range);
        };

        // ** 径向边 SDF (AVX2) **
        auto radial_edges_sdf_avx2 = [&](__m256 dx, __m256 dy) -> __m256 {
            if (!drawRadialEdges || isFullCircle) return _mm256_set1_ps(1000.0f);

            // --- 1. 旋转坐标 ---
            __m256 px_rot = _mm256_add_ps(_mm256_mul_ps(dx, start_cos_v), _mm256_mul_ps(dy, start_sin_v));
            __m256 py_rot = _mm256_sub_ps(_mm256_mul_ps(dy, start_cos_v), _mm256_mul_ps(dx, start_sin_v));

            // --- 2. 检查是否在扇形角内部 ---
            __m256 angle_rel = _mm256_atan2_ps(py_rot, px_rot);
            __m256 mask_neg = _mm256_cmp_ps(angle_rel, ZERO_256, _CMP_LT_OQ);
            __m256 two_pi = _mm256_set1_ps(2.0f * GEOMETRY_PI);
            angle_rel = _mm256_add_ps(angle_rel, _mm256_and_ps(two_pi, mask_neg));

            __m256 in_angle_range = _mm256_and_ps(
                _mm256_cmp_ps(angle_rel, ZERO_256, _CMP_GE_OQ),
                _mm256_cmp_ps(angle_rel, angleDiff_v, _CMP_LE_OQ)
            );

            // --- 3. 检查是否在半径范围内 ---
            __m256 dist_sq = _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy));
            __m256 dist = _mm256_sqrt_ps(dist_sq);
            __m256 in_radius_range = _mm256_cmp_ps(dist, radius_v, _CMP_LE_OQ);

            // 组合掩码：必须在角度和半径内部
            __m256 valid_edge = _mm256_and_ps(in_angle_range, in_radius_range);

            // --- 4. 计算到边缘光线的SDF ---
            __m256 dot_p_v_start_rot = px_rot;
            __m256 dist_to_start_line_rot = mm256_abs_ps(py_rot);
            __m256 behind_start_rot = _mm256_cmp_ps(dot_p_v_start_rot, ZERO_256, _CMP_LT_OQ);
            __m256 sdf_start_ray_rot = _mm256_blendv_ps(dist_to_start_line_rot, dist, behind_start_rot);

            __m256 dot_p_v_end_rot = _mm256_add_ps(
                _mm256_mul_ps(px_rot, angleDiff_cos_v),
                _mm256_mul_ps(py_rot, angleDiff_sin_v)
            );
            __m256 dist_to_end_line_rot = mm256_abs_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(px_rot, _mm256_sub_ps(ZERO_256, angleDiff_sin_v)),
                    _mm256_mul_ps(py_rot, angleDiff_cos_v)
                )
            );
            __m256 behind_end_rot = _mm256_cmp_ps(dot_p_v_end_rot, ZERO_256, _CMP_LT_OQ);
            __m256 sdf_end_ray_rot = _mm256_blendv_ps(dist_to_end_line_rot, dist, behind_end_rot);

            // 两个光线SDF的最小值
            __m256 dist_to_edges = _mm256_min_ps(sdf_start_ray_rot, sdf_end_ray_rot);

            __m256 modified_dist_to_edges = _mm256_sub_ps(dist_to_edges, edge_sdf_offset_v);

            // --- 5. 混合 ---
            return _mm256_blendv_ps(_mm256_set1_ps(1000.0f), modified_dist_to_edges, valid_edge);
        };

        // ** 组合描边 SDF (AVX2) **
        auto combined_stroke_sdf_avx2 = [&](__m256 dx, __m256 dy) -> __m256 {
            if (!drawStroke) return _mm256_set1_ps(1000.0f);

            __m256 arc_sdf = _mm256_set1_ps(1000.0f);
            __m256 edges_sdf = _mm256_set1_ps(1000.0f);

            if (drawArc) {
                arc_sdf = arc_sdf_avx2(dx, dy);
            }

            if (drawRadialEdges && !isFullCircle) {
                edges_sdf = radial_edges_sdf_avx2(dx, dy);
            }

            // 取最小的SDF值（最接近描边的距离）
            return _mm256_min_ps(arc_sdf, edges_sdf);
        };

        // ** 填充 SDF (SSE) **
        auto sector_sdf_sse = [&](__m128 dx, __m128 dy) -> __m128 {
            __m128 dist_sq = _mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy));
            __m128 dist = _mm_sqrt_ps(dist_sq);
            __m128 circle_sdf = _mm_sub_ps(dist, radius_sse);

            if (isFullCircle) return circle_sdf;

            // 旋转到起始角度为0
            __m128 px_rot = _mm_add_ps(_mm_mul_ps(dx, start_cos_sse), _mm_mul_ps(dy, start_sin_sse));
            __m128 py_rot = _mm_sub_ps(_mm_mul_ps(dy, start_cos_sse), _mm_mul_ps(dx, start_sin_sse));

            // 计算角度
            __m128 angle = _mm_atan2_ps(py_rot, px_rot);

            // 规范化角度
            __m128 mask_neg = _mm_cmplt_ps(angle, ZERO_128);
            __m128 two_pi = _mm_set1_ps(2.0f * GEOMETRY_PI);
            angle = _mm_add_ps(angle, _mm_and_ps(two_pi, mask_neg));

            // 检查是否在角度范围内 [0, angleDiff]
            __m128 in_range = _mm_and_ps(
                _mm_cmpge_ps(angle, ZERO_128),
                _mm_cmple_ps(angle, _mm_set1_ps(angleDiff))
            );

            // 计算到边缘的距离
            __m128 dist_to_start = _mm_mul_ps(dist, mm_abs_ps(angle));
            __m128 dist_to_end = _mm_mul_ps(dist, mm_abs_ps(_mm_sub_ps(angle, _mm_set1_ps(angleDiff))));
            __m128 dist_to_edge = _mm_min_ps(dist_to_start, dist_to_end);

            __m128 outside_sdf = _mm_max_ps(circle_sdf, dist_to_edge);

            return _mm_or_ps(
                _mm_and_ps(circle_sdf, in_range),
                _mm_andnot_ps(in_range, outside_sdf)
            );
        };

        // ** 弧线 SDF (SSE) **
        auto arc_sdf_sse = [&](__m128 dx, __m128 dy) -> __m128 {
            if (!drawArc || isFullCircle) return _mm_set1_ps(1000.0f);

            __m128 dist_sq = _mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy));
            __m128 dist = _mm_sqrt_ps(dist_sq);
            __m128 arc_sdf = mm_abs_ps(_mm_sub_ps(dist, radius_sse));

            // 使用旋转方法检查角度
            __m128 px_rot = _mm_add_ps(_mm_mul_ps(dx, start_cos_sse), _mm_mul_ps(dy, start_sin_sse));
            __m128 py_rot = _mm_sub_ps(_mm_mul_ps(dy, start_cos_sse), _mm_mul_ps(dx, start_sin_sse));
            __m128 angle_rel = _mm_atan2_ps(py_rot, px_rot);

            __m128 mask_neg = _mm_cmplt_ps(angle_rel, ZERO_128);
            __m128 two_pi = _mm_set1_ps(2.0f * GEOMETRY_PI);
            angle_rel = _mm_add_ps(angle_rel, _mm_and_ps(two_pi, mask_neg));

            __m128 in_angle_range = _mm_and_ps(
                _mm_cmpge_ps(angle_rel, ZERO_128),
                _mm_cmple_ps(angle_rel, _mm_set1_ps(angleDiff))
            );

            return _mm_or_ps(
                _mm_and_ps(arc_sdf, in_angle_range),
                _mm_andnot_ps(in_angle_range, _mm_set1_ps(1000.0f))
            );
        };

        // ** 径向边 SDF (SSE) **
        auto radial_edges_sdf_sse = [&](__m128 dx, __m128 dy) -> __m128 {
            if (!drawRadialEdges || isFullCircle) return _mm_set1_ps(1000.0f);

            // --- 1. 旋转坐标 ---
            __m128 px_rot = _mm_add_ps(_mm_mul_ps(dx, start_cos_sse), _mm_mul_ps(dy, start_sin_sse));
            __m128 py_rot = _mm_sub_ps(_mm_mul_ps(dy, start_cos_sse), _mm_mul_ps(dx, start_sin_sse));

            // --- 2. 检查是否在扇形角内部 ---
            __m128 angle_rel = _mm_atan2_ps(py_rot, px_rot);
            __m128 mask_neg = _mm_cmplt_ps(angle_rel, ZERO_128);
            __m128 two_pi = _mm_set1_ps(2.0f * GEOMETRY_PI);
            angle_rel = _mm_add_ps(angle_rel, _mm_and_ps(two_pi, mask_neg));

            __m128 in_angle_range = _mm_and_ps(
                _mm_cmpge_ps(angle_rel, ZERO_128),
                _mm_cmple_ps(angle_rel, angleDiff_sse)
            );

            // --- 3. 检查是否在半径范围内 ---
            __m128 dist_sq = _mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy));
            __m128 dist = _mm_sqrt_ps(dist_sq);
            __m128 in_radius_range = _mm_cmple_ps(dist, radius_sse);

            // 组合掩码
            __m128 valid_edge = _mm_and_ps(in_angle_range, in_radius_range);

            // --- 4. 计算到边缘光线的SDF ---
            __m128 dot_p_v_start_rot = px_rot;
            __m128 dist_to_start_line_rot = mm_abs_ps(py_rot);
            __m128 behind_start_rot = _mm_cmplt_ps(dot_p_v_start_rot, ZERO_128);
            __m128 sdf_start_ray_rot = _mm_or_ps(
                _mm_and_ps(dist, behind_start_rot),
                _mm_andnot_ps(behind_start_rot, dist_to_start_line_rot)
            );

            __m128 dot_p_v_end_rot = _mm_add_ps(
                _mm_mul_ps(px_rot, angleDiff_cos_sse),
                _mm_mul_ps(py_rot, angleDiff_sin_sse)
            );
            __m128 dist_to_end_line_rot = mm_abs_ps(
                _mm_add_ps(
                    _mm_mul_ps(px_rot, _mm_sub_ps(ZERO_128, angleDiff_sin_sse)),
                    _mm_mul_ps(py_rot, angleDiff_cos_sse)
                )
            );
            __m128 behind_end_rot = _mm_cmplt_ps(dot_p_v_end_rot, ZERO_128);
            __m128 sdf_end_ray_rot = _mm_or_ps(
                _mm_and_ps(dist, behind_end_rot),
                _mm_andnot_ps(behind_end_rot, dist_to_end_line_rot)
            );

            // 两个光线SDF的最小值
            __m128 dist_to_edges = _mm_min_ps(sdf_start_ray_rot, sdf_end_ray_rot);

            __m128 modified_dist_to_edges = _mm_sub_ps(dist_to_edges, edge_sdf_offset_sse);

            // --- 5. 混合 ---
            return _mm_or_ps(
                _mm_and_ps(modified_dist_to_edges, valid_edge),
                _mm_andnot_ps(valid_edge, _mm_set1_ps(1000.0f))
            );
        };

        // ** 组合描边 SDF (SSE) **
        auto combined_stroke_sdf_sse = [&](__m128 dx, __m128 dy) -> __m128 {
            if (!drawStroke) return _mm_set1_ps(1000.0f);

            __m128 arc_sdf = _mm_set1_ps(1000.0f);
            __m128 edges_sdf = _mm_set1_ps(1000.0f);

            if (drawArc) {
                arc_sdf = arc_sdf_sse(dx, dy);
            }

            if (drawRadialEdges && !isFullCircle) {
                edges_sdf = radial_edges_sdf_sse(dx, dy);
            }

            return _mm_min_ps(arc_sdf, edges_sdf);
        };

        // ** 标量版本 **
        auto sector_sdf_scalar = [&](float dx, float dy) -> float {
            float dist = std::sqrt(dx * dx + dy * dy);
            float circle_sdf = dist - radius;

            if (isFullCircle) return circle_sdf;

            float angle = std::atan2(dy, dx);
            if (angle < 0) angle += 2.0f * GEOMETRY_PI;

            float angle_from_start = angle - startAngle;
            if (angle_from_start < 0) angle_from_start += 2.0f * GEOMETRY_PI;

            bool in_sector = (angle_from_start <= angleDiff);

            if (in_sector) {
                return circle_sdf;
            }
            else {
                float dist_angle_to_start = std::abs(angle - startAngle);
                float dist_angle_to_end = std::abs(angle - actualEndAngle);

                dist_angle_to_start = std::min(dist_angle_to_start, 2.0f * GEOMETRY_PI - dist_angle_to_start);
                dist_angle_to_end = std::min(dist_angle_to_end, 2.0f * GEOMETRY_PI - dist_angle_to_end);

                float min_angle_dist = std::min(dist_angle_to_start, dist_angle_to_end);
                float dist_to_edge = dist * min_angle_dist * 0.5f;

                return std::max(circle_sdf, dist_to_edge);
            }
            };

        // ** 标量弧线 SDF **
        auto arc_sdf_scalar = [&](float dx, float dy) -> float {
            if (!drawArc || isFullCircle) return 1000.0f;

            float dist = std::sqrt(dx * dx + dy * dy);
            float arc_sdf = std::abs(dist - radius);

            // 使用旋转方法检查角度
            float px_rot = dx * std::cos(startAngle) + dy * std::sin(startAngle);
            float py_rot = dy * std::cos(startAngle) - dx * std::sin(startAngle);
            float angle_rel = std::atan2(py_rot, px_rot);
            if (angle_rel < 0.0f) angle_rel += 2.0f * GEOMETRY_PI;

            bool in_angle_range = (angle_rel >= 0.0f && angle_rel <= angleDiff);

            return in_angle_range ? arc_sdf : 1000.0f;
            };

        // ** 标量径向边 SDF **
        auto radial_edges_sdf_scalar = [&](float dx, float dy) -> float {
            if (!drawRadialEdges || isFullCircle) return 1000.0f;

            // --- 1. 旋转坐标 ---
            const float s_cos = std::cos(startAngle);
            const float s_sin = std::sin(startAngle);
            float px_rot = dx * s_cos + dy * s_sin;
            float py_rot = dy * s_cos - dx * s_sin;

            // --- 2. 检查是否在扇形角内部 ---
            float angle_rel = std::atan2(py_rot, px_rot);
            if (angle_rel < 0.0f) angle_rel += 2.0f * GEOMETRY_PI;

            bool in_angle_range = (angle_rel >= -EPSILON && angle_rel <= angleDiff + EPSILON);

            // --- 3. 检查是否在半径范围内 ---
            float dist = std::sqrt(dx * dx + dy * dy);
            bool in_radius_range = (dist <= radius + EPSILON);

            // 组合掩码
            if (!in_angle_range || !in_radius_range) {
                return 1000.0f;
            }

            // --- 4. 计算到边缘光线的SDF ---
            float dot_p_v_start_rot = px_rot;
            float dist_to_start_line_rot = std::abs(py_rot);
            float sdf_start_ray_rot = (dot_p_v_start_rot < 0.0f) ? dist : dist_to_start_line_rot;

            const float a_cos = std::cos(angleDiff);
            const float a_sin = std::sin(angleDiff);

            float dot_p_v_end_rot = px_rot * a_cos + py_rot * a_sin;
            float dist_to_end_line_rot = std::abs(px_rot * (-a_sin) + py_rot * a_cos);
            float sdf_end_ray_rot = (dot_p_v_end_rot < 0.0f) ? dist : dist_to_end_line_rot;

            // 两个光线SDF的最小值
            float dist_to_edges = std::min(sdf_start_ray_rot, sdf_end_ray_rot);

            return dist_to_edges - edge_sdf_offset_f;
            };

        auto combined_stroke_sdf_scalar = [&](float dx, float dy) -> float {
            if (!drawStroke) return 1000.0f;

            float arc_sdf = 1000.0f;
            float edges_sdf = 1000.0f;

            if (drawArc) {
                arc_sdf = arc_sdf_scalar(dx, dy);
            }

            if (drawRadialEdges && !isFullCircle) {
                edges_sdf = radial_edges_sdf_scalar(dx, dy);
            }

            return std::min(arc_sdf, edges_sdf);
            };

        // --- 6. 主绘制循环 ---
        for (int py = minY; py <= maxY; ++py) {
            const float fy = static_cast<float>(py) + 0.5f;
            pa2d::Color* row = &buffer.at(0, py);

            const __m256 py_v = _mm256_set1_ps(fy);
            const __m128 py_v_sse = _mm_set1_ps(fy);

            int px = minX;

            // AVX2处理 (8像素)
            for (; px <= maxX - 7; px += 8) {
                __m256i xBase = _mm256_setr_epi32(px, px + 1, px + 2, px + 3, px + 4, px + 5, px + 6, px + 7);
                __m256 px_v = _mm256_add_ps(_mm256_cvtepi32_ps(xBase), HALF_PIXEL_256);

                __m256 dx = _mm256_sub_ps(px_v, cx_v);
                __m256 dy = _mm256_sub_ps(py_v, cy_v);

                // 计算SDF
                __m256 sdf_fill = sector_sdf_avx2(dx, dy);
                __m256 sdf_stroke = combined_stroke_sdf_avx2(dx, dy);

                // 填充计算
                __m256 fill_alpha = ZERO_256;
                if (drawFill) {
                    __m256 t = _mm256_div_ps(_mm256_sub_ps(ANTIALIAS_RANGE_256, sdf_fill), ANTIALIAS_RANGE_256);
                    t = _mm256_max_ps(ZERO_256, _mm256_min_ps(ONE_256, t));
                    fill_alpha = _mm256_mul_ps(t, fillA_v);
                }

                __m256 stroke_alpha = ZERO_256;
                if (drawStroke) {
                    // 统一的有效半宽度
                    __m256 effective_half_stroke_width = halfArcStrokeWidth_v;

                    __m256 t_stroke = _mm256_div_ps(_mm256_sub_ps(effective_half_stroke_width, sdf_stroke), ANTIALIAS_RANGE_256);
                    t_stroke = _mm256_max_ps(ZERO_256, _mm256_min_ps(ONE_256, t_stroke));
                    stroke_alpha = _mm256_mul_ps(t_stroke, strokeA_v);
                }

                __m256 final_alpha;
                __m256 final_r, final_g, final_b;

                if (mode_stroke_over_fill) {
                    // 混合逻辑: A_new = A_stroke + A_fill * (1 - A_stroke)
                    // C_new = (C_stroke * A_stroke + C_fill * A_fill * (1 - A_stroke)) / A_new
                    __m256 one_minus_stroke = _mm256_sub_ps(ONE_256, stroke_alpha);
                    __m256 fill_contribution_alpha = _mm256_mul_ps(fill_alpha, one_minus_stroke);
                    final_alpha = _mm256_add_ps(stroke_alpha, fill_contribution_alpha);

                    // C_stroke * A_stroke: 描边像素对最终颜色的贡献 (已预乘)
                    __m256 stroke_premult_r = _mm256_mul_ps(strokeR_v, stroke_alpha);
                    __m256 stroke_premult_g = _mm256_mul_ps(strokeG_v, stroke_alpha);
                    __m256 stroke_premult_b = _mm256_mul_ps(strokeB_v, stroke_alpha);

                    // C_fill * A_fill * (1 - A_stroke): 填充像素对最终颜色的贡献 (已预乘)
                    __m256 fill_premult_r = _mm256_mul_ps(fillR_v, fill_contribution_alpha);
                    __m256 fill_premult_g = _mm256_mul_ps(fillG_v, fill_contribution_alpha);
                    __m256 fill_premult_b = _mm256_mul_ps(fillB_v, fill_contribution_alpha);

                    // 总颜色贡献 (预乘)
                    __m256 total_premult_r = _mm256_add_ps(stroke_premult_r, fill_premult_r);
                    __m256 total_premult_g = _mm256_add_ps(stroke_premult_g, fill_premult_g);
                    __m256 total_premult_b = _mm256_add_ps(stroke_premult_b, fill_premult_b);

                    // 解除预乘: C_new = C_premult / A_new
                    __m256 final_alpha_recip = _mm256_div_ps(ONE_256, final_alpha);

                    final_r = _mm256_mul_ps(total_premult_r, final_alpha_recip);
                    final_g = _mm256_mul_ps(total_premult_g, final_alpha_recip);
                    final_b = _mm256_mul_ps(total_premult_b, final_alpha_recip);
                }
                else if (mode_only_stroke) {
                    final_alpha = stroke_alpha;
                    final_r = strokeR_v;
                    final_g = strokeG_v;
                    final_b = strokeB_v;
                }
                else {
                    // 仅填充模式
                    final_alpha = fill_alpha;
                    final_r = fillR_v;
                    final_g = fillG_v;
                    final_b = fillB_v;
                }

                // 写入像素
                __m256 mask = _mm256_cmp_ps(final_alpha, ZERO_256, _CMP_GT_OQ);
                if (!_mm256_testz_ps(mask, mask)) {
                    __m256i dest = _mm256_loadu_si256((__m256i*) & row[px]);
                    __m256i rgba = blend_pixels_avx(final_alpha, dest, final_r, final_g, final_b);
                    rgba = _mm256_blendv_epi8(dest, rgba, _mm256_castps_si256(mask));
                    _mm256_storeu_si256((__m256i*) & row[px], rgba);
                }
            }

            // SSE处理 (4像素)
            for (; px <= maxX - 3; px += 4) {
                __m128i xBase = _mm_setr_epi32(px, px + 1, px + 2, px + 3);
                __m128 px_v_sse = _mm_add_ps(_mm_cvtepi32_ps(xBase), HALF_PIXEL_128);

                __m128 dx = _mm_sub_ps(px_v_sse, cx_sse);
                __m128 dy = _mm_sub_ps(py_v_sse, cy_sse);

                // 计算SDF
                __m128 sdf_fill = sector_sdf_sse(dx, dy);
                __m128 sdf_stroke = combined_stroke_sdf_sse(dx, dy);

                // 填充计算
                __m128 fill_alpha = ZERO_128;
                if (drawFill) {
                    __m128 t = _mm_div_ps(_mm_sub_ps(ANTIALIAS_RANGE_128, sdf_fill), ANTIALIAS_RANGE_128);
                    t = _mm_max_ps(ZERO_128, _mm_min_ps(ONE_128, t));
                    fill_alpha = _mm_mul_ps(t, fillA_sse);
                }

                __m128 stroke_alpha = ZERO_128;
                if (drawStroke) {
                    // 统一的有效半宽度
                    __m128 effective_half_stroke_width = halfArcStrokeWidth_sse;

                    __m128 t_stroke = _mm_div_ps(_mm_sub_ps(effective_half_stroke_width, sdf_stroke), ANTIALIAS_RANGE_128);
                    t_stroke = _mm_max_ps(ZERO_128, _mm_min_ps(ONE_128, t_stroke));
                    stroke_alpha = _mm_mul_ps(t_stroke, strokeA_sse);
                }

                __m128 final_alpha;
                __m128 final_r, final_g, final_b;

                if (mode_stroke_over_fill) {
                    __m128 one_minus_stroke = _mm_sub_ps(ONE_128, stroke_alpha);
                    __m128 fill_contribution_alpha = _mm_mul_ps(fill_alpha, one_minus_stroke);
                    final_alpha = _mm_add_ps(stroke_alpha, fill_contribution_alpha);

                    // 颜色预乘
                    __m128 stroke_premult_r = _mm_mul_ps(strokeR_sse, stroke_alpha);
                    __m128 stroke_premult_g = _mm_mul_ps(strokeG_sse, stroke_alpha);
                    __m128 stroke_premult_b = _mm_mul_ps(strokeB_sse, stroke_alpha);
                    __m128 fill_premult_r = _mm_mul_ps(fillR_sse, fill_contribution_alpha);
                    __m128 fill_premult_g = _mm_mul_ps(fillG_sse, fill_contribution_alpha);
                    __m128 fill_premult_b = _mm_mul_ps(fillB_sse, fill_contribution_alpha);

                    __m128 total_premult_r = _mm_add_ps(stroke_premult_r, fill_premult_r);
                    __m128 total_premult_g = _mm_add_ps(stroke_premult_g, fill_premult_g);
                    __m128 total_premult_b = _mm_add_ps(stroke_premult_b, fill_premult_b);

                    // 解除预乘
                    __m128 final_alpha_recip = _mm_div_ps(ONE_128, final_alpha);

                    final_r = _mm_mul_ps(total_premult_r, final_alpha_recip);
                    final_g = _mm_mul_ps(total_premult_g, final_alpha_recip);
                    final_b = _mm_mul_ps(total_premult_b, final_alpha_recip);
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

                // 写入像素
                __m128 mask = _mm_cmpgt_ps(final_alpha, ZERO_128);
                if (_mm_movemask_ps(mask)) {
                    __m128i dest = _mm_loadu_si128((__m128i*) & row[px]);
                    __m128i rgba = blend_pixels_sse(final_alpha, dest, final_r, final_g, final_b);
                    rgba = _mm_blendv_epi8(dest, rgba, _mm_castps_si128(mask));
                    _mm_storeu_si128((__m128i*) & row[px], rgba);
                }
            }

            // 标量收尾处理
            for (; px <= maxX; ++px) {
                const float fx = static_cast<float>(px) + 0.5f;
                float dx = fx - cx;
                float dy = fy - cy;

                // 计算SDF
                float sdf_fill = sector_sdf_scalar(dx, dy);
                float sdf_stroke = combined_stroke_sdf_scalar(dx, dy);

                // 填充alpha
                float fill_alpha = 0.0f;
                if (drawFill) {
                    float t = (1.0f - sdf_fill) / 1.0f;
                    t = std::max(0.0f, std::min(1.0f, t));
                    fill_alpha = t * finalFillOpacity;
                }

                float stroke_alpha = 0.0f;
                if (drawStroke) {
                    // 统一的有效半宽度
                    float effective_half_stroke_width = halfArcStrokeWidth;

                    float t_stroke = (effective_half_stroke_width - sdf_stroke) / 1.0f;
                    t_stroke = std::max(0.0f, std::min(1.0f, t_stroke));
                    stroke_alpha = t_stroke * finalStrokeOpacity;
                }

                if (fill_alpha > 0.0f || stroke_alpha > 0.0f) {
                    pa2d::Color srcColor;

                    if (mode_stroke_over_fill) {
                        float one_minus_stroke = 1.0f - stroke_alpha;
                        float fill_contribution_alpha = fill_alpha * one_minus_stroke;
                        float final_alpha = stroke_alpha + fill_contribution_alpha;

                        if (final_alpha > 0.0f) {
                            // 预乘颜色
                            float pre_r = strokeColor.r * stroke_alpha + fillColor.r * fill_contribution_alpha;
                            float pre_g = strokeColor.g * stroke_alpha + fillColor.g * fill_contribution_alpha;
                            float pre_b = strokeColor.b * stroke_alpha + fillColor.b * fill_contribution_alpha;

                            // 解除预乘
                            srcColor.r = static_cast<uint8_t>(std::min(255.0f, pre_r / final_alpha));
                            srcColor.g = static_cast<uint8_t>(std::min(255.0f, pre_g / final_alpha));
                            srcColor.b = static_cast<uint8_t>(std::min(255.0f, pre_b / final_alpha));
                            srcColor.a = static_cast<uint8_t>(std::min(255.0f, final_alpha * 255.0f));
                        }
                    }
                    else if (mode_only_stroke) {
                        srcColor.r = strokeColor.r;
                        srcColor.g = strokeColor.g;
                        srcColor.b = strokeColor.b;
                        srcColor.a = static_cast<uint8_t>(std::min(255.0f, stroke_alpha * 255.0f));
                    }
                    else {
                        // 仅填充模式
                        srcColor.r = fillColor.r;
                        srcColor.g = fillColor.g;
                        srcColor.b = fillColor.b;
                        srcColor.a = static_cast<uint8_t>(std::min(255.0f, fill_alpha * 255.0f));
                    }

                    if (srcColor.a > 0) {
                        pa2d::Color& dest = row[px];
                        row[px] = Blend(srcColor, dest);
                    }
                }
            }
        }
    }
}