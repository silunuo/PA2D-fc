#include"../include/draw.h"
#include"internal/blend_utils.h"

using namespace pa2d::utils;
using namespace pa2d::utils::simd;

namespace pa2d {
    void polyline(
        pa2d::Buffer& buffer,
        const std::vector<Point>& points,
        const pa2d::Color& color,
        float strokeWidth,
        bool closed)
    {
        // 早期退出检查
        if (!buffer.isValid() || color.a == 0 || strokeWidth <= 0 || points.size() < 2) {
            return;
        }

        const float colorAlpha_01 = color.a * (1.0f / 255.0f);
        const float halfWidth = strokeWidth * 0.5f;

        // 使用全局常量
        const float antialiasRange = 1.0f;  // 对应 ANTIALIAS_RANGE
        const float outerEdge = halfWidth + antialiasRange;

        if (outerEdge <= 0.0f) return;

        // 计算整个折线的包围盒
        Point minPt = points[0];
        Point maxPt = points[0];

        for (const auto& pt : points) {
            minPt.x = std::min(minPt.x, pt.x);
            minPt.y = std::min(minPt.y, pt.y);
            maxPt.x = std::max(maxPt.x, pt.x);
            maxPt.y = std::max(maxPt.y, pt.y);
        }

        // 扩展包围盒考虑线宽和抗锯齿
        int minX = static_cast<int>(std::floor(minPt.x - outerEdge));
        int maxX = static_cast<int>(std::ceil(maxPt.x + outerEdge));
        int minY = static_cast<int>(std::floor(minPt.y - outerEdge));
        int maxY = static_cast<int>(std::ceil(maxPt.y + outerEdge));

        // 裁剪到缓冲区
        minX = std::max(0, minX);
        maxX = std::min(buffer.width - 1, maxX);
        minY = std::max(0, minY);
        maxY = std::min(buffer.height - 1, maxY);

        if (minX > maxX || minY > maxY) return;

        // 准备线段数据
        std::vector<std::pair<Point, Point>> segments;
        for (size_t i = 0; i < points.size() - 1; ++i) {
            segments.emplace_back(points[i], points[i + 1]);
        }
        if (closed && points.size() >= 3) {
            segments.emplace_back(points.back(), points.front());
        }

        // 预计算 SIMD 常量 - 只保留需要计算的动态变量
        const __m256 v_halfWidth = _mm256_set1_ps(halfWidth);
        const __m256 srcR_avx = _mm256_set1_ps(color.r * (1.0f / 255.0f));
        const __m256 srcG_avx = _mm256_set1_ps(color.g * (1.0f / 255.0f));
        const __m256 srcB_avx = _mm256_set1_ps(color.b * (1.0f / 255.0f));
        const __m256 srcA_avx = _mm256_set1_ps(colorAlpha_01);

        const __m128 v_halfWidth_sse = _mm_set1_ps(halfWidth);
        const __m128 srcR_sse = _mm_set1_ps(color.r * (1.0f / 255.0f));
        const __m128 srcG_sse = _mm_set1_ps(color.g * (1.0f / 255.0f));
        const __m128 srcB_sse = _mm_set1_ps(color.b * (1.0f / 255.0f));
        const __m128 srcA_sse = _mm_set1_ps(colorAlpha_01);

        for (int y = minY; y <= maxY; ++y) {
            const float fy = static_cast<float>(y) + 0.5f;  // 使用标量 0.5f
            const __m256 v_fy_avx = _mm256_set1_ps(fy);
            const __m128 v_fy_sse = _mm_set1_ps(fy);
            pa2d::Color* row = &buffer.at(0, y);

            int x = minX;

            // AVX2 处理 (8像素)
            for (; x <= maxX - 7; x += 8) {
                __m256i xBase = _mm256_setr_epi32(x, x + 1, x + 2, x + 3, x + 4, x + 5, x + 6, x + 7);
                __m256 v_fx = _mm256_add_ps(_mm256_cvtepi32_ps(xBase), HALF_PIXEL_256);

                __m256 finalAlpha = _mm256_setzero_ps();

                // 对每个线段计算贡献
                for (const auto& segment : segments) {
                    const Point& p0 = segment.first;
                    const Point& p1 = segment.second;

                    const float dx = p1.x - p0.x;
                    const float dy = p1.y - p0.y;
                    const float length_sq = dx * dx + dy * dy;

                    if (length_sq < 0.0001f) continue;

                    const float inv_length_sq = 1.0f / length_sq;

                    __m256 lineX0 = _mm256_set1_ps(p0.x);
                    __m256 lineY0 = _mm256_set1_ps(p0.y);
                    __m256 lineDx = _mm256_set1_ps(dx);
                    __m256 lineDy = _mm256_set1_ps(dy);
                    __m256 v_inv_length_sq = _mm256_set1_ps(inv_length_sq);

                    // 计算到当前线段的最小距离
                    __m256 px = _mm256_sub_ps(v_fx, lineX0);
                    __m256 py = _mm256_sub_ps(v_fy_avx, lineY0);
                    __m256 dot = _mm256_add_ps(_mm256_mul_ps(px, lineDx), _mm256_mul_ps(py, lineDy));
                    __m256 t = _mm256_mul_ps(dot, v_inv_length_sq);
                    t = _mm256_max_ps(ZERO_256, _mm256_min_ps(ONE_256, t));
                    __m256 closestX = _mm256_add_ps(lineX0, _mm256_mul_ps(t, lineDx));
                    __m256 closestY = _mm256_add_ps(lineY0, _mm256_mul_ps(t, lineDy));
                    __m256 distX = _mm256_sub_ps(v_fx, closestX);
                    __m256 distY = _mm256_sub_ps(v_fy_avx, closestY);
                    __m256 distSq = _mm256_add_ps(_mm256_mul_ps(distX, distX), _mm256_mul_ps(distY, distY));

                    // 修正：使用精确的距离计算
                    __m256 dist = _mm256_sqrt_ps(distSq);

                    // 修正：正确的抗锯齿计算
                    // 核心区域：距离 <= halfWidth，alpha = 1.0
                    // 抗锯齿区域：halfWidth < 距离 <= halfWidth + antialiasRange，alpha 线性衰减
                    __m256 innerDist = _mm256_sub_ps(dist, v_halfWidth);
                    __m256 intensity = _mm256_sub_ps(ONE_256, _mm256_mul_ps(innerDist, _mm256_rcp_ps(ANTIALIAS_RANGE_256)));
                    __m256 segmentAlpha = _mm256_max_ps(ZERO_256, _mm256_min_ps(ONE_256, intensity));

                    // 合并alpha：取最大值（避免重叠区域过度变暗）
                    finalAlpha = _mm256_max_ps(finalAlpha, segmentAlpha);
                }

                // 应用混合
                __m256 mask = _mm256_cmp_ps(finalAlpha, ZERO_256, _CMP_GT_OQ);
                if (!_mm256_testz_ps(mask, mask)) {
                    __m256 combinedAlpha = _mm256_mul_ps(finalAlpha, srcA_avx);
                    __m256i dest = _mm256_loadu_si256((__m256i*) & row[x]);

                    __m256i rgba = blend_pixels_avx(
                        combinedAlpha, dest,
                        srcR_avx, srcG_avx, srcB_avx
                    );

                    rgba = _mm256_blendv_epi8(dest, rgba, _mm256_castps_si256(mask));
                    _mm256_storeu_si256((__m256i*) & row[x], rgba);
                }
            }

            // SSE 处理 (4像素)
            for (; x <= maxX - 3; x += 4) {
                __m128i xBase = _mm_setr_epi32(x, x + 1, x + 2, x + 3);
                __m128 v_fx = _mm_add_ps(_mm_cvtepi32_ps(xBase), HALF_PIXEL_128);

                __m128 finalAlpha = _mm_setzero_ps();

                for (const auto& segment : segments) {
                    const Point& p0 = segment.first;
                    const Point& p1 = segment.second;

                    const float dx = p1.x - p0.x;
                    const float dy = p1.y - p0.y;
                    const float length_sq = dx * dx + dy * dy;

                    if (length_sq < 0.0001f) continue;

                    const float inv_length_sq = 1.0f / length_sq;

                    __m128 lineX0_sse = _mm_set1_ps(p0.x);
                    __m128 lineY0_sse = _mm_set1_ps(p0.y);
                    __m128 lineDx_sse = _mm_set1_ps(dx);
                    __m128 lineDy_sse = _mm_set1_ps(dy);
                    __m128 v_inv_length_sq_sse = _mm_set1_ps(inv_length_sq);

                    __m128 px = _mm_sub_ps(v_fx, lineX0_sse);
                    __m128 py = _mm_sub_ps(v_fy_sse, lineY0_sse);
                    __m128 dot = _mm_add_ps(_mm_mul_ps(px, lineDx_sse), _mm_mul_ps(py, lineDy_sse));
                    __m128 t = _mm_mul_ps(dot, v_inv_length_sq_sse);
                    t = _mm_max_ps(ZERO_128, _mm_min_ps(ONE_128, t));
                    __m128 closestX = _mm_add_ps(lineX0_sse, _mm_mul_ps(t, lineDx_sse));
                    __m128 closestY = _mm_add_ps(lineY0_sse, _mm_mul_ps(t, lineDy_sse));
                    __m128 distX = _mm_sub_ps(v_fx, closestX);
                    __m128 distY = _mm_sub_ps(v_fy_sse, closestY);
                    __m128 distSq = _mm_add_ps(_mm_mul_ps(distX, distX), _mm_mul_ps(distY, distY));

                    // 修正：使用精确的距离计算
                    __m128 dist = _mm_sqrt_ps(distSq);

                    // 修正：正确的抗锯齿计算
                    __m128 innerDist = _mm_sub_ps(dist, v_halfWidth_sse);
                    __m128 intensity = _mm_sub_ps(ONE_128, _mm_mul_ps(innerDist, _mm_rcp_ps(ANTIALIAS_RANGE_128)));
                    __m128 segmentAlpha = _mm_max_ps(ZERO_128, _mm_min_ps(ONE_128, intensity));

                    finalAlpha = _mm_max_ps(finalAlpha, segmentAlpha);
                }

                __m128 mask = _mm_cmpgt_ps(finalAlpha, ZERO_128);
                if (_mm_movemask_ps(mask)) {
                    __m128 combinedAlpha = _mm_mul_ps(finalAlpha, srcA_sse);
                    __m128i dest = _mm_loadu_si128((__m128i*) & row[x]);

                    __m128i rgba = blend_pixels_sse(
                        combinedAlpha, dest,
                        srcR_sse, srcG_sse, srcB_sse
                    );

                    rgba = _mm_blendv_epi8(dest, rgba, _mm_castps_si128(mask));
                    _mm_storeu_si128((__m128i*) & row[x], rgba);
                }
            }

            // 标量处理 (剩余像素)
            for (; x <= maxX; ++x) {
                const float fx = static_cast<float>(x) + 0.5f;  // 使用标量 0.5f
                float maxAlpha = 0.0f;

                for (const auto& segment : segments) {
                    const Point& p0 = segment.first;
                    const Point& p1 = segment.second;

                    const float vec_x = p1.x - p0.x;
                    const float vec_y = p1.y - p0.y;
                    const float length_sq = vec_x * vec_x + vec_y * vec_y;
                    if (length_sq < 0.0001f) continue;

                    const Point pt(fx, fy);
                    const float toPt_x = pt.x - p0.x;
                    const float toPt_y = pt.y - p0.y;
                    const float t = std::max(0.0f, std::min(1.0f, (toPt_x * vec_x + toPt_y * vec_y) / length_sq));
                    const Point closest(p0.x + vec_x * t, p0.y + vec_y * t);
                    const float dx = pt.x - closest.x;
                    const float dy = pt.y - closest.y;
                    const float dist = std::sqrt(dx * dx + dy * dy);

                    if (dist <= halfWidth) {
                        maxAlpha = 1.0f;
                    }
                    else if (dist <= halfWidth + antialiasRange) {
                        float intensity = 1.0f - (dist - halfWidth) / antialiasRange;
                        maxAlpha = std::max(maxAlpha, intensity);
                    }
                }

                if (maxAlpha > 0.0f) {
                    pa2d::Color& dest = row[x];
                    pa2d::Color src = color;
                    src.a = static_cast<uint8_t>(colorAlpha_01 * maxAlpha * 255.0f);
                    row[x] = Blend(src, dest);
                }
            }
        }
    }
}