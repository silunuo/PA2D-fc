#include"../include/draw.h"
#include"internal/blend_utils.h"

using namespace pa2d::utils;
using namespace pa2d::utils::simd;

namespace pa2d {

    void line(
        pa2d::Buffer& buffer,
        float fx0, float fy0, float fx1, float fy1,
        const pa2d::Color& color, float strokeWidth)
    {
        // 早期退出检查
        if (!buffer.isValid() || color.a == 0 || strokeWidth <= 0) return;

        // 预计算 
        const float colorAlpha_01 = color.a * (1.0f / 255.0f);

        // 线条向量和长度
        const float dx = fx1 - fx0;
        const float dy = fy1 - fy0;
        const float length_sq = dx * dx + dy * dy;

        // 预计算倒数
        const float inv_length_sq = 1.0f / length_sq;
        const float halfWidth = strokeWidth * 0.5f;
        const float antialiasRange = 1.0f;
        const float inv_antialiasRange = 1.0f / antialiasRange;
        const float outerEdge = halfWidth + antialiasRange * 0.5f;

        if (outerEdge <= 0.0f) return;

        // 包围盒
        const float min_fx = std::min(fx0, fx1) - outerEdge;
        const float max_fx = std::max(fx0, fx1) + outerEdge;
        const float min_fy = std::min(fy0, fy1) - outerEdge;
        const float max_fy = std::max(fy0, fy1) + outerEdge;

        int minX = static_cast<int>(std::floor(min_fx));
        int maxX = static_cast<int>(std::ceil(max_fx));
        int minY = static_cast<int>(std::floor(min_fy));
        int maxY = static_cast<int>(std::ceil(max_fy));

        // 裁剪到缓冲区
        minX = std::max(0, minX);
        maxX = std::min(buffer.width - 1, maxX);
        minY = std::max(0, minY);
        maxY = std::min(buffer.height - 1, maxY);

        if (minX > maxX || minY > maxY) return;

        // --- SIMD 常量 (AVX) ---
        const __m256 v_outerEdge = _mm256_set1_ps(outerEdge);
        const __m256 lineX0 = _mm256_set1_ps(fx0);
        const __m256 lineY0 = _mm256_set1_ps(fy0);
        const __m256 lineDx = _mm256_set1_ps(dx);
        const __m256 lineDy = _mm256_set1_ps(dy);
        const __m256 v_inv_length_sq = _mm256_set1_ps(inv_length_sq);
        const __m256 srcR_avx = _mm256_set1_ps(color.r * (1.0f / 255.0f));
        const __m256 srcG_avx = _mm256_set1_ps(color.g * (1.0f / 255.0f));
        const __m256 srcB_avx = _mm256_set1_ps(color.b * (1.0f / 255.0f));
        const __m256 srcA_avx = _mm256_set1_ps(colorAlpha_01);

        // --- SIMD 常量 (SSE) ---
        const __m128 v_outerEdge_sse = _mm_set1_ps(outerEdge);
        const __m128 lineX0_sse = _mm_set1_ps(fx0);
        const __m128 lineY0_sse = _mm_set1_ps(fy0);
        const __m128 lineDx_sse = _mm_set1_ps(dx);
        const __m128 lineDy_sse = _mm_set1_ps(dy);
        const __m128 v_inv_length_sq_sse = _mm_set1_ps(inv_length_sq);
        const __m128 srcR_sse = _mm_set1_ps(color.r * (1.0f / 255.0f));
        const __m128 srcG_sse = _mm_set1_ps(color.g * (1.0f / 255.0f));
        const __m128 srcB_sse = _mm_set1_ps(color.b * (1.0f / 255.0f));
        const __m128 srcA_sse = _mm_set1_ps(colorAlpha_01);

        for (int y = minY; y <= maxY; ++y) {
            const float fy = static_cast<float>(y) + 0.5f;
            const __m256 v_fy_avx = _mm256_set1_ps(fy);
            const __m128 v_fy_sse = _mm_set1_ps(fy);
            pa2d::Color* row = &buffer.at(0, y);

            int x = minX;

            // --- AVX2 (8 像素) ---
            for (; x <= maxX - 7; x += 8) {
                // 1. 计算距离平方
                __m256i xBase = _mm256_setr_epi32(x, x + 1, x + 2, x + 3, x + 4, x + 5, x + 6, x + 7);
                __m256 v_fx = _mm256_add_ps(_mm256_cvtepi32_ps(xBase), HALF_PIXEL_256);
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

                // 2. 计算强度 (Alpha) 
                __m256 dist_approx = _mm256_sqrt_ps(distSq);

                __m256 intensity = _mm256_mul_ps(_mm256_sub_ps(v_outerEdge, dist_approx), ANTIALIAS_RANGE_INV_256);
                __m256 finalAlpha = _mm256_max_ps(ZERO_256, _mm256_min_ps(ONE_256, intensity));

                // 3. 检查 Mask
                __m256 mask = _mm256_cmp_ps(finalAlpha, ZERO_256, _CMP_GT_OQ);
                if (_mm256_testz_ps(mask, mask)) continue;

                // 4. 混合
                __m256 combinedAlpha = _mm256_mul_ps(finalAlpha, srcA_avx);
                __m256i dest = _mm256_loadu_si256((__m256i*) & row[x]);

                __m256i rgba = blend_pixels_avx(
                    combinedAlpha, dest,
                    srcR_avx, srcG_avx, srcB_avx
                );

                // 5. 写入
                rgba = _mm256_blendv_epi8(dest, rgba, _mm256_castps_si256(mask));
                _mm256_storeu_si256((__m256i*) & row[x], rgba);
            }

            // --- SSE (4 像素) ---
            for (; x <= maxX - 3; x += 4) {
                // 1. 距离平方
                __m128i xBase = _mm_setr_epi32(x, x + 1, x + 2, x + 3);
                __m128 v_fx = _mm_add_ps(_mm_cvtepi32_ps(xBase), HALF_PIXEL_128);
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

                // 2. 强度
                __m128 dist_approx = _mm_sqrt_ps(distSq);

                // 使用全局常量
                __m128 intensity = _mm_mul_ps(_mm_sub_ps(v_outerEdge_sse, dist_approx), ANTIALIAS_RANGE_INV_128);
                __m128 finalAlpha = _mm_max_ps(ZERO_128, _mm_min_ps(ONE_128, intensity));

                // 3. Mask
                __m128 mask = _mm_cmpgt_ps(finalAlpha, ZERO_128);
                if (_mm_movemask_ps(mask)) {
                    // 4. 混合
                    __m128 combinedAlpha = _mm_mul_ps(finalAlpha, srcA_sse);
                    __m128i dest = _mm_loadu_si128((__m128i*) & row[x]);

                    __m128i rgba = blend_pixels_sse(
                        combinedAlpha, dest,
                        srcR_sse, srcG_sse, srcB_sse
                    );

                    // 5. 写入
                    rgba = _mm_blendv_epi8(dest, rgba, _mm_castps_si128(mask));
                    _mm_storeu_si128((__m128i*) & row[x], rgba);
                }
            }

            // --- 标量 (剩余像素) ---
            for (; x <= maxX; ++x) {
                // 1. 距离平方
                const float px = static_cast<float>(x) + 0.5f;
                const float apx = px - fx0;
                const float apy = fy - fy0;
                float t_val = (apx * dx + apy * dy) * inv_length_sq;
                t_val = (t_val < 0.0f) ? 0.0f : (t_val > 1.0f ? 1.0f : t_val);
                const float closestX = fx0 + t_val * dx;
                const float closestY = fy0 + t_val * dy;
                const float distX = px - closestX;
                const float distY = fy - closestY;
                const float distSq = (distX * distX + distY * distY);

                // 2. 强度
                const float dist_approx = std::sqrt(distSq);

                float intensity = (outerEdge - dist_approx) * inv_antialiasRange;
                intensity = (intensity < 0.0f) ? 0.0f : (intensity > 1.0f ? 1.0f : intensity);

                // 3. 混合
                if (intensity > 0.0f) {
                    pa2d::Color& dest = row[x];
                    pa2d::Color src = color;
                    src.a = static_cast<uint8_t>(colorAlpha_01 * intensity * 255.0f);
                    row[x] = Blend(src, dest);
                }
            }
        }
    }
}