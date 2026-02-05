// draw_heart.cpp - 心形渲染函数实现
#include "../include/draw_heart.h"
#include <algorithm>
#include <cmath>
#include <immintrin.h> // AVX2
#include <emmintrin.h> // SSE
#include "../include/blend_utils.h"

namespace heart_sdf {
    /**
     * @brief 标量版本：心形 SDF
     */
    inline float sdHeartPrecise(float px, float py) {
        // 1. 水平系数
        float x = px / 0.87f;

        // 2. 纵向比例：
        float y = -py * 1.5f + 0.4f;

        float absX = std::abs(x);

        // 3. 凹陷深一点：将 sqrtX 的系数从 0.7f 增加到 1.1f 或 1.2f
        float sqrtX = std::sqrt(absX);
        float dy = y - 1.15f * sqrtX;

        // 4. 计算距离场
        float dist = x * x + dy * dy - 1.8f;

        return dist * 0.5f;
    }

    /**
     * @brief AVX2向量化版本：心形SDF
     */
    inline __m256 sdHeartPrecise_avx(__m256 px, __m256 py) {
        // 常量
        const __m256 inv_scale_x = _mm256_set1_ps(1.0f / 0.87f);
        const __m256 scale_y = _mm256_set1_ps(-1.5f);
        const __m256 offset_y = _mm256_set1_ps(0.4f);
        const __m256 sqrt_coeff = _mm256_set1_ps(1.15f);
        const __m256 dist_offset = _mm256_set1_ps(-1.8f);
        const __m256 half = _mm256_set1_ps(0.5f);
        const __m256 zero = _mm256_setzero_ps();

        // 1. 水平系数
        __m256 x = _mm256_mul_ps(px, inv_scale_x);

        // 2. 纵向比例
        __m256 y = _mm256_add_ps(_mm256_mul_ps(py, scale_y), offset_y);

        // 3. 计算绝对值
        __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        __m256 absX = _mm256_and_ps(x, abs_mask);

        // 4. 平方根计算
        __m256 sqrtX = _mm256_sqrt_ps(absX);

        // 5. 计算dy
        __m256 dy = _mm256_sub_ps(y, _mm256_mul_ps(sqrtX, sqrt_coeff));

        // 6. 计算距离场
        __m256 x_sq = _mm256_mul_ps(x, x);
        __m256 dy_sq = _mm256_mul_ps(dy, dy);
        __m256 dist = _mm256_add_ps(_mm256_add_ps(x_sq, dy_sq), dist_offset);

        // 7. 乘以0.5
        return _mm256_mul_ps(dist, half);
    }

    /**
     * @brief SSE向量化版本：心形SDF
     */
    inline __m128 sdHeartPrecise_sse(__m128 px, __m128 py) {
        // 常量
        const __m128 inv_scale_x = _mm_set1_ps(1.0f / 0.87f);
        const __m128 scale_y = _mm_set1_ps(-1.5f);
        const __m128 offset_y = _mm_set1_ps(0.4f);
        const __m128 sqrt_coeff = _mm_set1_ps(1.15f);
        const __m128 dist_offset = _mm_set1_ps(-1.8f);
        const __m128 half = _mm_set1_ps(0.5f);
        const __m128 zero = _mm_setzero_ps();

        // 1. 水平系数
        __m128 x = _mm_mul_ps(px, inv_scale_x);

        // 2. 纵向比例
        __m128 y = _mm_add_ps(_mm_mul_ps(py, scale_y), offset_y);

        // 3. 计算绝对值
        const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
        __m128 absX = _mm_and_ps(x, abs_mask);

        // 4. 平方根计算
        __m128 sqrtX = _mm_sqrt_ps(absX);

        // 5. 计算dy
        __m128 dy = _mm_sub_ps(y, _mm_mul_ps(sqrtX, sqrt_coeff));

        // 6. 计算距离场
        __m128 x_sq = _mm_mul_ps(x, x);
        __m128 dy_sq = _mm_mul_ps(dy, dy);
        __m128 dist = _mm_add_ps(_mm_add_ps(x_sq, dy_sq), dist_offset);

        // 7. 乘以0.5
        return _mm_mul_ps(dist, half);
    }
}

/**
 * @brief SIMD优化的高质量心形渲染
 */
void drawHeart(
    pa2d::Canvas& canvas,
    float cx, float cy,
    float size,
    const pa2d::Color& color
) {
    auto& buffer = canvas.getBuffer();
    if (!buffer) return;

    // 扩大包围盒，确保心形的"耳朵"部分不会被裁剪
    float renderRange = size * 1.5f;
    int startX = std::max(0, (int)(cx - renderRange));
    int endX = std::min(buffer.width - 1, (int)(cx + renderRange));
    int startY = std::max(0, (int)(cy - renderRange));
    int endY = std::min(buffer.height - 1, (int)(cy + renderRange));

    if (startX > endX || startY > endY) return;

    // 预计算常量
    float invSize = 1.0f / size;
    float alphaScale = color.a / 255.0f;

    // 颜色分量归一化
    float srcR_norm = color.r * (1.0f / 255.0f);
    float srcG_norm = color.g * (1.0f / 255.0f);
    float srcB_norm = color.b * (1.0f / 255.0f);

    // AVX2常量
    const __m256 cx_avx = _mm256_set1_ps(cx);
    const __m256 cy_avx = _mm256_set1_ps(cy);
    const __m256 invSize_avx = _mm256_set1_ps(invSize);
    const __m256 edgeWidth_avx = _mm256_set1_ps(invSize * 1.2f);
    const __m256 alphaScale_avx = _mm256_set1_ps(alphaScale);
    const __m256 half_avx = _mm256_set1_ps(0.5f);
    const __m256 one_avx = _mm256_set1_ps(1.0f);
    const __m256 zero_avx = _mm256_setzero_ps();
    const __m256 eps_avx = _mm256_set1_ps(0.001f);
    const __m256 colorR_avx = _mm256_set1_ps(srcR_norm);
    const __m256 colorG_avx = _mm256_set1_ps(srcG_norm);
    const __m256 colorB_avx = _mm256_set1_ps(srcB_norm);

    // SSE常量
    const __m128 cx_sse = _mm_set1_ps(cx);
    const __m128 cy_sse = _mm_set1_ps(cy);
    const __m128 invSize_sse = _mm_set1_ps(invSize);
    const __m128 edgeWidth_sse = _mm_set1_ps(invSize * 1.2f);
    const __m128 alphaScale_sse = _mm_set1_ps(alphaScale);
    const __m128 half_sse = _mm_set1_ps(0.5f);
    const __m128 one_sse = _mm_set1_ps(1.0f);
    const __m128 zero_sse = _mm_setzero_ps();
    const __m128 eps_sse = _mm_set1_ps(0.001f);
    const __m128 colorR_sse = _mm_set1_ps(srcR_norm);
    const __m128 colorG_sse = _mm_set1_ps(srcG_norm);
    const __m128 colorB_sse = _mm_set1_ps(srcB_norm);

    for (int y = startY; y <= endY; ++y) {
        pa2d::Color* row = buffer.color + y * buffer.width;
        float py_norm = (y + 0.5f - cy) * invSize;
        __m256 py_norm_avx = _mm256_set1_ps(py_norm);
        __m128 py_norm_sse = _mm_set1_ps(py_norm);

        int x = startX;

        // === AVX2循环 (8像素并行) ===
        for (; x <= endX - 7; x += 8) {
            // 1. 生成x坐标
            __m256i x_indices = _mm256_setr_epi32(x, x + 1, x + 2, x + 3, x + 4, x + 5, x + 6, x + 7);
            __m256 x_coords = _mm256_add_ps(_mm256_cvtepi32_ps(x_indices),
                _mm256_set1_ps(0.5f));

            // 2. 转换到局部坐标
            __m256 px_raw = _mm256_sub_ps(x_coords, cx_avx);
            __m256 px_norm = _mm256_mul_ps(px_raw, invSize_avx);

            // 3. 计算SDF
            __m256 sdf = heart_sdf::sdHeartPrecise_avx(px_norm, py_norm_avx);

            // 4. 抗锯齿处理
            // factor = 1.0 - clamp((d / edgeWidth) + 0.5, 0.0, 1.0)
            __m256 d_div_edge = _mm256_div_ps(sdf, edgeWidth_avx);
            __m256 d_plus_half = _mm256_add_ps(d_div_edge, half_avx);

            // clamp到[0, 1]
            __m256 factor = _mm256_sub_ps(one_avx,
                _mm256_min_ps(one_avx,
                    _mm256_max_ps(zero_avx, d_plus_half)));

            // 5. 检查factor > 0.001
            __m256 mask = _mm256_cmp_ps(factor, eps_avx, _CMP_GT_OQ);

            if (!_mm256_testz_ps(mask, mask)) {
                // 6. 计算最终alpha
                __m256 final_alpha = _mm256_mul_ps(factor, alphaScale_avx);

                // 7. 读取目标像素
                __m256i dest = _mm256_loadu_si256((__m256i*) & row[x]);

                // 8. 混合像素
                __m256i blended = blend_pixels_avx(final_alpha, dest,
                    colorR_avx, colorG_avx, colorB_avx);

                // 9. 只更新mask为真的像素
                blended = _mm256_blendv_epi8(dest, blended, _mm256_castps_si256(mask));

                // 10. 写回结果
                _mm256_storeu_si256((__m256i*) & row[x], blended);
            }
        }

        // === SSE循环 (4像素并行) ===
        for (; x <= endX - 3; x += 4) {
            // 1. 生成x坐标
            __m128i x_indices = _mm_setr_epi32(x, x + 1, x + 2, x + 3);
            __m128 x_coords = _mm_add_ps(_mm_cvtepi32_ps(x_indices),
                _mm_set1_ps(0.5f));

            // 2. 转换到局部坐标
            __m128 px_raw = _mm_sub_ps(x_coords, cx_sse);
            __m128 px_norm = _mm_mul_ps(px_raw, invSize_sse);

            // 3. 计算SDF
            __m128 sdf = heart_sdf::sdHeartPrecise_sse(px_norm, py_norm_sse);

            // 4. 抗锯齿处理
            __m128 d_div_edge = _mm_div_ps(sdf, edgeWidth_sse);
            __m128 d_plus_half = _mm_add_ps(d_div_edge, half_sse);

            __m128 factor = _mm_sub_ps(one_sse,
                _mm_min_ps(one_sse,
                    _mm_max_ps(zero_sse, d_plus_half)));

            // 5. 检查factor > 0.001
            __m128 mask = _mm_cmpgt_ps(factor, eps_sse);

            if (_mm_movemask_ps(mask)) {
                // 6. 计算最终alpha
                __m128 final_alpha = _mm_mul_ps(factor, alphaScale_sse);

                // 7. 读取目标像素
                __m128i dest = _mm_loadu_si128((__m128i*) & row[x]);

                // 8. 混合像素
                __m128i blended = blend_pixels_sse(final_alpha, dest,
                    colorR_sse, colorG_sse, colorB_sse);

                // 9. 只更新mask为真的像素
                blended = _mm_blendv_epi8(dest, blended, _mm_castps_si128(mask));

                // 10. 写回结果
                _mm_storeu_si128((__m128i*) & row[x], blended);
            }
        }

        // === 标量循环 (剩余像素) ===
        for (; x <= endX; ++x) {
            float px = (x + 0.5f - cx) * invSize;
            float py = py_norm;

            // 获取 SDF 值
            float d = heart_sdf::sdHeartPrecise(px, py);

            // 抗锯齿处理
            float edgeWidth = invSize * 1.2f;
            float clampedValue = (d / edgeWidth) + 0.5f;
            if (clampedValue < 0.0f) {
                clampedValue = 0.0f;
            }
            else if (clampedValue > 1.0f) {
                clampedValue = 1.0f;
            }
            float factor = 1.0f - clampedValue;

            if (factor > 0.001f) {
                pa2d::Color src = color;
                src.a = static_cast<uint8_t>(factor * alphaScale * 255.0f);

                // 使用混合函数
                row[x] = Blend(src, row[x]);
            }
        }
    }
}