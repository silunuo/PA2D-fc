// draw_particle.cpp - 粒子绘制函数
#include "../include/draw_particle.h"
#include <algorithm>
#include <cmath>
#include "../include/blend_utils.h"
#include <immintrin.h> // AVX2
#include <emmintrin.h> // SSE

/**
 * @brief 绘制渐变圆形粒子
 */
void drawParticle(
    pa2d::Canvas& src,
    float x, float y,
    float radius,
    const pa2d::Color& centerColor,
    const pa2d::Color& edgeColor
) {
    pa2d::Buffer& buffer = src.getBuffer();
    if (radius <= 0.0f) return;
    if (!buffer) return;

    // --- 1. 边界计算 ---
    const float outerRadius = radius + 2.0f;
    const int centerX = static_cast<int>(x);
    const int centerY = static_cast<int>(y);

    int minX = std::max(0, centerX - static_cast<int>(outerRadius));
    int maxX = std::min(buffer.width - 1, centerX + static_cast<int>(outerRadius));
    int minY = std::max(0, centerY - static_cast<int>(outerRadius));
    int maxY = std::min(buffer.height - 1, centerY + static_cast<int>(outerRadius));

    if (minX > maxX || minY > maxY) return;

    // --- 2. 颜色差值计算 ---
    const int dr = edgeColor.r - centerColor.r;
    const int dg = edgeColor.g - centerColor.g;
    const int db = edgeColor.b - centerColor.b;
    const int da = edgeColor.a - centerColor.a;

    const bool needColorLerp = (dr | dg | db | da);

    // --- 3. 优化常量 ---
    const int outerRadiusSq = static_cast<int>(outerRadius * outerRadius);
    const float radiusRecip = 1.0f / radius;

    // --- 4. SIMD常量 ---
    // AVX2常量
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 radiusVec = _mm256_set1_ps(radius);
    const __m256 radiusRecipVec = _mm256_set1_ps(radiusRecip);
    const __m256 _255 = _mm256_set1_ps(255.0f);
    const __m256 _255recip = _mm256_set1_ps(1.0f / 255.0f);

    // 颜色常量
    const __m256 centerR = _mm256_set1_ps(centerColor.r * (1.0f / 255.0f));
    const __m256 centerG = _mm256_set1_ps(centerColor.g * (1.0f / 255.0f));
    const __m256 centerB = _mm256_set1_ps(centerColor.b * (1.0f / 255.0f));
    const __m256 centerA = _mm256_set1_ps(centerColor.a * (1.0f / 255.0f));

    const __m256 edgeR = _mm256_set1_ps(edgeColor.r * (1.0f / 255.0f));
    const __m256 edgeG = _mm256_set1_ps(edgeColor.g * (1.0f / 255.0f));
    const __m256 edgeB = _mm256_set1_ps(edgeColor.b * (1.0f / 255.0f));
    const __m256 edgeA = _mm256_set1_ps(edgeColor.a * (1.0f / 255.0f));

    const __m256 colorDeltaR = _mm256_set1_ps(dr * (1.0f / 255.0f));
    const __m256 colorDeltaG = _mm256_set1_ps(dg * (1.0f / 255.0f));
    const __m256 colorDeltaB = _mm256_set1_ps(db * (1.0f / 255.0f));
    const __m256 colorDeltaA = _mm256_set1_ps(da * (1.0f / 255.0f));

    // --- 5. 主渲染循环 ---
    for (int py = minY; py <= maxY; ++py) {
        const int dy = py - centerY;
        const int dySq = dy * dy;
        pa2d::Color* row = &buffer.at(0, py);

        int px = minX;

        // === AVX2循环（8像素） ===
        for (; px <= maxX - 7; px += 8) {
            // 距离计算
            __m256i dxBase = _mm256_setr_epi32(px, px + 1, px + 2, px + 3, px + 4, px + 5, px + 6, px + 7);
            __m256 dx = _mm256_sub_ps(_mm256_cvtepi32_ps(dxBase), _mm256_set1_ps(x));

            __m256 dxSq = _mm256_mul_ps(dx, dx);
            __m256 distSq = _mm256_add_ps(dxSq, _mm256_set1_ps(dySq));
            __m256 dist = _mm256_sqrt_ps(distSq);

            // alpha计算
            __m256 alpha = _mm256_sub_ps(one, _mm256_sub_ps(dist, radiusVec));
            alpha = _mm256_max_ps(zero, _mm256_min_ps(one, alpha));

            __m256 mask = _mm256_cmp_ps(alpha, zero, _CMP_GT_OQ);
            if (_mm256_testz_ps(mask, mask)) continue;

            // 颜色插值
            __m256 srcR, srcG, srcB, srcA;
            if (needColorLerp) {
                __m256 t = _mm256_mul_ps(dist, radiusRecipVec);
                srcR = _mm256_add_ps(centerR, _mm256_mul_ps(colorDeltaR, t));
                srcG = _mm256_add_ps(centerG, _mm256_mul_ps(colorDeltaG, t));
                srcB = _mm256_add_ps(centerB, _mm256_mul_ps(colorDeltaB, t));
                srcA = _mm256_add_ps(centerA, _mm256_mul_ps(colorDeltaA, t));
            }
            else {
                srcR = centerR; srcG = centerG; srcB = centerB; srcA = centerA;
            }

            // 修正：预乘alpha并考虑源颜色的透明度通道
            __m256 combinedAlpha = _mm256_mul_ps(alpha, srcA);

            // 使用blend_utils.h中的混合函数
            __m256i dest = _mm256_loadu_si256((__m256i*) & row[px]);

            // 调用blend_pixels_avx进行正确的透明度混合
            __m256i rgba = blend_pixels_avx(combinedAlpha, dest, srcR, srcG, srcB);

            // 只更新alpha > 0的像素
            rgba = _mm256_blendv_epi8(dest, rgba, _mm256_castps_si256(mask));
            _mm256_storeu_si256((__m256i*) & row[px], rgba);
        }
        // === SSE循环（4像素） ===
        for (; px <= maxX - 3; px += 4) {
            // 距离计算
            __m128i dxBase = _mm_setr_epi32(px, px + 1, px + 2, px + 3);
            __m128 dx = _mm_sub_ps(_mm_cvtepi32_ps(dxBase), _mm_set1_ps(x));

            __m128 dxSq = _mm_mul_ps(dx, dx);
            __m128 dySqVec = _mm_set1_ps(static_cast<float>(dy * dy));
            __m128 distSq = _mm_add_ps(dxSq, dySqVec);
            __m128 dist = _mm_sqrt_ps(distSq);

            // alpha计算
            __m128 radiusVec = _mm_set1_ps(radius);
            __m128 alpha = _mm_sub_ps(_mm_set1_ps(1.0f),
                _mm_sub_ps(dist, radiusVec));
            alpha = _mm_max_ps(_mm_setzero_ps(),
                _mm_min_ps(_mm_set1_ps(1.0f), alpha));

            __m128 mask = _mm_cmpgt_ps(alpha, _mm_setzero_ps());
            if (!_mm_movemask_ps(mask)) continue;

            // 颜色插值
            __m128 srcR, srcG, srcB, srcA;
            if (needColorLerp) {
                // 计算插值因子t
                __m128 t = _mm_mul_ps(dist, _mm_set1_ps(radiusRecip));
                t = _mm_min_ps(_mm_set1_ps(1.0f), t);  // 限制在[0, 1]

                // 预计算的颜色差值（归一化到[0,1]）
                __m128 colorDeltaR = _mm_set1_ps(dr * (1.0f / 255.0f));
                __m128 colorDeltaG = _mm_set1_ps(dg * (1.0f / 255.0f));
                __m128 colorDeltaB = _mm_set1_ps(db * (1.0f / 255.0f));
                __m128 colorDeltaA = _mm_set1_ps(da * (1.0f / 255.0f));

                __m128 centerR = _mm_set1_ps(centerColor.r * (1.0f / 255.0f));
                __m128 centerG = _mm_set1_ps(centerColor.g * (1.0f / 255.0f));
                __m128 centerB = _mm_set1_ps(centerColor.b * (1.0f / 255.0f));
                __m128 centerA = _mm_set1_ps(centerColor.a * (1.0f / 255.0f));

                srcR = _mm_add_ps(centerR, _mm_mul_ps(colorDeltaR, t));
                srcG = _mm_add_ps(centerG, _mm_mul_ps(colorDeltaG, t));
                srcB = _mm_add_ps(centerB, _mm_mul_ps(colorDeltaB, t));
                srcA = _mm_add_ps(centerA, _mm_mul_ps(colorDeltaA, t));
            }
            else {
                srcR = _mm_set1_ps(centerColor.r * (1.0f / 255.0f));
                srcG = _mm_set1_ps(centerColor.g * (1.0f / 255.0f));
                srcB = _mm_set1_ps(centerColor.b * (1.0f / 255.0f));
                srcA = _mm_set1_ps(centerColor.a * (1.0f / 255.0f));
            }

            // 计算组合alpha：形状alpha * 颜色alpha
            __m128 combinedAlpha = _mm_mul_ps(alpha, srcA);

            // 加载目标像素
            __m128i dest = _mm_loadu_si128((__m128i*) & row[px]);

            // 使用SSE混合函数
            __m128i rgba = blend_pixels_sse(combinedAlpha, dest, srcR, srcG, srcB);

            // 只更新alpha > 0的像素
            rgba = _mm_blendv_epi8(dest, rgba, _mm_castps_si128(mask));
            _mm_storeu_si128((__m128i*) & row[px], rgba);
        }

        // === SSE循环（4像素） ===
        for (; px <= maxX - 3; px += 4) {
            __m128i dxBase = _mm_setr_epi32(px, px + 1, px + 2, px + 3);
            __m128 dx = _mm_sub_ps(_mm_cvtepi32_ps(dxBase), _mm_set1_ps(x));

            __m128 dxSq = _mm_mul_ps(dx, dx);
            __m128 distSq = _mm_add_ps(dxSq, _mm_set1_ps(dySq));
            __m128 dist = _mm_sqrt_ps(distSq);

            __m128 alpha = _mm_sub_ps(_mm_set1_ps(1.0f), _mm_sub_ps(dist, _mm_set1_ps(radius)));
            alpha = _mm_max_ps(_mm_setzero_ps(), _mm_min_ps(_mm_set1_ps(1.0f), alpha));

            __m128 mask = _mm_cmpgt_ps(alpha, _mm_setzero_ps());
            if (!_mm_movemask_ps(mask)) continue;

            // 颜色插值
            __m128 srcR, srcG, srcB, srcA;
            if (needColorLerp) {
                __m128 t = _mm_mul_ps(dist, _mm_set1_ps(radiusRecip));
                __m128 colorDeltaR_sse = _mm_set1_ps(dr * (1.0f / 255.0f));
                __m128 colorDeltaG_sse = _mm_set1_ps(dg * (1.0f / 255.0f));
                __m128 colorDeltaB_sse = _mm_set1_ps(db * (1.0f / 255.0f));
                __m128 colorDeltaA_sse = _mm_set1_ps(da * (1.0f / 255.0f));

                srcR = _mm_add_ps(_mm_set1_ps(centerColor.r * (1.0f / 255.0f)), _mm_mul_ps(colorDeltaR_sse, t));
                srcG = _mm_add_ps(_mm_set1_ps(centerColor.g * (1.0f / 255.0f)), _mm_mul_ps(colorDeltaG_sse, t));
                srcB = _mm_add_ps(_mm_set1_ps(centerColor.b * (1.0f / 255.0f)), _mm_mul_ps(colorDeltaB_sse, t));
                srcA = _mm_add_ps(_mm_set1_ps(centerColor.a * (1.0f / 255.0f)), _mm_mul_ps(colorDeltaA_sse, t));
            }
            else {
                srcR = _mm_set1_ps(centerColor.r * (1.0f / 255.0f));
                srcG = _mm_set1_ps(centerColor.g * (1.0f / 255.0f));
                srcB = _mm_set1_ps(centerColor.b * (1.0f / 255.0f));
                srcA = _mm_set1_ps(centerColor.a * (1.0f / 255.0f));
            }

            // 修正：计算combinedAlpha
            __m128 combinedAlpha = _mm_mul_ps(alpha, srcA);

            // 使用blend_pixels_sse
            __m128i dest = _mm_loadu_si128((__m128i*) & row[px]);
            __m128i rgba = blend_pixels_sse(combinedAlpha, dest, srcR, srcG, srcB);

            rgba = _mm_blendv_epi8(dest, rgba, _mm_castps_si128(mask));
            _mm_storeu_si128((__m128i*) & row[px], rgba);
        }

        // === 标量循环（剩余像素） ===
        for (; px <= maxX; ++px) {
            const int dx = px - centerX;
            const int distSq = dx * dx + dySq;

            if (distSq > outerRadiusSq) continue;

            const float dist = std::sqrt(static_cast<float>(distSq));
            float alpha = 1.0f - (dist - radius);
            if (alpha <= 0.0f) continue;
            if (alpha < 0.0f) {
                alpha = 0.0f;
            }
            else if (alpha > 1.0f) {
                alpha = 1.0f;
            }

            // 颜色插值
            pa2d::Color src;
            if (needColorLerp) {
                const float t = (dist * radiusRecip < 0.0f) ? 0.0f :
                    (dist * radiusRecip > 1.0f) ? 1.0f :
                    dist * radiusRecip;
                src.a = static_cast<uint8_t>(centerColor.a + da * t);
                src.r = static_cast<uint8_t>(centerColor.r + dr * t);
                src.g = static_cast<uint8_t>(centerColor.g + dg * t);
                src.b = static_cast<uint8_t>(centerColor.b + db * t);
            }
            else {
                src = centerColor;
            }

            // 修正：计算combinedAlpha并调用Blend函数
            const float combinedAlpha = alpha * (src.a / 255.0f);
            if (combinedAlpha <= 0.0f) continue;

            // 创建带alpha的源颜色
            pa2d::Color srcWithAlpha = src;
            srcWithAlpha.a = static_cast<uint8_t>(combinedAlpha * 255.0f);

            // 使用blend_utils.h中的Blend函数
            pa2d::Color& dest = row[px];
            dest = Blend(srcWithAlpha, dest);
        }
    }
}