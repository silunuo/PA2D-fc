// blend_utils.h - 渲染工具
#pragma once
#include "pa2d.h"
#include <immintrin.h> // AVX2
#include <emmintrin.h> // SSE

const float GEOMETRY_PI = 3.14159265358979323846f;
const float GEOMETRY_EPSILON = 1e-6f;

namespace simd {
    // AVX 常量
    alignas(32) const __m256 ZERO_256 = _mm256_setzero_ps();
    alignas(32) const __m256 ONE_256 = _mm256_set1_ps(1.0f);
    alignas(32) const __m256 _255_256 = _mm256_set1_ps(255.0f);
    alignas(32) const __m256 _255_RECIP_256 = _mm256_set1_ps(1.0f / 255.0f);

    // 几何相关常量
    alignas(32) const __m256 HALF_PIXEL_256 = _mm256_set1_ps(0.5f);
    alignas(32) const __m256 ANTIALIAS_RANGE_256 = _mm256_set1_ps(1.0f);
    alignas(32) const __m256 ANTIALIAS_RANGE_INV_256 = _mm256_set1_ps(1.0f / 1.0f);

    // 整型常量
    alignas(32) const __m256i MASK_RED = _mm256_set1_epi32(0x00FF0000);
    alignas(32) const __m256i MASK_GREEN = _mm256_set1_epi32(0x0000FF00);
    alignas(32) const __m256i MASK_BLUE = _mm256_set1_epi32(0x000000FF);
    alignas(32) const __m256i MASK_ALPHA = _mm256_set1_epi32(0xFF000000);
    alignas(32) const __m256i SHIFT_8 = _mm256_set1_epi32(8);
    alignas(32) const __m256i SHIFT_16 = _mm256_set1_epi32(16);
    alignas(32) const __m256i SHIFT_24 = _mm256_set1_epi32(24);

    // SSE 常量
    alignas(16) const __m128 ZERO_128 = _mm_setzero_ps();
    alignas(16) const __m128 ONE_128 = _mm_set1_ps(1.0f);
    alignas(16) const __m128 _255_128 = _mm_set1_ps(255.0f);
    alignas(16) const __m128 _255_RECIP_128 = _mm_set1_ps(1.0f / 255.0f);

    // 几何相关常量
    alignas(16) const __m128 HALF_PIXEL_128 = _mm_set1_ps(0.5f);
    alignas(16) const __m128 ANTIALIAS_RANGE_128 = _mm_set1_ps(1.0f);
    alignas(16) const __m128 ANTIALIAS_RANGE_INV_128 = _mm_set1_ps(1.0f / 1.0f);

    // 整型常量
    alignas(16) const __m128i MASK_RED_128 = _mm_set1_epi32(0x00FF0000);
    alignas(16) const __m128i MASK_GREEN_128 = _mm_set1_epi32(0x0000FF00);
    alignas(16) const __m128i MASK_BLUE_128 = _mm_set1_epi32(0x000000FF);
    alignas(16) const __m128i SHIFT_8_128 = _mm_set1_epi32(8);
    alignas(16) const __m128i SHIFT_16_128 = _mm_set1_epi32(16);
    alignas(16) const __m128i SHIFT_24_128 = _mm_set1_epi32(24);

    // 用于计算浮点数绝对值的掩码（清除符号位）
    const __m256 abs_mask_256 = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m128 abs_mask_128 = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));

    // AVX2 atan2近似
    inline __m256 avx2_atan2(__m256 y, __m256 x) {
        // 使用快速近似计算atan2
        __m256 angle = _mm256_atan_ps(_mm256_div_ps(y, x));

        // 调整象限
        __m256 pi = _mm256_set1_ps(GEOMETRY_PI);
        __m256 pi_half = _mm256_set1_ps(GEOMETRY_PI * 0.5f);

        // 处理x < 0的情况
        __m256 mask_x_neg = _mm256_cmp_ps(x, ZERO_256, _CMP_LT_OQ);
        __m256 adjust_x_neg = _mm256_blendv_ps(ZERO_256, pi, mask_x_neg);
        angle = _mm256_add_ps(angle, adjust_x_neg);

        // 处理y < 0的情况
        __m256 mask_y_neg = _mm256_cmp_ps(y, ZERO_256, _CMP_LT_OQ);
        __m256 adjust_y_neg = _mm256_blendv_ps(ZERO_256, _mm256_set1_ps(2.0f * GEOMETRY_PI), mask_y_neg);
        angle = _mm256_add_ps(angle, _mm256_and_ps(mask_y_neg, adjust_y_neg));

        return angle;
    }

    // SSE atan2近似
    inline __m128 sse_atan2(__m128 y, __m128 x) {
        // 简化版本
        __m128 angle = _mm_atan_ps(_mm_div_ps(y, x));

        __m128 pi = _mm_set1_ps(GEOMETRY_PI);
        __m128 mask_x_neg = _mm_cmplt_ps(x, ZERO_128);
        __m128 adjust_x_neg = _mm_and_ps(mask_x_neg, pi);
        angle = _mm_add_ps(angle, adjust_x_neg);

        return angle;
    }

    // AVX2 cos近似
    inline __m256 avx2_cos(__m256 x) {
        // 使用多项式近似
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x4 = _mm256_mul_ps(x2, x2);

        __m256 c0 = _mm256_set1_ps(0.99940307f);
        __m256 c1 = _mm256_set1_ps(-0.49558072f);
        __m256 c2 = _mm256_set1_ps(0.03679168f);

        __m256 result = _mm256_add_ps(c0, _mm256_mul_ps(c1, x2));
        result = _mm256_add_ps(result, _mm256_mul_ps(c2, x4));

        return result;
    }

    // AVX2 sin近似
    inline __m256 avx2_sin(__m256 x) {
        // sin(x) ≈ cos(x - π/2)
        __m256 pi_half = _mm256_set1_ps(GEOMETRY_PI * 0.5f);
        __m256 x_shift = _mm256_sub_ps(x, pi_half);
        return avx2_cos(x_shift);
    }

    // SSE cos近似
    inline __m128 sse_cos(__m128 x) {
        __m128 x2 = _mm_mul_ps(x, x);
        __m128 x4 = _mm_mul_ps(x2, x2);

        __m128 c0 = _mm_set1_ps(0.99940307f);
        __m128 c1 = _mm_set1_ps(-0.49558072f);
        __m128 c2 = _mm_set1_ps(0.03679168f);

        __m128 result = _mm_add_ps(c0, _mm_mul_ps(c1, x2));
        result = _mm_add_ps(result, _mm_mul_ps(c2, x4));

        return result;
    }

    // SSE sin近似
    inline __m128 sse_sin(__m128 x) {
        __m128 pi_half = _mm_set1_ps(GEOMETRY_PI * 0.5f);
        __m128 x_shift = _mm_sub_ps(x, pi_half);
        return sse_cos(x_shift);
    }

    // SSE floor
    inline __m128 sse_floor(__m128 x) {
        // 使用转换实现近似的floor
        __m128i xi = _mm_cvttps_epi32(x);
        return _mm_cvtepi32_ps(xi);
    }
}


inline __m256i blend_pixels_avx(
    const __m256& combinedAlpha,
    const __m256i& dest,
    const __m256& srcR_01,
    const __m256& srcG_01,
    const __m256& srcB_01
) {
    using namespace simd; // 使用命名空间中的常量

    // 1. 解包目标
    __m256i b_i = _mm256_and_si256(dest, MASK_BLUE);
    __m256i g_i = _mm256_and_si256(_mm256_srli_epi32(dest, 8), MASK_BLUE);
    __m256i r_i = _mm256_and_si256(_mm256_srli_epi32(dest, 16), MASK_BLUE);
    __m256i a_i = _mm256_srli_epi32(dest, 24);

    __m256 fb = _mm256_mul_ps(_mm256_cvtepi32_ps(b_i), _255_RECIP_256);
    __m256 fg = _mm256_mul_ps(_mm256_cvtepi32_ps(g_i), _255_RECIP_256);
    __m256 fr = _mm256_mul_ps(_mm256_cvtepi32_ps(r_i), _255_RECIP_256);
    __m256 fa = _mm256_mul_ps(_mm256_cvtepi32_ps(a_i), _255_RECIP_256); // DstA

    // 2. 混合
    __m256 invCombinedAlpha = _mm256_sub_ps(ONE_256, combinedAlpha); // 1 - SrcA

    // OutA = SrcA + DstA * (1 - SrcA)
    __m256 blendedA = _mm256_add_ps(combinedAlpha, _mm256_mul_ps(fa, invCombinedAlpha));

    __m256 invOutAlpha = _mm256_div_ps(ONE_256, blendedA);
    __m256 zero_mask = _mm256_cmp_ps(blendedA, ZERO_256, _CMP_EQ_OQ);
    invOutAlpha = _mm256_andnot_ps(zero_mask, invOutAlpha); // 避免除以 0

    // OutRGB = (SrcRGB * SrcA + DstRGB * DstA * (1 - SrcA)) / OutA
    __m256 numR = _mm256_add_ps(_mm256_mul_ps(srcR_01, combinedAlpha), _mm256_mul_ps(fr, _mm256_mul_ps(fa, invCombinedAlpha)));
    __m256 numG = _mm256_add_ps(_mm256_mul_ps(srcG_01, combinedAlpha), _mm256_mul_ps(fg, _mm256_mul_ps(fa, invCombinedAlpha)));
    __m256 numB = _mm256_add_ps(_mm256_mul_ps(srcB_01, combinedAlpha), _mm256_mul_ps(fb, _mm256_mul_ps(fa, invCombinedAlpha)));

    __m256 blendedR = _mm256_mul_ps(numR, invOutAlpha);
    __m256 blendedG = _mm256_mul_ps(numG, invOutAlpha);
    __m256 blendedB = _mm256_mul_ps(numB, invOutAlpha);

    // 3. 转换回 0-255
    __m256i ri = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(blendedR, ZERO_256), ONE_256), _255_256));
    __m256i gi = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(blendedG, ZERO_256), ONE_256), _255_256));
    __m256i bi = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(blendedB, ZERO_256), ONE_256), _255_256));
    __m256i ai = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(blendedA, ZERO_256), ONE_256), _255_256));

    // 4. 打包
    return _mm256_or_si256(
        _mm256_or_si256(_mm256_slli_epi32(ai, 24), _mm256_slli_epi32(ri, 16)),
        _mm256_or_si256(_mm256_slli_epi32(gi, 8), bi)
    );
}

inline __m128i blend_pixels_sse(
    const __m128& combinedAlpha,
    const __m128i& dest,
    const __m128& srcR_01,
    const __m128& srcG_01,
    const __m128& srcB_01
) {
    using namespace simd; // 使用命名空间中的常量

    // 1. 解包
    __m128i b_i = _mm_and_si128(dest, MASK_BLUE_128);
    __m128i g_i = _mm_and_si128(_mm_srli_epi32(dest, 8), MASK_BLUE_128);
    __m128i r_i = _mm_and_si128(_mm_srli_epi32(dest, 16), MASK_BLUE_128);
    __m128i a_i = _mm_srli_epi32(dest, 24);

    __m128 fb = _mm_mul_ps(_mm_cvtepi32_ps(b_i), _255_RECIP_128);
    __m128 fg = _mm_mul_ps(_mm_cvtepi32_ps(g_i), _255_RECIP_128);
    __m128 fr = _mm_mul_ps(_mm_cvtepi32_ps(r_i), _255_RECIP_128);
    __m128 fa = _mm_mul_ps(_mm_cvtepi32_ps(a_i), _255_RECIP_128);

    // 2. 混合
    __m128 invCombinedAlpha = _mm_sub_ps(ONE_128, combinedAlpha);
    __m128 blendedA = _mm_add_ps(combinedAlpha, _mm_mul_ps(fa, invCombinedAlpha));
    __m128 invOutAlpha = _mm_div_ps(ONE_128, blendedA);
    invOutAlpha = _mm_andnot_ps(_mm_cmpeq_ps(blendedA, ZERO_128), invOutAlpha);

    __m128 numR = _mm_add_ps(_mm_mul_ps(srcR_01, combinedAlpha), _mm_mul_ps(fr, _mm_mul_ps(fa, invCombinedAlpha)));
    __m128 numG = _mm_add_ps(_mm_mul_ps(srcG_01, combinedAlpha), _mm_mul_ps(fg, _mm_mul_ps(fa, invCombinedAlpha)));
    __m128 numB = _mm_add_ps(_mm_mul_ps(srcB_01, combinedAlpha), _mm_mul_ps(fb, _mm_mul_ps(fa, invCombinedAlpha)));

    __m128 blendedR = _mm_mul_ps(numR, invOutAlpha);
    __m128 blendedG = _mm_mul_ps(numG, invOutAlpha);
    __m128 blendedB = _mm_mul_ps(numB, invOutAlpha);

    // 3. 转换
    __m128i ri = _mm_cvtps_epi32(_mm_mul_ps(_mm_min_ps(_mm_max_ps(blendedR, ZERO_128), ONE_128), _255_128));
    __m128i gi = _mm_cvtps_epi32(_mm_mul_ps(_mm_min_ps(_mm_max_ps(blendedG, ZERO_128), ONE_128), _255_128));
    __m128i bi = _mm_cvtps_epi32(_mm_mul_ps(_mm_min_ps(_mm_max_ps(blendedB, ZERO_128), ONE_128), _255_128));
    __m128i ai = _mm_cvtps_epi32(_mm_mul_ps(_mm_min_ps(_mm_max_ps(blendedA, ZERO_128), ONE_128), _255_128));

    // 4. 打包
    return _mm_or_si128(
        _mm_or_si128(_mm_slli_epi32(ai, 24), _mm_slli_epi32(ri, 16)),
        _mm_or_si128(_mm_slli_epi32(gi, 8), bi)
    );
}

inline pa2d::Color Blend(const pa2d::Color& src, const pa2d::Color& dst) {
    // 提取Alpha分量
    unsigned int srcAlpha = src.a;
    unsigned int dstAlpha = dst.a;

    // 计算混合后的总Alpha
    // 公式：out_alpha = src_alpha + dst_alpha * (1 - src_alpha/255)
    unsigned int outAlpha = srcAlpha + dstAlpha - (srcAlpha * dstAlpha) / 255;

    // 如果总Alpha为0，返回完全透明色
    if (outAlpha == 0) {
        return pa2d::Color(0, 0, 0, 0);
    }

    // 计算混合后的RGB分量
    // 公式：color = (src * src_alpha + dst * dst_alpha * (1 - src_alpha/255)) / out_alpha
    unsigned int invSrcAlpha = 255 - srcAlpha;
    unsigned int dstWeight = dstAlpha * invSrcAlpha / 255;

    unsigned int red = src.r * srcAlpha + dst.r * dstWeight;
    unsigned int green = src.g * srcAlpha + dst.g * dstWeight;
    unsigned int blue = src.b * srcAlpha + dst.b * dstWeight;

    // 归一化并返回结果
    return pa2d::Color(
        static_cast<unsigned char>(outAlpha),
        static_cast<unsigned char>(red / outAlpha),
        static_cast<unsigned char>(green / outAlpha),
        static_cast<unsigned char>(blue / outAlpha)
    );
}