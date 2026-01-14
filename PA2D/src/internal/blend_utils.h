// blend_utils.h
#include"../include/color.h"
#include <immintrin.h>
#include <emmintrin.h>


namespace pa2d {
    namespace utils {
        const float GEOMETRY_PI = 3.14159265358979323846f;
        const float GEOMETRY_EPSILON = 1e-6f;

        namespace simd {
            // AVX 常量
            extern const __m256 ZERO_256;
            extern const __m256 ONE_256;
            extern const __m256 _255_256;
            extern const __m256 _255_RECIP_256;

            // 几何相关常量
            extern const __m256 HALF_PIXEL_256;  // 0.5f
            extern const __m256 ANTIALIAS_RANGE_256;  // 1.0f
            extern const __m256 ANTIALIAS_RANGE_INV_256;  // 1.0f / 1.0f = 1.0f

            extern const __m256i MASK_RED;
            extern const __m256i MASK_GREEN;
            extern const __m256i MASK_BLUE;
            extern const __m256i MASK_ALPHA;
            extern const __m256i SHIFT_8;
            extern const __m256i SHIFT_16;
            extern const __m256i SHIFT_24;

            // SSE 常量
            extern const __m128 ZERO_128;
            extern const __m128 ONE_128;
            extern const __m128 _255_128;
            extern const __m128 _255_RECIP_128;

            // 几何相关常量
            extern const __m128 HALF_PIXEL_128;  // 0.5f
            extern const __m128 ANTIALIAS_RANGE_128;  // 1.0f
            extern const __m128 ANTIALIAS_RANGE_INV_128;  // 1.0f / 1.0f = 1.0f

            extern const __m128i MASK_RED_128;
            extern const __m128i MASK_GREEN_128;
            extern const __m128i MASK_BLUE_128;
            extern const __m128i SHIFT_8_128;
            extern const __m128i SHIFT_16_128;
            extern const __m128i SHIFT_24_128;
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

        inline Color Blend(const Color& src, const Color& dst) {
            // 提取Alpha分量
            unsigned int srcAlpha = src.a;
            unsigned int dstAlpha = dst.a;

            // 计算混合后的总Alpha
            // 公式：out_alpha = src_alpha + dst_alpha * (1 - src_alpha/255)
            unsigned int outAlpha = srcAlpha + dstAlpha - (srcAlpha * dstAlpha) / 255;

            // 如果总Alpha为0，返回完全透明色
            if (outAlpha == 0) {
                return Color(0, 0, 0, 0);
            }

            // 计算混合后的RGB分量
            // 公式：color = (src * src_alpha + dst * dst_alpha * (1 - src_alpha/255)) / out_alpha
            unsigned int invSrcAlpha = 255 - srcAlpha;
            unsigned int dstWeight = dstAlpha * invSrcAlpha / 255;

            unsigned int red = src.r * srcAlpha + dst.r * dstWeight;
            unsigned int green = src.g * srcAlpha + dst.g * dstWeight;
            unsigned int blue = src.b * srcAlpha + dst.b * dstWeight;

            // 归一化并返回结果
            return Color(
                static_cast<unsigned char>(outAlpha),
                static_cast<unsigned char>(red / outAlpha),
                static_cast<unsigned char>(green / outAlpha),
                static_cast<unsigned char>(blue / outAlpha)
            );
        }
    }
}
