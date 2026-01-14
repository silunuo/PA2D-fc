#include "blend_utils.h"

namespace pa2d {
    namespace utils {
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
        }
    }
}
