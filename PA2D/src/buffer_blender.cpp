#include "../include/buffer_blender.h"
#include "../include/buffer.h"
#include <algorithm>
#include <immintrin.h>
#include <emmintrin.h>

namespace pa2d {

    // ============================================================================
    // 简单拷贝混合（无透明度混合）
    // ============================================================================
    void blit(const Buffer& src, Buffer& dst, int dstX, int dstY) {
        if (!src.isValid() || !dst.isValid()) return;

        // 边界计算
        int startX = std::max(0, dstX);
        int startY = std::max(0, dstY);
        int endX = std::min(dst.width, dstX + src.width);
        int endY = std::min(dst.height, dstY + src.height);

        if (startX >= endX || startY >= endY) return;

        // 逐行拷贝
        for (int y = startY; y < endY; ++y) {
            int srcY = y - dstY;
            Color* destRow = dst.getRow(y) + startX;
            const Color* srcRow = src.getRow(srcY) + (startX - dstX);
            int copyWidth = endX - startX;

            // 使用memcpy进行批量拷贝
            std::memcpy(destRow, srcRow, copyWidth * sizeof(Color));
        }
    }

    // ============================================================================
    // Alpha透明度混合
    // ============================================================================
    void alphaBlend(const Buffer& src, Buffer& dst, int dstX, int dstY, int opacity) {
        if (!src.isValid() || !dst.isValid() || opacity == 0) return;

        // 边界计算
        int startX = std::max(0, dstX);
        int startY = std::max(0, dstY);
        int endX = std::min(dst.width, dstX + src.width);
        int endY = std::min(dst.height, dstY + src.height);

        if (startX >= endX || startY >= endY) return;

        // 预计算常量
        const __m256i global_opacity_vec = _mm256_set1_epi32(opacity);
        const __m128i global_opacity_vec_sse = _mm_set1_epi32(opacity);

        for (int y = startY; y < endY; ++y) {
            int srcY = y - dstY;
            Color* destRow = dst.getRow(y) + startX;
            const Color* srcRow = src.getRow(srcY) + (startX - dstX);
            int rowWidth = endX - startX;

            size_t x = 0;

            // AVX2优化（8像素并行）
            const size_t simd_count8 = rowWidth & ~7;
            if (simd_count8 > 0) {
                const __m256i mask_ff = _mm256_set1_epi32(0xFF);
                const __m256i const_255 = _mm256_set1_epi32(255);
                const __m256i zero = _mm256_setzero_si256();

                for (; x < simd_count8; x += 8) {
                    // 预取下一缓存行
                    _mm_prefetch((const char*)(srcRow + x + 16), _MM_HINT_T0);
                    _mm_prefetch((const char*)(destRow + x + 16), _MM_HINT_T0);

                    __m256i src_vec = _mm256_loadu_si256((const __m256i*)(srcRow + x));
                    __m256i dst_vec = _mm256_loadu_si256((const __m256i*)(destRow + x));

                    // 解包RGBA通道
                    __m256i src_a = _mm256_srli_epi32(src_vec, 24);
                    __m256i src_r = _mm256_and_si256(_mm256_srli_epi32(src_vec, 16), mask_ff);
                    __m256i src_g = _mm256_and_si256(_mm256_srli_epi32(src_vec, 8), mask_ff);
                    __m256i src_b = _mm256_and_si256(src_vec, mask_ff);

                    __m256i dst_a = _mm256_srli_epi32(dst_vec, 24);
                    __m256i dst_r = _mm256_and_si256(_mm256_srli_epi32(dst_vec, 16), mask_ff);
                    __m256i dst_g = _mm256_and_si256(_mm256_srli_epi32(dst_vec, 8), mask_ff);
                    __m256i dst_b = _mm256_and_si256(dst_vec, mask_ff);

                    // 计算有效alpha并快速检查
                    __m256i effective_a = _mm256_srli_epi32(_mm256_mullo_epi32(src_a, global_opacity_vec), 8);
                    if (_mm256_testz_si256(effective_a, effective_a)) {
                        continue; // 所有像素都透明，跳过
                    }

                    // 混合计算
                    __m256i inv_alpha = _mm256_sub_epi32(const_255, effective_a);

                    __m256i blended_r = _mm256_srli_epi32(_mm256_add_epi32(
                        _mm256_mullo_epi32(src_r, effective_a),
                        _mm256_mullo_epi32(dst_r, inv_alpha)), 8);

                    __m256i blended_g = _mm256_srli_epi32(_mm256_add_epi32(
                        _mm256_mullo_epi32(src_g, effective_a),
                        _mm256_mullo_epi32(dst_g, inv_alpha)), 8);

                    __m256i blended_b = _mm256_srli_epi32(_mm256_add_epi32(
                        _mm256_mullo_epi32(src_b, effective_a),
                        _mm256_mullo_epi32(dst_b, inv_alpha)), 8);

                    __m256i blended_a = _mm256_add_epi32(effective_a,
                        _mm256_srli_epi32(_mm256_mullo_epi32(dst_a, inv_alpha), 8));

                    // 饱和运算确保范围
                    blended_r = _mm256_min_epu8(blended_r, const_255);
                    blended_g = _mm256_min_epu8(blended_g, const_255);
                    blended_b = _mm256_min_epu8(blended_b, const_255);
                    blended_a = _mm256_min_epu8(blended_a, const_255);

                    // 打包结果
                    __m256i result = _mm256_or_si256(
                        _mm256_slli_epi32(blended_r, 16),
                        _mm256_slli_epi32(blended_g, 8));
                    result = _mm256_or_si256(result, blended_b);
                    result = _mm256_or_si256(result, _mm256_slli_epi32(blended_a, 24));

                    _mm256_storeu_si256((__m256i*)(destRow + x), result);
                }
            }

            // SSE4优化（4像素并行）
            const size_t simd_count4 = rowWidth & ~3;
            if (simd_count4 > x) {
                const __m128i mask_ff = _mm_set1_epi32(0xFF);
                const __m128i const_255 = _mm_set1_epi32(255);
                const __m128i zero = _mm_setzero_si128();

                for (; x < simd_count4; x += 4) {
                    __m128i src_vec = _mm_loadu_si128((const __m128i*)(srcRow + x));
                    __m128i dst_vec = _mm_loadu_si128((const __m128i*)(destRow + x));

                    // 解包RGBA通道
                    __m128i src_a = _mm_srli_epi32(src_vec, 24);
                    __m128i src_r = _mm_and_si128(_mm_srli_epi32(src_vec, 16), mask_ff);
                    __m128i src_g = _mm_and_si128(_mm_srli_epi32(src_vec, 8), mask_ff);
                    __m128i src_b = _mm_and_si128(src_vec, mask_ff);

                    __m128i dst_a = _mm_srli_epi32(dst_vec, 24);
                    __m128i dst_r = _mm_and_si128(_mm_srli_epi32(dst_vec, 16), mask_ff);
                    __m128i dst_g = _mm_and_si128(_mm_srli_epi32(dst_vec, 8), mask_ff);
                    __m128i dst_b = _mm_and_si128(dst_vec, mask_ff);

                    // 计算有效alpha
                    __m128i effective_a = _mm_srli_epi32(_mm_mullo_epi32(src_a, global_opacity_vec_sse), 8);
                    if (_mm_test_all_zeros(effective_a, effective_a)) {
                        continue;
                    }

                    // 混合计算
                    __m128i inv_alpha = _mm_sub_epi32(const_255, effective_a);

                    __m128i blended_r = _mm_srli_epi32(_mm_add_epi32(
                        _mm_mullo_epi32(src_r, effective_a),
                        _mm_mullo_epi32(dst_r, inv_alpha)), 8);

                    __m128i blended_g = _mm_srli_epi32(_mm_add_epi32(
                        _mm_mullo_epi32(src_g, effective_a),
                        _mm_mullo_epi32(dst_g, inv_alpha)), 8);

                    __m128i blended_b = _mm_srli_epi32(_mm_add_epi32(
                        _mm_mullo_epi32(src_b, effective_a),
                        _mm_mullo_epi32(dst_b, inv_alpha)), 8);

                    __m128i blended_a = _mm_add_epi32(effective_a,
                        _mm_srli_epi32(_mm_mullo_epi32(dst_a, inv_alpha), 8));

                    // 饱和运算
                    blended_r = _mm_min_epu8(blended_r, const_255);
                    blended_g = _mm_min_epu8(blended_g, const_255);
                    blended_b = _mm_min_epu8(blended_b, const_255);
                    blended_a = _mm_min_epu8(blended_a, const_255);

                    // 打包
                    __m128i result = _mm_or_si128(
                        _mm_slli_epi32(blended_r, 16),
                        _mm_slli_epi32(blended_g, 8));
                    result = _mm_or_si128(result, blended_b);
                    result = _mm_or_si128(result, _mm_slli_epi32(blended_a, 24));

                    _mm_storeu_si128((__m128i*)(destRow + x), result);
                }
            }

            // 标量处理（剩余像素）
            for (; x < rowWidth; ++x) {
                const Color& src_color = srcRow[x];
                Color& dst_color = destRow[x];

                // 快速路径检查
                if (src_color.a == 0 || opacity == 0) continue;

                uint32_t effective_alpha = (static_cast<uint32_t>(src_color.a) * opacity) >> 8;
                if (effective_alpha == 0) continue;

                uint32_t inv_alpha = 255 - effective_alpha;

                // 混合计算
                uint32_t r = (static_cast<uint32_t>(src_color.r) * effective_alpha +
                    static_cast<uint32_t>(dst_color.r) * inv_alpha) >> 8;
                uint32_t g = (static_cast<uint32_t>(src_color.g) * effective_alpha +
                    static_cast<uint32_t>(dst_color.g) * inv_alpha) >> 8;
                uint32_t b = (static_cast<uint32_t>(src_color.b) * effective_alpha +
                    static_cast<uint32_t>(dst_color.b) * inv_alpha) >> 8;
                uint32_t a = effective_alpha +
                    ((static_cast<uint32_t>(dst_color.a) * inv_alpha) >> 8);

                // 确保范围
                dst_color = Color(
                    static_cast<uint8_t>(a > 255 ? 255 : a),
                    static_cast<uint8_t>(r > 255 ? 255 : r),
                    static_cast<uint8_t>(g > 255 ? 255 : g),
                    static_cast<uint8_t>(b > 255 ? 255 : b)
                );
            }
        }
    }

    // ============================================================================
    // 加法混合（变亮效果）
    // ============================================================================
    void addBlend(const Buffer& src, Buffer& dst, int dstX, int dstY, int opacity) {
        if (!src.isValid() || !dst.isValid() || opacity == 0) return;

        // 边界计算
        int startX = std::max(0, dstX);
        int startY = std::max(0, dstY);
        int endX = std::min(dst.width, dstX + src.width);
        int endY = std::min(dst.height, dstY + src.height);

        if (startX >= endX || startY >= endY) return;

        const __m256i global_opacity_vec = _mm256_set1_epi32(opacity);
        const __m128i global_opacity_vec_sse = _mm_set1_epi32(opacity);

        for (int y = startY; y < endY; ++y) {
            int srcY = y - dstY;
            Color* destRow = dst.getRow(y) + startX;
            const Color* srcRow = src.getRow(srcY) + (startX - dstX);
            int rowWidth = endX - startX;

            size_t x = 0;

            // AVX2优化
            const size_t simd_count8 = rowWidth & ~7;
            if (simd_count8 > 0) {
                const __m256i mask_ff = _mm256_set1_epi32(0xFF);
                const __m256i const_255 = _mm256_set1_epi32(255);
                const __m256i zero = _mm256_setzero_si256();

                for (; x < simd_count8; x += 8) {
                    // 预取下一缓存行
                    _mm_prefetch((const char*)(srcRow + x + 16), _MM_HINT_T0);
                    _mm_prefetch((const char*)(destRow + x + 16), _MM_HINT_T0);

                    __m256i src_vec = _mm256_loadu_si256((const __m256i*)(srcRow + x));
                    __m256i dst_vec = _mm256_loadu_si256((const __m256i*)(destRow + x));

                    // 解包RGBA通道
                    __m256i src_a = _mm256_srli_epi32(src_vec, 24);
                    __m256i src_r = _mm256_and_si256(_mm256_srli_epi32(src_vec, 16), mask_ff);
                    __m256i src_g = _mm256_and_si256(_mm256_srli_epi32(src_vec, 8), mask_ff);
                    __m256i src_b = _mm256_and_si256(src_vec, mask_ff);

                    __m256i dst_a = _mm256_srli_epi32(dst_vec, 24);
                    __m256i dst_r = _mm256_and_si256(_mm256_srli_epi32(dst_vec, 16), mask_ff);
                    __m256i dst_g = _mm256_and_si256(_mm256_srli_epi32(dst_vec, 8), mask_ff);
                    __m256i dst_b = _mm256_and_si256(dst_vec, mask_ff);

                    // 饱和加法
                    __m256i r_sum = _mm256_add_epi32(src_r, dst_r);
                    __m256i g_sum = _mm256_add_epi32(src_g, dst_g);
                    __m256i b_sum = _mm256_add_epi32(src_b, dst_b);

                    // 创建比较掩码
                    __m256i r_mask = _mm256_cmpgt_epi32(r_sum, const_255);
                    __m256i g_mask = _mm256_cmpgt_epi32(g_sum, const_255);
                    __m256i b_mask = _mm256_cmpgt_epi32(b_sum, const_255);

                    // 使用混合指令选择结果
                    __m256i r_blend = _mm256_blendv_epi8(r_sum, const_255, r_mask);
                    __m256i g_blend = _mm256_blendv_epi8(g_sum, const_255, g_mask);
                    __m256i b_blend = _mm256_blendv_epi8(b_sum, const_255, b_mask);

                    // 计算有效alpha并快速检查
                    __m256i effective_a = _mm256_srli_epi32(
                        _mm256_mullo_epi32(src_a, global_opacity_vec), 8);
                    if (_mm256_testz_si256(effective_a, effective_a)) {
                        continue;
                    }

                    // 混合计算
                    __m256i inv_effective_a = _mm256_sub_epi32(const_255, effective_a);

                    __m256i r = _mm256_srli_epi32(_mm256_add_epi32(
                        _mm256_mullo_epi32(r_blend, effective_a),
                        _mm256_mullo_epi32(dst_r, inv_effective_a)), 8);

                    __m256i g = _mm256_srli_epi32(_mm256_add_epi32(
                        _mm256_mullo_epi32(g_blend, effective_a),
                        _mm256_mullo_epi32(dst_g, inv_effective_a)), 8);

                    __m256i b = _mm256_srli_epi32(_mm256_add_epi32(
                        _mm256_mullo_epi32(b_blend, effective_a),
                        _mm256_mullo_epi32(dst_b, inv_effective_a)), 8);

                    __m256i a = _mm256_add_epi32(effective_a,
                        _mm256_srli_epi32(_mm256_mullo_epi32(dst_a, inv_effective_a), 8));

                    // 确保范围
                    r = _mm256_min_epu8(_mm256_max_epu8(r, zero), const_255);
                    g = _mm256_min_epu8(_mm256_max_epu8(g, zero), const_255);
                    b = _mm256_min_epu8(_mm256_max_epu8(b, zero), const_255);
                    a = _mm256_min_epu8(_mm256_max_epu8(a, zero), const_255);

                    // 打包结果
                    __m256i result = _mm256_or_si256(
                        _mm256_slli_epi32(r, 16),
                        _mm256_slli_epi32(g, 8));
                    result = _mm256_or_si256(result, b);
                    result = _mm256_or_si256(result, _mm256_slli_epi32(a, 24));

                    _mm256_storeu_si256((__m256i*)(destRow + x), result);
                }
            }

            // SSE4优化
            const size_t simd_count4 = rowWidth & ~3;
            if (simd_count4 > x) {
                const __m128i mask_ff = _mm_set1_epi32(0xFF);
                const __m128i const_255 = _mm_set1_epi32(255);
                const __m128i zero = _mm_setzero_si128();

                for (; x < simd_count4; x += 4) {
                    __m128i src_vec = _mm_loadu_si128((const __m128i*)(srcRow + x));
                    __m128i dst_vec = _mm_loadu_si128((const __m128i*)(destRow + x));

                    // 解包
                    __m128i src_a = _mm_srli_epi32(src_vec, 24);
                    __m128i src_r = _mm_and_si128(_mm_srli_epi32(src_vec, 16), mask_ff);
                    __m128i src_g = _mm_and_si128(_mm_srli_epi32(src_vec, 8), mask_ff);
                    __m128i src_b = _mm_and_si128(src_vec, mask_ff);

                    __m128i dst_a = _mm_srli_epi32(dst_vec, 24);
                    __m128i dst_r = _mm_and_si128(_mm_srli_epi32(dst_vec, 16), mask_ff);
                    __m128i dst_g = _mm_and_si128(_mm_srli_epi32(dst_vec, 8), mask_ff);
                    __m128i dst_b = _mm_and_si128(dst_vec, mask_ff);

                    // 饱和加法
                    __m128i r_sum = _mm_add_epi32(src_r, dst_r);
                    __m128i g_sum = _mm_add_epi32(src_g, dst_g);
                    __m128i b_sum = _mm_add_epi32(src_b, dst_b);

                    __m128i r_mask = _mm_cmpgt_epi32(r_sum, const_255);
                    __m128i g_mask = _mm_cmpgt_epi32(g_sum, const_255);
                    __m128i b_mask = _mm_cmpgt_epi32(b_sum, const_255);

                    __m128i r_blend = _mm_blendv_epi8(r_sum, const_255, r_mask);
                    __m128i g_blend = _mm_blendv_epi8(g_sum, const_255, g_mask);
                    __m128i b_blend = _mm_blendv_epi8(b_sum, const_255, b_mask);

                    // 计算有效alpha
                    __m128i effective_a = _mm_srli_epi32(
                        _mm_mullo_epi32(src_a, global_opacity_vec_sse), 8);
                    if (_mm_test_all_zeros(effective_a, effective_a)) {
                        continue;
                    }

                    // 混合计算
                    __m128i inv_effective_a = _mm_sub_epi32(const_255, effective_a);

                    __m128i r = _mm_srli_epi32(_mm_add_epi32(
                        _mm_mullo_epi32(r_blend, effective_a),
                        _mm_mullo_epi32(dst_r, inv_effective_a)), 8);

                    __m128i g = _mm_srli_epi32(_mm_add_epi32(
                        _mm_mullo_epi32(g_blend, effective_a),
                        _mm_mullo_epi32(dst_g, inv_effective_a)), 8);

                    __m128i b = _mm_srli_epi32(_mm_add_epi32(
                        _mm_mullo_epi32(b_blend, effective_a),
                        _mm_mullo_epi32(dst_b, inv_effective_a)), 8);

                    __m128i a = _mm_add_epi32(effective_a,
                        _mm_srli_epi32(_mm_mullo_epi32(dst_a, inv_effective_a), 8));

                    // 确保范围
                    r = _mm_min_epu8(_mm_max_epu8(r, zero), const_255);
                    g = _mm_min_epu8(_mm_max_epu8(g, zero), const_255);
                    b = _mm_min_epu8(_mm_max_epu8(b, zero), const_255);
                    a = _mm_min_epu8(_mm_max_epu8(a, zero), const_255);

                    // 打包
                    __m128i result = _mm_or_si128(
                        _mm_slli_epi32(r, 16),
                        _mm_slli_epi32(g, 8));
                    result = _mm_or_si128(result, b);
                    result = _mm_or_si128(result, _mm_slli_epi32(a, 24));

                    _mm_storeu_si128((__m128i*)(destRow + x), result);
                }
            }

            // 标量处理
            for (; x < rowWidth; ++x) {
                const Color& src_color = srcRow[x];
                Color& dst_color = destRow[x];

                // 快速路径检查
                if (src_color.a == 0 || opacity == 0) continue;

                uint32_t effective_alpha = (static_cast<uint32_t>(src_color.a) * opacity) >> 8;
                if (effective_alpha == 0) continue;

                // 饱和加法
                uint32_t r_sum = static_cast<uint32_t>(src_color.r) + dst_color.r;
                uint32_t g_sum = static_cast<uint32_t>(src_color.g) + dst_color.g;
                uint32_t b_sum = static_cast<uint32_t>(src_color.b) + dst_color.b;

                uint32_t r_blend = r_sum > 255 ? 255 : r_sum;
                uint32_t g_blend = g_sum > 255 ? 255 : g_sum;
                uint32_t b_blend = b_sum > 255 ? 255 : b_sum;

                // 混合计算
                uint32_t inv_alpha = 255 - effective_alpha;

                uint32_t r = (r_blend * effective_alpha +
                    static_cast<uint32_t>(dst_color.r) * inv_alpha) >> 8;
                uint32_t g = (g_blend * effective_alpha +
                    static_cast<uint32_t>(dst_color.g) * inv_alpha) >> 8;
                uint32_t b = (b_blend * effective_alpha +
                    static_cast<uint32_t>(dst_color.b) * inv_alpha) >> 8;
                uint32_t a = effective_alpha +
                    ((static_cast<uint32_t>(dst_color.a) * inv_alpha) >> 8);

                // 确保范围
                dst_color = Color(
                    static_cast<uint8_t>(a > 255 ? 255 : a),
                    static_cast<uint8_t>(r > 255 ? 255 : r),
                    static_cast<uint8_t>(g > 255 ? 255 : g),
                    static_cast<uint8_t>(b > 255 ? 255 : b)
                );
            }
        }
    }

    // ============================================================================
    // 正片叠底混合（变暗效果）
    // ============================================================================
    void multiplyBlend(const Buffer& src, Buffer& dst, int dstX, int dstY, int opacity) {
        if (!src.isValid() || !dst.isValid() || opacity == 0) return;

        // 边界计算
        int startX = std::max(0, dstX);
        int startY = std::max(0, dstY);
        int endX = std::min(dst.width, dstX + src.width);
        int endY = std::min(dst.height, dstY + src.height);

        if (startX >= endX || startY >= endY) return;

        const __m256i global_opacity_vec = _mm256_set1_epi32(opacity);
        const __m128i global_opacity_vec_sse = _mm_set1_epi32(opacity);

        for (int y = startY; y < endY; ++y) {
            int srcY = y - dstY;
            Color* destRow = dst.getRow(y) + startX;
            const Color* srcRow = src.getRow(srcY) + (startX - dstX);
            int rowWidth = endX - startX;

            size_t x = 0;

            // AVX2优化
            const size_t simd_count8 = rowWidth & ~7;
            if (simd_count8 > 0) {
                const __m256i mask_ff = _mm256_set1_epi32(0xFF);
                const __m256i const_255 = _mm256_set1_epi32(255);
                const __m256i zero = _mm256_setzero_si256();

                for (; x < simd_count8; x += 8) {
                    // 预取下一缓存行
                    _mm_prefetch((const char*)(srcRow + x + 16), _MM_HINT_T0);
                    _mm_prefetch((const char*)(destRow + x + 16), _MM_HINT_T0);

                    __m256i src_vec = _mm256_loadu_si256((const __m256i*)(srcRow + x));
                    __m256i dst_vec = _mm256_loadu_si256((const __m256i*)(destRow + x));

                    // 解包RGBA通道
                    __m256i src_a = _mm256_srli_epi32(src_vec, 24);
                    __m256i src_r = _mm256_and_si256(_mm256_srli_epi32(src_vec, 16), mask_ff);
                    __m256i src_g = _mm256_and_si256(_mm256_srli_epi32(src_vec, 8), mask_ff);
                    __m256i src_b = _mm256_and_si256(src_vec, mask_ff);

                    __m256i dst_a = _mm256_srli_epi32(dst_vec, 24);
                    __m256i dst_r = _mm256_and_si256(_mm256_srli_epi32(dst_vec, 16), mask_ff);
                    __m256i dst_g = _mm256_and_si256(_mm256_srli_epi32(dst_vec, 8), mask_ff);
                    __m256i dst_b = _mm256_and_si256(dst_vec, mask_ff);

                    // 计算有效alpha并快速检查
                    __m256i effective_a = _mm256_srli_epi32(
                        _mm256_mullo_epi32(src_a, global_opacity_vec), 8);
                    if (_mm256_testz_si256(effective_a, effective_a)) {
                        continue;
                    }

                    // 合并计算：正片叠底 + alpha混合
                    __m256i inv_effective_a = _mm256_sub_epi32(const_255, effective_a);

                    __m256i r_num = _mm256_add_epi32(
                        _mm256_mullo_epi32(_mm256_mullo_epi32(src_r, dst_r), effective_a),
                        _mm256_mullo_epi32(_mm256_mullo_epi32(dst_r, const_255), inv_effective_a)
                    );

                    __m256i g_num = _mm256_add_epi32(
                        _mm256_mullo_epi32(_mm256_mullo_epi32(src_g, dst_g), effective_a),
                        _mm256_mullo_epi32(_mm256_mullo_epi32(dst_g, const_255), inv_effective_a)
                    );

                    __m256i b_num = _mm256_add_epi32(
                        _mm256_mullo_epi32(_mm256_mullo_epi32(src_b, dst_b), effective_a),
                        _mm256_mullo_epi32(_mm256_mullo_epi32(dst_b, const_255), inv_effective_a)
                    );

                    // 右移16位完成除以65536
                    __m256i r = _mm256_srli_epi32(r_num, 16);
                    __m256i g = _mm256_srli_epi32(g_num, 16);
                    __m256i b = _mm256_srli_epi32(b_num, 16);

                    // Alpha通道混合
                    __m256i a = _mm256_add_epi32(effective_a,
                        _mm256_srli_epi32(_mm256_mullo_epi32(dst_a, inv_effective_a), 8));

                    // 饱和运算
                    r = _mm256_min_epu8(_mm256_max_epu8(r, zero), const_255);
                    g = _mm256_min_epu8(_mm256_max_epu8(g, zero), const_255);
                    b = _mm256_min_epu8(_mm256_max_epu8(b, zero), const_255);
                    a = _mm256_min_epu8(_mm256_max_epu8(a, zero), const_255);

                    // 打包结果
                    __m256i result = _mm256_or_si256(
                        _mm256_slli_epi32(r, 16),
                        _mm256_slli_epi32(g, 8));
                    result = _mm256_or_si256(result, b);
                    result = _mm256_or_si256(result, _mm256_slli_epi32(a, 24));

                    _mm256_storeu_si256((__m256i*)(destRow + x), result);
                }
            }

            // SSE4优化
            const size_t simd_count4 = rowWidth & ~3;
            if (simd_count4 > x) {
                const __m128i mask_ff = _mm_set1_epi32(0xFF);
                const __m128i const_255 = _mm_set1_epi32(255);
                const __m128i zero = _mm_setzero_si128();

                for (; x < simd_count4; x += 4) {
                    __m128i src_vec = _mm_loadu_si128((const __m128i*)(srcRow + x));
                    __m128i dst_vec = _mm_loadu_si128((const __m128i*)(destRow + x));

                    // 解包
                    __m128i src_a = _mm_srli_epi32(src_vec, 24);
                    __m128i src_r = _mm_and_si128(_mm_srli_epi32(src_vec, 16), mask_ff);
                    __m128i src_g = _mm_and_si128(_mm_srli_epi32(src_vec, 8), mask_ff);
                    __m128i src_b = _mm_and_si128(src_vec, mask_ff);

                    __m128i dst_a = _mm_srli_epi32(dst_vec, 24);
                    __m128i dst_r = _mm_and_si128(_mm_srli_epi32(dst_vec, 16), mask_ff);
                    __m128i dst_g = _mm_and_si128(_mm_srli_epi32(dst_vec, 8), mask_ff);
                    __m128i dst_b = _mm_and_si128(dst_vec, mask_ff);

                    // 计算有效alpha
                    __m128i effective_a = _mm_srli_epi32(
                        _mm_mullo_epi32(src_a, global_opacity_vec_sse), 8);
                    if (_mm_test_all_zeros(effective_a, effective_a)) {
                        continue;
                    }

                    // 合并计算
                    __m128i inv_effective_a = _mm_sub_epi32(const_255, effective_a);

                    __m128i r_num = _mm_add_epi32(
                        _mm_mullo_epi32(_mm_mullo_epi32(src_r, dst_r), effective_a),
                        _mm_mullo_epi32(_mm_mullo_epi32(dst_r, const_255), inv_effective_a)
                    );

                    __m128i g_num = _mm_add_epi32(
                        _mm_mullo_epi32(_mm_mullo_epi32(src_g, dst_g), effective_a),
                        _mm_mullo_epi32(_mm_mullo_epi32(dst_g, const_255), inv_effective_a)
                    );

                    __m128i b_num = _mm_add_epi32(
                        _mm_mullo_epi32(_mm_mullo_epi32(src_b, dst_b), effective_a),
                        _mm_mullo_epi32(_mm_mullo_epi32(dst_b, const_255), inv_effective_a)
                    );

                    __m128i r = _mm_srli_epi32(r_num, 16);
                    __m128i g = _mm_srli_epi32(g_num, 16);
                    __m128i b = _mm_srli_epi32(b_num, 16);
                    __m128i a = _mm_add_epi32(effective_a,
                        _mm_srli_epi32(_mm_mullo_epi32(dst_a, inv_effective_a), 8));

                    // 饱和运算
                    r = _mm_min_epu8(_mm_max_epu8(r, zero), const_255);
                    g = _mm_min_epu8(_mm_max_epu8(g, zero), const_255);
                    b = _mm_min_epu8(_mm_max_epu8(b, zero), const_255);
                    a = _mm_min_epu8(_mm_max_epu8(a, zero), const_255);

                    // 打包
                    __m128i result = _mm_or_si128(
                        _mm_slli_epi32(r, 16),
                        _mm_slli_epi32(g, 8));
                    result = _mm_or_si128(result, b);
                    result = _mm_or_si128(result, _mm_slli_epi32(a, 24));

                    _mm_storeu_si128((__m128i*)(destRow + x), result);
                }
            }

            // 标量处理
            for (; x < rowWidth; ++x) {
                const Color& src_color = srcRow[x];
                Color& dst_color = destRow[x];

                // 快速路径检查
                if (src_color.a == 0 || opacity == 0) continue;

                uint32_t effective_alpha = (static_cast<uint32_t>(src_color.a) * opacity) >> 8;
                if (effective_alpha == 0) continue;

                // 合并计算
                uint32_t inv_alpha = 255 - effective_alpha;

                uint32_t r = (static_cast<uint32_t>(src_color.r) * dst_color.r * effective_alpha +
                    dst_color.r * 255 * inv_alpha) >> 16;

                uint32_t g = (static_cast<uint32_t>(src_color.g) * dst_color.g * effective_alpha +
                    dst_color.g * 255 * inv_alpha) >> 16;

                uint32_t b = (static_cast<uint32_t>(src_color.b) * dst_color.b * effective_alpha +
                    dst_color.b * 255 * inv_alpha) >> 16;

                uint32_t a = effective_alpha +
                    ((static_cast<uint32_t>(dst_color.a) * inv_alpha) >> 8);

                // 确保范围
                dst_color = Color(
                    static_cast<uint8_t>(a > 255 ? 255 : a),
                    static_cast<uint8_t>(r > 255 ? 255 : r),
                    static_cast<uint8_t>(g > 255 ? 255 : g),
                    static_cast<uint8_t>(b > 255 ? 255 : b)
                );
            }
        }
    }

    // ============================================================================
    // 屏幕混合（变亮效果）
    // ============================================================================
    void screenBlend(const Buffer& src, Buffer& dst, int dstX, int dstY, int opacity) {
        if (!src.isValid() || !dst.isValid() || opacity == 0) return;

        // 边界计算
        int startX = std::max(0, dstX);
        int startY = std::max(0, dstY);
        int endX = std::min(dst.width, dstX + src.width);
        int endY = std::min(dst.height, dstY + src.height);

        if (startX >= endX || startY >= endY) return;

        const __m256i global_opacity_vec = _mm256_set1_epi32(opacity);
        const __m128i global_opacity_vec_sse = _mm_set1_epi32(opacity);

        for (int y = startY; y < endY; ++y) {
            int srcY = y - dstY;
            Color* destRow = dst.getRow(y) + startX;
            const Color* srcRow = src.getRow(srcY) + (startX - dstX);
            int rowWidth = endX - startX;

            size_t x = 0;

            // AVX2优化
            const size_t simd_count8 = rowWidth & ~7;
            if (simd_count8 > 0) {
                const __m256i mask_ff = _mm256_set1_epi32(0xFF);
                const __m256i const_255 = _mm256_set1_epi32(255);
                const __m256i zero = _mm256_setzero_si256();

                for (; x < simd_count8; x += 8) {
                    // 预取下一缓存行
                    _mm_prefetch((const char*)(srcRow + x + 16), _MM_HINT_T0);
                    _mm_prefetch((const char*)(destRow + x + 16), _MM_HINT_T0);

                    __m256i src_vec = _mm256_loadu_si256((const __m256i*)(srcRow + x));
                    __m256i dst_vec = _mm256_loadu_si256((const __m256i*)(destRow + x));

                    // 解包RGBA通道
                    __m256i src_a = _mm256_srli_epi32(src_vec, 24);
                    __m256i src_r = _mm256_and_si256(_mm256_srli_epi32(src_vec, 16), mask_ff);
                    __m256i src_g = _mm256_and_si256(_mm256_srli_epi32(src_vec, 8), mask_ff);
                    __m256i src_b = _mm256_and_si256(src_vec, mask_ff);

                    __m256i dst_a = _mm256_srli_epi32(dst_vec, 24);
                    __m256i dst_r = _mm256_and_si256(_mm256_srli_epi32(dst_vec, 16), mask_ff);
                    __m256i dst_g = _mm256_and_si256(_mm256_srli_epi32(dst_vec, 8), mask_ff);
                    __m256i dst_b = _mm256_and_si256(dst_vec, mask_ff);

                    // 计算有效alpha并快速检查
                    __m256i effective_a = _mm256_srli_epi32(
                        _mm256_mullo_epi32(src_a, global_opacity_vec), 8);
                    if (_mm256_testz_si256(effective_a, effective_a)) {
                        continue;
                    }

                    // 屏幕混合公式：src + dst - (src*dst >> 8)
                    __m256i r_product = _mm256_srli_epi32(_mm256_mullo_epi32(src_r, dst_r), 8);
                    __m256i g_product = _mm256_srli_epi32(_mm256_mullo_epi32(src_g, dst_g), 8);
                    __m256i b_product = _mm256_srli_epi32(_mm256_mullo_epi32(src_b, dst_b), 8);

                    __m256i r_blend = _mm256_sub_epi32(_mm256_add_epi32(src_r, dst_r), r_product);
                    __m256i g_blend = _mm256_sub_epi32(_mm256_add_epi32(src_g, dst_g), g_product);
                    __m256i b_blend = _mm256_sub_epi32(_mm256_add_epi32(src_b, dst_b), b_product);

                    // 合并屏幕混合和alpha混合
                    __m256i inv_effective_a = _mm256_sub_epi32(const_255, effective_a);

                    __m256i r_num = _mm256_add_epi32(
                        _mm256_mullo_epi32(r_blend, effective_a),
                        _mm256_mullo_epi32(dst_r, inv_effective_a)
                    );

                    __m256i g_num = _mm256_add_epi32(
                        _mm256_mullo_epi32(g_blend, effective_a),
                        _mm256_mullo_epi32(dst_g, inv_effective_a)
                    );

                    __m256i b_num = _mm256_add_epi32(
                        _mm256_mullo_epi32(b_blend, effective_a),
                        _mm256_mullo_epi32(dst_b, inv_effective_a)
                    );

                    // 右移8位完成除以256
                    __m256i r = _mm256_srli_epi32(r_num, 8);
                    __m256i g = _mm256_srli_epi32(g_num, 8);
                    __m256i b = _mm256_srli_epi32(b_num, 8);

                    // Alpha通道混合
                    __m256i a = _mm256_add_epi32(effective_a,
                        _mm256_srli_epi32(_mm256_mullo_epi32(dst_a, inv_effective_a), 8));

                    // 饱和运算
                    r = _mm256_min_epu8(_mm256_max_epu8(r, zero), const_255);
                    g = _mm256_min_epu8(_mm256_max_epu8(g, zero), const_255);
                    b = _mm256_min_epu8(_mm256_max_epu8(b, zero), const_255);
                    a = _mm256_min_epu8(_mm256_max_epu8(a, zero), const_255);

                    // 打包结果
                    __m256i result = _mm256_or_si256(
                        _mm256_slli_epi32(r, 16),
                        _mm256_slli_epi32(g, 8));
                    result = _mm256_or_si256(result, b);
                    result = _mm256_or_si256(result, _mm256_slli_epi32(a, 24));

                    _mm256_storeu_si256((__m256i*)(destRow + x), result);
                }
            }

            // SSE4优化
            const size_t simd_count4 = rowWidth & ~3;
            if (simd_count4 > x) {
                const __m128i mask_ff = _mm_set1_epi32(0xFF);
                const __m128i const_255 = _mm_set1_epi32(255);
                const __m128i zero = _mm_setzero_si128();

                for (; x < simd_count4; x += 4) {
                    __m128i src_vec = _mm_loadu_si128((const __m128i*)(srcRow + x));
                    __m128i dst_vec = _mm_loadu_si128((const __m128i*)(destRow + x));

                    // 解包
                    __m128i src_a = _mm_srli_epi32(src_vec, 24);
                    __m128i src_r = _mm_and_si128(_mm_srli_epi32(src_vec, 16), mask_ff);
                    __m128i src_g = _mm_and_si128(_mm_srli_epi32(src_vec, 8), mask_ff);
                    __m128i src_b = _mm_and_si128(src_vec, mask_ff);

                    __m128i dst_a = _mm_srli_epi32(dst_vec, 24);
                    __m128i dst_r = _mm_and_si128(_mm_srli_epi32(dst_vec, 16), mask_ff);
                    __m128i dst_g = _mm_and_si128(_mm_srli_epi32(dst_vec, 8), mask_ff);
                    __m128i dst_b = _mm_and_si128(dst_vec, mask_ff);

                    // 计算有效alpha
                    __m128i effective_a = _mm_srli_epi32(
                        _mm_mullo_epi32(src_a, global_opacity_vec_sse), 8);
                    if (_mm_test_all_zeros(effective_a, effective_a)) {
                        continue;
                    }

                    // 屏幕混合计算
                    __m128i r_product = _mm_srli_epi32(_mm_mullo_epi32(src_r, dst_r), 8);
                    __m128i g_product = _mm_srli_epi32(_mm_mullo_epi32(src_g, dst_g), 8);
                    __m128i b_product = _mm_srli_epi32(_mm_mullo_epi32(src_b, dst_b), 8);

                    __m128i r_blend = _mm_sub_epi32(_mm_add_epi32(src_r, dst_r), r_product);
                    __m128i g_blend = _mm_sub_epi32(_mm_add_epi32(src_g, dst_g), g_product);
                    __m128i b_blend = _mm_sub_epi32(_mm_add_epi32(src_b, dst_b), b_product);

                    // 合并计算
                    __m128i inv_effective_a = _mm_sub_epi32(const_255, effective_a);

                    __m128i r_num = _mm_add_epi32(
                        _mm_mullo_epi32(r_blend, effective_a),
                        _mm_mullo_epi32(dst_r, inv_effective_a)
                    );

                    __m128i g_num = _mm_add_epi32(
                        _mm_mullo_epi32(g_blend, effective_a),
                        _mm_mullo_epi32(dst_g, inv_effective_a)
                    );

                    __m128i b_num = _mm_add_epi32(
                        _mm_mullo_epi32(b_blend, effective_a),
                        _mm_mullo_epi32(dst_b, inv_effective_a)
                    );

                    __m128i r = _mm_srli_epi32(r_num, 8);
                    __m128i g = _mm_srli_epi32(g_num, 8);
                    __m128i b = _mm_srli_epi32(b_num, 8);
                    __m128i a = _mm_add_epi32(effective_a,
                        _mm_srli_epi32(_mm_mullo_epi32(dst_a, inv_effective_a), 8));

                    // 饱和运算
                    r = _mm_min_epu8(_mm_max_epu8(r, zero), const_255);
                    g = _mm_min_epu8(_mm_max_epu8(g, zero), const_255);
                    b = _mm_min_epu8(_mm_max_epu8(b, zero), const_255);
                    a = _mm_min_epu8(_mm_max_epu8(a, zero), const_255);

                    // 打包
                    __m128i result = _mm_or_si128(
                        _mm_slli_epi32(r, 16),
                        _mm_slli_epi32(g, 8));
                    result = _mm_or_si128(result, b);
                    result = _mm_or_si128(result, _mm_slli_epi32(a, 24));

                    _mm_storeu_si128((__m128i*)(destRow + x), result);
                }
            }

            // 标量处理
            for (; x < rowWidth; ++x) {
                const Color& src_color = srcRow[x];
                Color& dst_color = destRow[x];

                // 快速路径检查
                if (src_color.a == 0 || opacity == 0) continue;

                uint32_t effective_alpha = (static_cast<uint32_t>(src_color.a) * opacity) >> 8;
                if (effective_alpha == 0) continue;

                // 屏幕混合公式
                uint32_t r_product = (static_cast<uint32_t>(src_color.r) * dst_color.r) >> 8;
                uint32_t g_product = (static_cast<uint32_t>(src_color.g) * dst_color.g) >> 8;
                uint32_t b_product = (static_cast<uint32_t>(src_color.b) * dst_color.b) >> 8;

                uint32_t r_blend = src_color.r + dst_color.r - r_product;
                uint32_t g_blend = src_color.g + dst_color.g - g_product;
                uint32_t b_blend = src_color.b + dst_color.b - b_product;

                // 合并计算
                uint32_t inv_alpha = 255 - effective_alpha;

                uint32_t r = (r_blend * effective_alpha +
                    dst_color.r * inv_alpha) >> 8;

                uint32_t g = (g_blend * effective_alpha +
                    dst_color.g * inv_alpha) >> 8;

                uint32_t b = (b_blend * effective_alpha +
                    dst_color.b * inv_alpha) >> 8;

                uint32_t a = effective_alpha +
                    ((static_cast<uint32_t>(dst_color.a) * inv_alpha) >> 8);

                // 确保范围
                dst_color = Color(
                    static_cast<uint8_t>(a > 255 ? 255 : a),
                    static_cast<uint8_t>(r > 255 ? 255 : r),
                    static_cast<uint8_t>(g > 255 ? 255 : g),
                    static_cast<uint8_t>(b > 255 ? 255 : b)
                );
            }
        }
    }

    // ============================================================================
    // 叠加混合（增强对比度）
    // ============================================================================
    void overlayBlend(const Buffer& src, Buffer& dst, int dstX, int dstY, int opacity) {
        if (!src.isValid() || !dst.isValid() || opacity == 0) return;

        // 边界计算
        int startX = std::max(0, dstX);
        int startY = std::max(0, dstY);
        int endX = std::min(dst.width, dstX + src.width);
        int endY = std::min(dst.height, dstY + src.height);

        if (startX >= endX || startY >= endY) return;

        const __m256i global_opacity_vec = _mm256_set1_epi32(opacity);
        const __m128i global_opacity_vec_sse = _mm_set1_epi32(opacity);

        for (int y = startY; y < endY; ++y) {
            int srcY = y - dstY;
            Color* destRow = dst.getRow(y) + startX;
            const Color* srcRow = src.getRow(srcY) + (startX - dstX);
            int rowWidth = endX - startX;

            size_t x = 0;

            // AVX2优化
            const size_t simd_count8 = rowWidth & ~7;
            if (simd_count8 > 0) {
                const __m256i mask_ff = _mm256_set1_epi32(0xFF);
                const __m256i const_255 = _mm256_set1_epi32(255);
                const __m256i zero = _mm256_setzero_si256();
                const __m256i half_val = _mm256_set1_epi32(128);

                for (; x < simd_count8; x += 8) {
                    // 预取下一缓存行
                    _mm_prefetch((const char*)(srcRow + x + 16), _MM_HINT_T0);
                    _mm_prefetch((const char*)(destRow + x + 16), _MM_HINT_T0);

                    __m256i src_vec = _mm256_loadu_si256((const __m256i*)(srcRow + x));
                    __m256i dst_vec = _mm256_loadu_si256((const __m256i*)(destRow + x));

                    // 解包RGBA通道
                    __m256i src_a = _mm256_srli_epi32(src_vec, 24);
                    __m256i src_r = _mm256_and_si256(_mm256_srli_epi32(src_vec, 16), mask_ff);
                    __m256i src_g = _mm256_and_si256(_mm256_srli_epi32(src_vec, 8), mask_ff);
                    __m256i src_b = _mm256_and_si256(src_vec, mask_ff);

                    __m256i dst_a = _mm256_srli_epi32(dst_vec, 24);
                    __m256i dst_r = _mm256_and_si256(_mm256_srli_epi32(dst_vec, 16), mask_ff);
                    __m256i dst_g = _mm256_and_si256(_mm256_srli_epi32(dst_vec, 8), mask_ff);
                    __m256i dst_b = _mm256_and_si256(dst_vec, mask_ff);

                    // 计算有效alpha并快速检查
                    __m256i effective_a = _mm256_srli_epi32(
                        _mm256_mullo_epi32(src_a, global_opacity_vec), 8);
                    if (_mm256_testz_si256(effective_a, effective_a)) {
                        continue;
                    }

                    // 无分支叠加混合计算
                    auto overlay_channel_avx = [&](__m256i src_ch, __m256i dst_ch) -> __m256i {
                        // 计算两种情况的中间结果
                        __m256i dark = _mm256_srli_epi32(_mm256_mullo_epi32(src_ch, dst_ch), 7); // *2/256

                        __m256i inv_src = _mm256_sub_epi32(const_255, src_ch);
                        __m256i inv_dst = _mm256_sub_epi32(const_255, dst_ch);
                        __m256i light = _mm256_sub_epi32(const_255,
                            _mm256_srli_epi32(_mm256_mullo_epi32(inv_src, inv_dst), 7));

                        // 使用比较结果作为混合掩码
                        __m256i use_light_mask = _mm256_cmpgt_epi32(dst_ch, half_val);

                        return _mm256_blendv_epi8(dark, light, use_light_mask);
                    };

                    // 计算叠加混合结果
                    __m256i r_blend = overlay_channel_avx(src_r, dst_r);
                    __m256i g_blend = overlay_channel_avx(src_g, dst_g);
                    __m256i b_blend = overlay_channel_avx(src_b, dst_b);

                    // 混合计算
                    __m256i inv_effective_a = _mm256_sub_epi32(const_255, effective_a);

                    __m256i r = _mm256_srli_epi32(_mm256_add_epi32(
                        _mm256_mullo_epi32(r_blend, effective_a),
                        _mm256_mullo_epi32(dst_r, inv_effective_a)), 8);

                    __m256i g = _mm256_srli_epi32(_mm256_add_epi32(
                        _mm256_mullo_epi32(g_blend, effective_a),
                        _mm256_mullo_epi32(dst_g, inv_effective_a)), 8);

                    __m256i b = _mm256_srli_epi32(_mm256_add_epi32(
                        _mm256_mullo_epi32(b_blend, effective_a),
                        _mm256_mullo_epi32(dst_b, inv_effective_a)), 8);

                    // Alpha通道混合
                    __m256i a = _mm256_add_epi32(effective_a,
                        _mm256_srli_epi32(_mm256_mullo_epi32(dst_a, inv_effective_a), 8));

                    // 饱和运算
                    r = _mm256_min_epu8(_mm256_max_epu8(r, zero), const_255);
                    g = _mm256_min_epu8(_mm256_max_epu8(g, zero), const_255);
                    b = _mm256_min_epu8(_mm256_max_epu8(b, zero), const_255);
                    a = _mm256_min_epu8(_mm256_max_epu8(a, zero), const_255);

                    // 打包结果
                    __m256i result = _mm256_or_si256(
                        _mm256_slli_epi32(r, 16),
                        _mm256_slli_epi32(g, 8));
                    result = _mm256_or_si256(result, b);
                    result = _mm256_or_si256(result, _mm256_slli_epi32(a, 24));

                    _mm256_storeu_si256((__m256i*)(destRow + x), result);
                }
            }

            // SSE4优化
            const size_t simd_count4 = rowWidth & ~3;
            if (simd_count4 > x) {
                const __m128i mask_ff = _mm_set1_epi32(0xFF);
                const __m128i const_255 = _mm_set1_epi32(255);
                const __m128i zero = _mm_setzero_si128();
                const __m128i half_val = _mm_set1_epi32(128);

                for (; x < simd_count4; x += 4) {
                    __m128i src_vec = _mm_loadu_si128((const __m128i*)(srcRow + x));
                    __m128i dst_vec = _mm_loadu_si128((const __m128i*)(destRow + x));

                    // 解包
                    __m128i src_a = _mm_srli_epi32(src_vec, 24);
                    __m128i src_r = _mm_and_si128(_mm_srli_epi32(src_vec, 16), mask_ff);
                    __m128i src_g = _mm_and_si128(_mm_srli_epi32(src_vec, 8), mask_ff);
                    __m128i src_b = _mm_and_si128(src_vec, mask_ff);

                    __m128i dst_a = _mm_srli_epi32(dst_vec, 24);
                    __m128i dst_r = _mm_and_si128(_mm_srli_epi32(dst_vec, 16), mask_ff);
                    __m128i dst_g = _mm_and_si128(_mm_srli_epi32(dst_vec, 8), mask_ff);
                    __m128i dst_b = _mm_and_si128(dst_vec, mask_ff);

                    // 计算有效alpha
                    __m128i effective_a = _mm_srli_epi32(
                        _mm_mullo_epi32(src_a, global_opacity_vec_sse), 8);
                    if (_mm_test_all_zeros(effective_a, effective_a)) {
                        continue;
                    }

                    // 无分支叠加混合（SSE版本）
                    auto overlay_channel_sse = [&](__m128i src_ch, __m128i dst_ch) -> __m128i {
                        __m128i dark = _mm_srli_epi32(_mm_mullo_epi32(src_ch, dst_ch), 7);

                        __m128i inv_src = _mm_sub_epi32(const_255, src_ch);
                        __m128i inv_dst = _mm_sub_epi32(const_255, dst_ch);
                        __m128i light = _mm_sub_epi32(const_255,
                            _mm_srli_epi32(_mm_mullo_epi32(inv_src, inv_dst), 7));

                        __m128i use_light_mask = _mm_cmpgt_epi32(dst_ch, half_val);

                        return _mm_blendv_epi8(dark, light, use_light_mask);
                    };

                    // 计算叠加混合结果
                    __m128i r_blend = overlay_channel_sse(src_r, dst_r);
                    __m128i g_blend = overlay_channel_sse(src_g, dst_g);
                    __m128i b_blend = overlay_channel_sse(src_b, dst_b);

                    // 混合计算
                    __m128i inv_effective_a = _mm_sub_epi32(const_255, effective_a);

                    __m128i r = _mm_srli_epi32(_mm_add_epi32(
                        _mm_mullo_epi32(r_blend, effective_a),
                        _mm_mullo_epi32(dst_r, inv_effective_a)), 8);

                    __m128i g = _mm_srli_epi32(_mm_add_epi32(
                        _mm_mullo_epi32(g_blend, effective_a),
                        _mm_mullo_epi32(dst_g, inv_effective_a)), 8);

                    __m128i b = _mm_srli_epi32(_mm_add_epi32(
                        _mm_mullo_epi32(b_blend, effective_a),
                        _mm_mullo_epi32(dst_b, inv_effective_a)), 8);

                    __m128i a = _mm_add_epi32(effective_a,
                        _mm_srli_epi32(_mm_mullo_epi32(dst_a, inv_effective_a), 8));

                    // 饱和运算
                    r = _mm_min_epu8(_mm_max_epu8(r, zero), const_255);
                    g = _mm_min_epu8(_mm_max_epu8(g, zero), const_255);
                    b = _mm_min_epu8(_mm_max_epu8(b, zero), const_255);
                    a = _mm_min_epu8(_mm_max_epu8(a, zero), const_255);

                    // 打包
                    __m128i result = _mm_or_si128(
                        _mm_slli_epi32(r, 16),
                        _mm_slli_epi32(g, 8));
                    result = _mm_or_si128(result, b);
                    result = _mm_or_si128(result, _mm_slli_epi32(a, 24));

                    _mm_storeu_si128((__m128i*)(destRow + x), result);
                }
            }

            // 标量处理
            for (; x < rowWidth; ++x) {
                const Color& src_color = srcRow[x];
                Color& dst_color = destRow[x];

                // 快速路径检查
                if (src_color.a == 0 || opacity == 0) continue;

                uint32_t effective_alpha = (static_cast<uint32_t>(src_color.a) * opacity) >> 8;
                if (effective_alpha == 0) continue;

                // 无分支叠加混合计算
                auto overlay_channel_scalar = [](uint32_t s, uint32_t d) -> uint32_t {
                    // 预先计算两种情况
                    uint32_t dark = (2 * s * d) >> 8;
                    uint32_t light = 255 - ((2 * (255 - s) * (255 - d)) >> 8);

                    // 使用条件选择替代分支
                    return d < 128 ? dark : light;
                    };

                uint32_t r_blend = overlay_channel_scalar(src_color.r, dst_color.r);
                uint32_t g_blend = overlay_channel_scalar(src_color.g, dst_color.g);
                uint32_t b_blend = overlay_channel_scalar(src_color.b, dst_color.b);

                // 混合计算
                uint32_t inv_alpha = 255 - effective_alpha;

                uint32_t r = (r_blend * effective_alpha +
                    dst_color.r * inv_alpha) >> 8;

                uint32_t g = (g_blend * effective_alpha +
                    dst_color.g * inv_alpha) >> 8;

                uint32_t b = (b_blend * effective_alpha +
                    dst_color.b * inv_alpha) >> 8;

                uint32_t a = effective_alpha +
                    ((static_cast<uint32_t>(dst_color.a) * inv_alpha) >> 8);

                // 确保范围
                dst_color = Color(
                    static_cast<uint8_t>(a > 255 ? 255 : a),
                    static_cast<uint8_t>(r > 255 ? 255 : r),
                    static_cast<uint8_t>(g > 255 ? 255 : g),
                    static_cast<uint8_t>(b > 255 ? 255 : b)
                );
            }
        }
    }

    // ============================================================================
    // 目标Alpha混合（基于目标alpha的混合）
    // ============================================================================
    void destAlphaBlend(const Buffer& src, Buffer& dst, int dstX, int dstY, int opacity) {
        if (!src.isValid() || !dst.isValid() || opacity == 0) return;

        // 边界计算
        int startX = std::max(0, dstX);
        int startY = std::max(0, dstY);
        int endX = std::min(dst.width, dstX + src.width);
        int endY = std::min(dst.height, dstY + src.height);

        if (startX >= endX || startY >= endY) return;

        // 在循环外设置常量
        const __m256i global_opacity_vec = _mm256_set1_epi32(opacity);
        const __m128i global_opacity_vec_sse = _mm_set1_epi32(opacity);

        for (int y = startY; y < endY; ++y) {
            int srcY = y - dstY;
            Color* destRow = dst.getRow(y) + startX;
            const Color* srcRow = src.getRow(srcY) + (startX - dstX);
            int rowWidth = endX - startX;

            size_t x = 0;

            // AVX2优化
            const size_t simd_count8 = rowWidth & ~7;
            if (simd_count8 > 0) {
                const __m256i mask_ff = _mm256_set1_epi32(0xFF);
                const __m256i const_255 = _mm256_set1_epi32(255);
                const __m256i zero = _mm256_setzero_si256();

                for (; x < simd_count8; x += 8) {
                    // 预取下一缓存行
                    _mm_prefetch((const char*)(srcRow + x + 16), _MM_HINT_T0);
                    _mm_prefetch((const char*)(destRow + x + 16), _MM_HINT_T0);

                    __m256i src_vec = _mm256_loadu_si256((const __m256i*)(srcRow + x));
                    __m256i dst_vec = _mm256_loadu_si256((const __m256i*)(destRow + x));

                    // 解包RGBA通道
                    __m256i src_a = _mm256_srli_epi32(src_vec, 24);
                    __m256i src_r = _mm256_and_si256(_mm256_srli_epi32(src_vec, 16), mask_ff);
                    __m256i src_g = _mm256_and_si256(_mm256_srli_epi32(src_vec, 8), mask_ff);
                    __m256i src_b = _mm256_and_si256(src_vec, mask_ff);

                    __m256i dst_a = _mm256_srli_epi32(dst_vec, 24);
                    __m256i dst_r = _mm256_and_si256(_mm256_srli_epi32(dst_vec, 16), mask_ff);
                    __m256i dst_g = _mm256_and_si256(_mm256_srli_epi32(dst_vec, 8), mask_ff);
                    __m256i dst_b = _mm256_and_si256(dst_vec, mask_ff);

                    // 合并计算步骤：effective_blend_alpha = (src.a * dst.a * opacity) >> 16
                    __m256i effective_blend_alpha = _mm256_srli_epi32(
                        _mm256_mullo_epi32(_mm256_mullo_epi32(src_a, dst_a), global_opacity_vec), 16);

                    // 快速路径：如果所有混合alpha都为0，跳过处理
                    if (_mm256_testz_si256(effective_blend_alpha, effective_blend_alpha)) {
                        continue;
                    }

                    // 合并颜色和alpha混合计算
                    __m256i inv_blend_alpha = _mm256_sub_epi32(const_255, effective_blend_alpha);

                    __m256i r_num = _mm256_add_epi32(
                        _mm256_mullo_epi32(src_r, effective_blend_alpha),
                        _mm256_mullo_epi32(dst_r, inv_blend_alpha)
                    );
                    __m256i g_num = _mm256_add_epi32(
                        _mm256_mullo_epi32(src_g, effective_blend_alpha),
                        _mm256_mullo_epi32(dst_g, inv_blend_alpha)
                    );
                    __m256i b_num = _mm256_add_epi32(
                        _mm256_mullo_epi32(src_b, effective_blend_alpha),
                        _mm256_mullo_epi32(dst_b, inv_blend_alpha)
                    );

                    // Alpha通道混合
                    __m256i a_num = _mm256_add_epi32(
                        _mm256_mullo_epi32(effective_blend_alpha, const_255),
                        _mm256_mullo_epi32(dst_a, inv_blend_alpha)
                    );

                    // 右移8位完成除以256
                    __m256i r = _mm256_srli_epi32(r_num, 8);
                    __m256i g = _mm256_srli_epi32(g_num, 8);
                    __m256i b = _mm256_srli_epi32(b_num, 8);
                    __m256i a = _mm256_srli_epi32(a_num, 8);

                    // 饱和运算
                    r = _mm256_min_epu8(_mm256_max_epu8(r, zero), const_255);
                    g = _mm256_min_epu8(_mm256_max_epu8(g, zero), const_255);
                    b = _mm256_min_epu8(_mm256_max_epu8(b, zero), const_255);
                    a = _mm256_min_epu8(_mm256_max_epu8(a, zero), const_255);

                    // 打包结果
                    __m256i result = _mm256_or_si256(
                        _mm256_slli_epi32(r, 16),
                        _mm256_slli_epi32(g, 8));
                    result = _mm256_or_si256(result, b);
                    result = _mm256_or_si256(result, _mm256_slli_epi32(a, 24));

                    _mm256_storeu_si256((__m256i*)(destRow + x), result);
                }
            }

            // SSE4优化
            const size_t simd_count4 = rowWidth & ~3;
            if (simd_count4 > x) {
                const __m128i mask_ff = _mm_set1_epi32(0xFF);
                const __m128i const_255 = _mm_set1_epi32(255);
                const __m128i zero = _mm_setzero_si128();

                for (; x < simd_count4; x += 4) {
                    __m128i src_vec = _mm_loadu_si128((const __m128i*)(srcRow + x));
                    __m128i dst_vec = _mm_loadu_si128((const __m128i*)(destRow + x));

                    // 解包
                    __m128i src_a = _mm_srli_epi32(src_vec, 24);
                    __m128i src_r = _mm_and_si128(_mm_srli_epi32(src_vec, 16), mask_ff);
                    __m128i src_g = _mm_and_si128(_mm_srli_epi32(src_vec, 8), mask_ff);
                    __m128i src_b = _mm_and_si128(src_vec, mask_ff);

                    __m128i dst_a = _mm_srli_epi32(dst_vec, 24);
                    __m128i dst_r = _mm_and_si128(_mm_srli_epi32(dst_vec, 16), mask_ff);
                    __m128i dst_g = _mm_and_si128(_mm_srli_epi32(dst_vec, 8), mask_ff);
                    __m128i dst_b = _mm_and_si128(dst_vec, mask_ff);

                    // 合并计算步骤
                    __m128i effective_blend_alpha = _mm_srli_epi32(
                        _mm_mullo_epi32(_mm_mullo_epi32(src_a, dst_a), global_opacity_vec_sse), 16);

                    // 快速检查
                    if (_mm_test_all_zeros(effective_blend_alpha, effective_blend_alpha)) {
                        continue;
                    }

                    // 合并混合计算
                    __m128i inv_blend_alpha = _mm_sub_epi32(const_255, effective_blend_alpha);

                    __m128i r_num = _mm_add_epi32(
                        _mm_mullo_epi32(src_r, effective_blend_alpha),
                        _mm_mullo_epi32(dst_r, inv_blend_alpha)
                    );
                    __m128i g_num = _mm_add_epi32(
                        _mm_mullo_epi32(src_g, effective_blend_alpha),
                        _mm_mullo_epi32(dst_g, inv_blend_alpha)
                    );
                    __m128i b_num = _mm_add_epi32(
                        _mm_mullo_epi32(src_b, effective_blend_alpha),
                        _mm_mullo_epi32(dst_b, inv_blend_alpha)
                    );
                    __m128i a_num = _mm_add_epi32(
                        _mm_mullo_epi32(effective_blend_alpha, const_255),
                        _mm_mullo_epi32(dst_a, inv_blend_alpha)
                    );

                    __m128i r = _mm_srli_epi32(r_num, 8);
                    __m128i g = _mm_srli_epi32(g_num, 8);
                    __m128i b = _mm_srli_epi32(b_num, 8);
                    __m128i a = _mm_srli_epi32(a_num, 8);

                    // 饱和运算
                    r = _mm_min_epu8(_mm_max_epu8(r, zero), const_255);
                    g = _mm_min_epu8(_mm_max_epu8(g, zero), const_255);
                    b = _mm_min_epu8(_mm_max_epu8(b, zero), const_255);
                    a = _mm_min_epu8(_mm_max_epu8(a, zero), const_255);

                    // 打包
                    __m128i result = _mm_or_si128(
                        _mm_slli_epi32(r, 16),
                        _mm_slli_epi32(g, 8));
                    result = _mm_or_si128(result, b);
                    result = _mm_or_si128(result, _mm_slli_epi32(a, 24));

                    _mm_storeu_si128((__m128i*)(destRow + x), result);
                }
            }

            // 标量处理
            for (; x < rowWidth; ++x) {
                const Color& src_color = srcRow[x];
                Color& dst_color = destRow[x];

                // 合并计算步骤：effective_blend_alpha = (src.a * dst.a * opacity) >> 16
                uint32_t effective_blend_alpha = (static_cast<uint32_t>(src_color.a) *
                    static_cast<uint32_t>(dst_color.a) * opacity) >> 16;

                // 快速路径检查
                if (effective_blend_alpha == 0) {
                    continue;
                }

                // 合并混合计算
                uint32_t inv_blend_alpha = 255 - effective_blend_alpha;

                uint32_t r = (static_cast<uint32_t>(src_color.r) * effective_blend_alpha +
                    static_cast<uint32_t>(dst_color.r) * inv_blend_alpha) >> 8;

                uint32_t g = (static_cast<uint32_t>(src_color.g) * effective_blend_alpha +
                    static_cast<uint32_t>(dst_color.g) * inv_blend_alpha) >> 8;

                uint32_t b = (static_cast<uint32_t>(src_color.b) * effective_blend_alpha +
                    static_cast<uint32_t>(dst_color.b) * inv_blend_alpha) >> 8;

                uint32_t a = (effective_blend_alpha * 255 +
                    static_cast<uint32_t>(dst_color.a) * inv_blend_alpha) >> 8;

                // 确保范围
                dst_color = Color(
                    static_cast<uint8_t>(a > 255 ? 255 : a),
                    static_cast<uint8_t>(r > 255 ? 255 : r),
                    static_cast<uint8_t>(g > 255 ? 255 : g),
                    static_cast<uint8_t>(b > 255 ? 255 : b)
                );
            }
        }
    }

} // namespace pa2d