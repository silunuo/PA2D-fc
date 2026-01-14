#include "../include/buffer.h"
#include "../include/buffer_blender.h"
#include "../include/canvas.h"
#include <cstddef>
#include <cstring>
#include <immintrin.h>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace pa2d {
    // 快速通道旋转函数声明
    static inline void drawRotated90Scaled(Buffer& dest, const Buffer& src, float centerX, float centerY, float scaleX, float scaleY);
    static inline void drawRotated180Scaled(Buffer& dest, const Buffer& src, float centerX, float centerY, float scaleX, float scaleY);
    static inline void drawRotated270Scaled(Buffer& dest, const Buffer& src, float centerX, float centerY, float scaleX, float scaleY);

    // 工具函数
    template<bool IS_90_DEG>
    inline void drawOrthogonalRotated(Buffer& dest, const Buffer& src, float centerX, float centerY, float scaleX, float scaleY);

    // SIMD颜色解包/打包函数
    static inline void unpack_colors_avx2(__m256i v_colors, __m256& r, __m256& g, __m256& b, __m256& a);
    static inline __m256i pack_colors_avx2(__m256 r, __m256 g, __m256 b, __m256 a);
    static inline void unpack_colors_sse(__m128i v_colors, __m128& r, __m128& g, __m128& b, __m128& a);
    static inline __m128i pack_colors_sse(__m128 r, __m128 g, __m128 b, __m128 a);

    // 旋转尺寸计算辅助函数
    inline void calculateRotatedSize(float halfW, float halfH, float cosA, float sinA, int& outWidth, int& outHeight);

    // ============================================================================
    // Buffer类成员函数
    // ============================================================================

    Buffer::Buffer(int width, int height, const Color& init_color)
        : width(width), height(height), color(nullptr) {
        if (width > 0 && height > 0) {
            color = new Color[width * height];
            if (init_color) clear(init_color);
        }
    }

    Buffer::Buffer(const Buffer& other) : width(0), height(0), color(nullptr) {
        copy(*this, other);
    }

    Buffer::Buffer(Buffer&& other) noexcept
        : width(other.width), height(other.height), color(other.color) {
        other.width = 0;
        other.height = 0;
        other.color = nullptr;
    }

    Buffer& Buffer::operator=(const Buffer& other) {
        copy(*this, other);
        return *this;
    }

    Buffer& Buffer::operator=(Buffer&& other) noexcept {
        delete[] color;
        width = other.width;
        height = other.height;
        color = other.color;
        other.width = 0;
        other.height = 0;
        other.color = nullptr;
        return *this;
    }

    Buffer::~Buffer() {
        delete[] color;
    }

    Color& Buffer::at(int x, int y) {
        return color[y * width + x];
    }

    const Color& Buffer::at(int x, int y) const {
        return color[y * width + x];
    }

    Color* Buffer::getRow(int y) {
        return color + y * width;
    }

    const Color* Buffer::getRow(int y) const {
        return color + y * width;
    }

    // ============================================================================
    // 基础操作：拷贝、清除、调整大小
    // ============================================================================

    void copy(Buffer& dest, const Buffer& src) {
        if (!src.isValid()) {
            if (dest.color) delete[] dest.color;
            dest.color = nullptr;
            dest.width = dest.height = 0;
            return;
        }

        if (dest.width != src.width || dest.height != src.height) {
            if (dest.color) delete[] dest.color;
            dest.width = src.width;
            dest.height = src.height;
            dest.color = new Color[src.width * src.height];
        }

        if (!dest.isValid()) return;

        size_t total_pixels = static_cast<size_t>(src.width) * src.height;
        Color* dst_ptr = dest.color;
        const Color* src_ptr = src.color;
        const Color* end_ptr = src_ptr + total_pixels;

        // AVX2批量拷贝 (32像素/次，循环展开)
        __m256i data1, data2, data3, data4;
        while (src_ptr + 32 <= end_ptr) {
            data1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr));
            data2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + 8));
            data3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + 16));
            data4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + 24));

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_ptr), data1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_ptr + 8), data2);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_ptr + 16), data3);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_ptr + 24), data4);

            src_ptr += 32;
            dst_ptr += 32;
        }

        // 处理剩余像素
        if (src_ptr + 16 <= end_ptr) {
            data1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr));
            data2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + 8));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_ptr), data1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_ptr + 8), data2);
            src_ptr += 16;
            dst_ptr += 16;
        }

        if (src_ptr + 8 <= end_ptr) {
            data1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_ptr), data1);
            src_ptr += 8;
            dst_ptr += 8;
        }

        if (src_ptr + 4 <= end_ptr) {
            __m128i data_sse = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_ptr), data_sse);
            src_ptr += 4;
            dst_ptr += 4;
        }

        switch (end_ptr - src_ptr) {
        case 3: dst_ptr[2] = src_ptr[2];
        case 2: dst_ptr[1] = src_ptr[1];
        case 1: dst_ptr[0] = src_ptr[0];
        default: break;
        }
    }

    void Buffer::resize(int newWidth, int newHeight, const Color& clear_color) {
        if (newWidth <= 0 || newHeight <= 0) {
            if (color) delete[] color;
            color = nullptr;
            width = height = 0;
            return;
        }

        if (newWidth == width && newHeight == height) return;

        Color* new_color = new Color[newWidth * newHeight];
        if (!new_color) return;

        // 清空新缓冲区
        if (clear_color.data != 0) {
            uint32_t clear_data = clear_color.data;
            int total_pixels = newWidth * newHeight;
            Color* ptr = new_color;
            Color* end_ptr = new_color + total_pixels;

            __m256i color_vec = _mm256_set1_epi32(static_cast<int>(clear_data));

            while (ptr + 32 <= end_ptr) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), color_vec);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr + 8), color_vec);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr + 16), color_vec);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr + 24), color_vec);
                ptr += 32;
            }

            int remaining = end_ptr - ptr;
            if (remaining >= 16) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), color_vec);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr + 8), color_vec);
                ptr += 16;
                remaining -= 16;
            }
            if (remaining >= 8) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), color_vec);
                ptr += 8;
                remaining -= 8;
            }
            if (remaining >= 4) {
                __m128i color_vec_sse = _mm_set1_epi32(static_cast<int>(clear_data));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), color_vec_sse);
                ptr += 4;
                remaining -= 4;
            }

            switch (remaining) {
            case 3: ptr[2].data = clear_data;
            case 2: ptr[1].data = clear_data;
            case 1: ptr[0].data = clear_data;
            default: break;
            }
        }

        // 拷贝现有数据
        if (color && width > 0 && height > 0) {
            int copy_width = std::min(width, newWidth);
            int copy_height = std::min(height, newHeight);

            if (width == newWidth && copy_width == width) {
                size_t total_bytes = static_cast<size_t>(copy_height) * width * sizeof(Color);
                Color* dst_ptr = new_color;
                const Color* src_ptr = color;
                const Color* end_ptr = src_ptr + copy_height * width;

                while (src_ptr + 32 <= end_ptr) {
                    __m256i data1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr));
                    __m256i data2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + 8));
                    __m256i data3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + 16));
                    __m256i data4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + 24));

                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_ptr), data1);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_ptr + 8), data2);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_ptr + 16), data3);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_ptr + 24), data4);

                    src_ptr += 32;
                    dst_ptr += 32;
                }

                size_t copied_bytes = (src_ptr - color) * sizeof(Color);
                if (copied_bytes < total_bytes) {
                    memcpy(dst_ptr, src_ptr, total_bytes - copied_bytes);
                }
            }
            else {
                int bytes_per_row = copy_width * sizeof(Color);
                for (int y = 0; y < copy_height; ++y) {
                    Color* dst_row = new_color + y * newWidth;
                    const Color* src_row = color + y * width;

                    int x = 0;
                    for (; x + 32 <= copy_width; x += 32) {
                        __m256i data1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_row + x));
                        __m256i data2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_row + x + 8));
                        __m256i data3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_row + x + 16));
                        __m256i data4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_row + x + 24));

                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_row + x), data1);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_row + x + 8), data2);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_row + x + 16), data3);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_row + x + 24), data4);
                    }

                    if (x < copy_width) {
                        memcpy(dst_row + x, src_row + x, (copy_width - x) * sizeof(Color));
                    }
                }
            }
        }

        delete[] color;
        color = new_color;
        width = newWidth;
        height = newHeight;
    }

    void Buffer::clear(const Color& clear_color) {
        if (!isValid()) return;

        int total_pixels = width * height;
        uint32_t clear_data = clear_color.data;
        Color* ptr = color;
        Color* end_ptr = color + total_pixels;

        __m256i color_vec = _mm256_set1_epi32(static_cast<int>(clear_data));
        while (ptr + 8 <= end_ptr) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), color_vec);
            ptr += 8;
        }

        if (ptr + 4 <= end_ptr) {
            __m128i color_vec_sse = _mm_set1_epi32(static_cast<int>(clear_data));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), color_vec_sse);
            ptr += 4;
        }

        switch (end_ptr - ptr) {
        case 3: ptr[2].data = clear_data;
        case 2: ptr[1].data = clear_data;
        case 1: ptr[0].data = clear_data;
        default: break;
        }
    }

    // ============================================================================
    // SIMD颜色转换函数
    // ============================================================================

    static inline void unpack_colors_avx2(__m256i v_colors, __m256& r, __m256& g, __m256& b, __m256& a) {
        const __m256i mask_FF = _mm256_set1_epi32(0xFF);

        __m256i i_b = _mm256_and_si256(v_colors, mask_FF);
        b = _mm256_cvtepi32_ps(i_b);

        __m256i i_g = _mm256_and_si256(_mm256_srli_epi32(v_colors, 8), mask_FF);
        g = _mm256_cvtepi32_ps(i_g);

        __m256i i_r = _mm256_and_si256(_mm256_srli_epi32(v_colors, 16), mask_FF);
        r = _mm256_cvtepi32_ps(i_r);

        __m256i i_a = _mm256_srli_epi32(v_colors, 24);
        a = _mm256_cvtepi32_ps(i_a);
    }

    static inline __m256i pack_colors_avx2(__m256 r, __m256 g, __m256 b, __m256 a) {
        __m256i ir = _mm256_cvtps_epi32(r);
        __m256i ig = _mm256_cvtps_epi32(g);
        __m256i ib = _mm256_cvtps_epi32(b);
        __m256i ia = _mm256_cvtps_epi32(a);

        __m256i res = ib;
        res = _mm256_or_si256(res, _mm256_slli_epi32(ig, 8));
        res = _mm256_or_si256(res, _mm256_slli_epi32(ir, 16));
        res = _mm256_or_si256(res, _mm256_slli_epi32(ia, 24));
        return res;
    }

    static inline void unpack_colors_sse(__m128i v_colors, __m128& r, __m128& g, __m128& b, __m128& a) {
        const __m128i mask_FF = _mm_set1_epi32(0xFF);

        __m128i i_b = _mm_and_si128(v_colors, mask_FF);
        b = _mm_cvtepi32_ps(i_b);

        __m128i i_g = _mm_and_si128(_mm_srli_epi32(v_colors, 8), mask_FF);
        g = _mm_cvtepi32_ps(i_g);

        __m128i i_r = _mm_and_si128(_mm_srli_epi32(v_colors, 16), mask_FF);
        r = _mm_cvtepi32_ps(i_r);

        __m128i i_a = _mm_srli_epi32(v_colors, 24);
        a = _mm_cvtepi32_ps(i_a);
    }

    static inline __m128i pack_colors_sse(__m128 r, __m128 g, __m128 b, __m128 a) {
        __m128i ir = _mm_cvtps_epi32(r);
        __m128i ig = _mm_cvtps_epi32(g);
        __m128i ib = _mm_cvtps_epi32(b);
        __m128i ia = _mm_cvtps_epi32(a);

        __m128i res = ib;
        res = _mm_or_si128(res, _mm_slli_epi32(ig, 8));
        res = _mm_or_si128(res, _mm_slli_epi32(ir, 16));
        res = _mm_or_si128(res, _mm_slli_epi32(ia, 24));
        return res;
    }

    // ============================================================================
    // 图像处理函数：裁剪、缩放、旋转
    // ============================================================================

    Buffer crop(const Buffer& src, int x, int y, int w, int h) {
        if (!src.isValid() || w <= 0 || h <= 0) {
            return Buffer();
        }

        x = std::max(0, x);
        y = std::max(0, y);
        w = std::min(w, src.width - x);
        h = std::min(h, src.height - y);

        if (w <= 0 || h <= 0) {
            return Buffer();
        }

        Buffer result(w, h);
        if (!result.isValid()) {
            return Buffer();
        }

        const Color* src_data = src.color + y * src.width + x;
        Color* dst_data = result.color;
        const size_t row_bytes = static_cast<size_t>(w) * sizeof(Color);

        for (int row = 0; row < h; ++row) {
            const Color* src_row = src_data + row * src.width;
            Color* dst_row = dst_data + row * w;
            std::memcpy(dst_row, src_row, row_bytes);
        }

        return result;
    }

    Buffer scaled(const Buffer& src, float scaleX, float scaleY) {
        if (!src.isValid() || scaleX <= 0.0f || scaleY <= 0.0f) return Buffer();

        int newWidth = std::max(1, static_cast<int>(src.width * scaleX));
        int newHeight = std::max(1, static_cast<int>(src.height * scaleY));

        if (src.width == newWidth && src.height == newHeight) {
            Buffer result;
            copy(result, src);
            return result;
        }

        Buffer result(newWidth, newHeight, 0);
        if (!result.isValid()) return Buffer();

        const float invScaleX = static_cast<float>(src.width) / newWidth;
        const float invScaleY = static_cast<float>(src.height) / newHeight;
        const float srcW_limit = static_cast<float>(src.width) - 1.05f;
        const int src_stride = src.width;

        const __m256 v_invScaleX = _mm256_set1_ps(invScaleX);
        const __m256 v_ones = _mm256_set1_ps(1.0f);
        const __m256 v_idx_step8 = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256 v_width_limit = _mm256_set1_ps(srcW_limit);

        const __m128 v_invScaleX_sse = _mm_set1_ps(invScaleX);
        const __m128 v_idx_step4 = _mm_setr_ps(0, 1, 2, 3);
        const __m128 v_width_limit_sse = _mm_set1_ps(srcW_limit);
        const __m128 v_ones_sse = _mm_set1_ps(1.0f);

        for (int y = 0; y < newHeight; ++y) {
            const float srcY = y * invScaleY;
            const int sy0 = static_cast<int>(srcY);
            const int sy1 = std::min(sy0 + 1, src.height - 1);
            const float dy = srcY - sy0;

            const __m256 v_dy = _mm256_set1_ps(dy);
            const __m256 v_inv_dy = _mm256_sub_ps(v_ones, v_dy);

            const __m128 v_dy_sse = _mm_set1_ps(dy);
            const __m128 v_inv_dy_sse = _mm_sub_ps(v_ones_sse, v_dy_sse);

            Color* row_ptr = result.getRow(y);
            const Color* src_base = src.color;

            int x = 0;
            // AVX2主循环：每次处理8个像素
            for (; x <= newWidth - 8; x += 8) {
                __m256 v_x_base = _mm256_set1_ps((float)x);
                __m256 v_srcX = _mm256_mul_ps(_mm256_add_ps(v_x_base, v_idx_step8), v_invScaleX);
                v_srcX = _mm256_min_ps(v_srcX, v_width_limit);

                __m256 v_floor_x = _mm256_floor_ps(v_srcX);
                __m256i v_sx0 = _mm256_cvtps_epi32(v_floor_x);
                __m256 v_dx = _mm256_sub_ps(v_srcX, v_floor_x);
                __m256 v_inv_dx = _mm256_sub_ps(v_ones, v_dx);

                __m256 v_w1 = _mm256_mul_ps(v_inv_dx, v_inv_dy);
                __m256 v_w2 = _mm256_mul_ps(v_dx, v_inv_dy);
                __m256 v_w3 = _mm256_mul_ps(v_inv_dx, v_dy);
                __m256 v_w4 = _mm256_mul_ps(v_dx, v_dy);

                __m256i v_offset0 = _mm256_add_epi32(_mm256_set1_epi32(sy0 * src_stride), v_sx0);
                __m256i v_offset1 = _mm256_add_epi32(_mm256_set1_epi32(sy1 * src_stride), v_sx0);

                __m256i v_c00 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(src_base), v_offset0, 4);
                __m256i v_c10 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(src_base), _mm256_add_epi32(v_offset0, _mm256_set1_epi32(1)), 4);
                __m256i v_c01 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(src_base), v_offset1, 4);
                __m256i v_c11 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(src_base), _mm256_add_epi32(v_offset1, _mm256_set1_epi32(1)), 4);

                __m256 r00, g00, b00, a00; unpack_colors_avx2(v_c00, r00, g00, b00, a00);
                __m256 r10, g10, b10, a10; unpack_colors_avx2(v_c10, r10, g10, b10, a10);
                __m256 r01, g01, b01, a01; unpack_colors_avx2(v_c01, r01, g01, b01, a01);
                __m256 r11, g11, b11, a11; unpack_colors_avx2(v_c11, r11, g11, b11, a11);

                __m256 res_r = _mm256_fmadd_ps(r00, v_w1, _mm256_fmadd_ps(r10, v_w2, _mm256_fmadd_ps(r01, v_w3, _mm256_mul_ps(r11, v_w4))));
                __m256 res_g = _mm256_fmadd_ps(g00, v_w1, _mm256_fmadd_ps(g10, v_w2, _mm256_fmadd_ps(g01, v_w3, _mm256_mul_ps(g11, v_w4))));
                __m256 res_b = _mm256_fmadd_ps(b00, v_w1, _mm256_fmadd_ps(b10, v_w2, _mm256_fmadd_ps(b01, v_w3, _mm256_mul_ps(b11, v_w4))));
                __m256 res_a = _mm256_fmadd_ps(a00, v_w1, _mm256_fmadd_ps(a10, v_w2, _mm256_fmadd_ps(a01, v_w3, _mm256_mul_ps(a11, v_w4))));

                __m256i v_final = pack_colors_avx2(res_r, res_g, res_b, res_a);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(row_ptr + x), v_final);
            }

            // SSE循环：每次处理4个像素
            for (; x <= newWidth - 4; x += 4) {
                __m128 v_x_base = _mm_set1_ps((float)x);
                __m128 v_srcX = _mm_mul_ps(_mm_add_ps(v_x_base, v_idx_step4), v_invScaleX_sse);
                v_srcX = _mm_min_ps(v_srcX, v_width_limit_sse);

                __m128 v_floor_x = _mm_floor_ps(v_srcX);
                __m128i v_sx0 = _mm_cvtps_epi32(v_floor_x);
                __m128 v_dx = _mm_sub_ps(v_srcX, v_floor_x);
                __m128 v_inv_dx = _mm_sub_ps(v_ones_sse, v_dx);

                __m128 v_w1 = _mm_mul_ps(v_inv_dx, v_inv_dy_sse);
                __m128 v_w2 = _mm_mul_ps(v_dx, v_inv_dy_sse);
                __m128 v_w3 = _mm_mul_ps(v_inv_dx, v_dy_sse);
                __m128 v_w4 = _mm_mul_ps(v_dx, v_dy_sse);

                alignas(16) int indices[4];
                _mm_store_si128(reinterpret_cast<__m128i*>(indices), v_sx0);

                __m128i v_c00 = _mm_set_epi32(
                    reinterpret_cast<const int*>(src_base)[sy0 * src_stride + indices[3]],
                    reinterpret_cast<const int*>(src_base)[sy0 * src_stride + indices[2]],
                    reinterpret_cast<const int*>(src_base)[sy0 * src_stride + indices[1]],
                    reinterpret_cast<const int*>(src_base)[sy0 * src_stride + indices[0]]
                );

                __m128i v_c10 = _mm_set_epi32(
                    reinterpret_cast<const int*>(src_base)[sy0 * src_stride + indices[3] + 1],
                    reinterpret_cast<const int*>(src_base)[sy0 * src_stride + indices[2] + 1],
                    reinterpret_cast<const int*>(src_base)[sy0 * src_stride + indices[1] + 1],
                    reinterpret_cast<const int*>(src_base)[sy0 * src_stride + indices[0] + 1]
                );

                __m128i v_c01 = _mm_set_epi32(
                    reinterpret_cast<const int*>(src_base)[sy1 * src_stride + indices[3]],
                    reinterpret_cast<const int*>(src_base)[sy1 * src_stride + indices[2]],
                    reinterpret_cast<const int*>(src_base)[sy1 * src_stride + indices[1]],
                    reinterpret_cast<const int*>(src_base)[sy1 * src_stride + indices[0]]
                );

                __m128i v_c11 = _mm_set_epi32(
                    reinterpret_cast<const int*>(src_base)[sy1 * src_stride + indices[3] + 1],
                    reinterpret_cast<const int*>(src_base)[sy1 * src_stride + indices[2] + 1],
                    reinterpret_cast<const int*>(src_base)[sy1 * src_stride + indices[1] + 1],
                    reinterpret_cast<const int*>(src_base)[sy1 * src_stride + indices[0] + 1]
                );

                __m128 r00, g00, b00, a00; unpack_colors_sse(v_c00, r00, g00, b00, a00);
                __m128 r10, g10, b10, a10; unpack_colors_sse(v_c10, r10, g10, b10, a10);
                __m128 r01, g01, b01, a01; unpack_colors_sse(v_c01, r01, g01, b01, a01);
                __m128 r11, g11, b11, a11; unpack_colors_sse(v_c11, r11, g11, b11, a11);

                __m128 res_r = _mm_fmadd_ps(r00, v_w1, _mm_fmadd_ps(r10, v_w2, _mm_fmadd_ps(r01, v_w3, _mm_mul_ps(r11, v_w4))));
                __m128 res_g = _mm_fmadd_ps(g00, v_w1, _mm_fmadd_ps(g10, v_w2, _mm_fmadd_ps(g01, v_w3, _mm_mul_ps(g11, v_w4))));
                __m128 res_b = _mm_fmadd_ps(b00, v_w1, _mm_fmadd_ps(b10, v_w2, _mm_fmadd_ps(b01, v_w3, _mm_mul_ps(b11, v_w4))));
                __m128 res_a = _mm_fmadd_ps(a00, v_w1, _mm_fmadd_ps(a10, v_w2, _mm_fmadd_ps(a01, v_w3, _mm_mul_ps(a11, v_w4))));

                __m128i v_final = pack_colors_sse(res_r, res_g, res_b, res_a);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(row_ptr + x), v_final);
            }

            // 标量处理剩余像素
            for (; x < newWidth; ++x) {
                const float srcX = x * invScaleX;
                const int srcX0 = static_cast<int>(srcX);
                const int srcX1 = std::min(srcX0 + 1, src.width - 1);
                const float dx = srcX - srcX0;

                const Color& c00 = src.at(srcX0, sy0);
                const Color& c10 = src.at(srcX1, sy0);
                const Color& c01 = src.at(srcX0, sy1);
                const Color& c11 = src.at(srcX1, sy1);

                float w1 = (1.0f - dx) * (1.0f - dy);
                float w2 = dx * (1.0f - dy);
                float w3 = (1.0f - dx) * dy;
                float w4 = dx * dy;

                Color ret;
                ret.r = (uint8_t)(c00.r * w1 + c10.r * w2 + c01.r * w3 + c11.r * w4);
                ret.g = (uint8_t)(c00.g * w1 + c10.g * w2 + c01.g * w3 + c11.g * w4);
                ret.b = (uint8_t)(c00.b * w1 + c10.b * w2 + c01.b * w3 + c11.b * w4);
                ret.a = (uint8_t)(c00.a * w1 + c10.a * w2 + c01.a * w3 + c11.a * w4);
                row_ptr[x] = ret;
            }
        }
        return result;
    }

    Buffer resized(const Buffer& src, int width, int height) {
        if (!src.isValid() || width <= 0 || height <= 0) return Buffer();

        if (src.width == width && src.height == height) {
            Buffer result;
            copy(result, src);
            return result;
        }

        Buffer result(width, height, 0);
        if (!result.isValid()) return Buffer();

        const float invScaleX = static_cast<float>(src.width) / width;
        const float invScaleY = static_cast<float>(src.height) / height;
        const float srcW_limit = static_cast<float>(src.width) - 1.05f;
        const int src_stride = src.width;

        const __m256 v_invScaleX = _mm256_set1_ps(invScaleX);
        const __m256 v_ones = _mm256_set1_ps(1.0f);
        const __m256 v_idx_step8 = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256 v_width_limit = _mm256_set1_ps(srcW_limit);

        const __m128 v_invScaleX_sse = _mm_set1_ps(invScaleX);
        const __m128 v_idx_step4 = _mm_setr_ps(0, 1, 2, 3);
        const __m128 v_width_limit_sse = _mm_set1_ps(srcW_limit);
        const __m128 v_ones_sse = _mm_set1_ps(1.0f);

        for (int y = 0; y < height; ++y) {
            const float srcY = y * invScaleY;
            const int sy0 = static_cast<int>(srcY);
            const int sy1 = std::min(sy0 + 1, src.height - 1);
            const float dy = srcY - sy0;

            const __m256 v_dy = _mm256_set1_ps(dy);
            const __m256 v_inv_dy = _mm256_sub_ps(v_ones, v_dy);

            const __m128 v_dy_sse = _mm_set1_ps(dy);
            const __m128 v_inv_dy_sse = _mm_sub_ps(v_ones_sse, v_dy_sse);

            Color* row_ptr = result.getRow(y);
            const Color* src_base = src.color;

            int x = 0;
            for (; x <= width - 8; x += 8) {
                __m256 v_x_base = _mm256_set1_ps((float)x);
                __m256 v_srcX = _mm256_mul_ps(_mm256_add_ps(v_x_base, v_idx_step8), v_invScaleX);
                v_srcX = _mm256_min_ps(v_srcX, v_width_limit);

                __m256 v_floor_x = _mm256_floor_ps(v_srcX);
                __m256i v_sx0 = _mm256_cvtps_epi32(v_floor_x);
                __m256 v_dx = _mm256_sub_ps(v_srcX, v_floor_x);
                __m256 v_inv_dx = _mm256_sub_ps(v_ones, v_dx);

                __m256 v_w1 = _mm256_mul_ps(v_inv_dx, v_inv_dy);
                __m256 v_w2 = _mm256_mul_ps(v_dx, v_inv_dy);
                __m256 v_w3 = _mm256_mul_ps(v_inv_dx, v_dy);
                __m256 v_w4 = _mm256_mul_ps(v_dx, v_dy);

                __m256i v_offset0 = _mm256_add_epi32(_mm256_set1_epi32(sy0 * src_stride), v_sx0);
                __m256i v_offset1 = _mm256_add_epi32(_mm256_set1_epi32(sy1 * src_stride), v_sx0);

                __m256i v_c00 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(src_base), v_offset0, 4);
                __m256i v_c10 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(src_base), _mm256_add_epi32(v_offset0, _mm256_set1_epi32(1)), 4);
                __m256i v_c01 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(src_base), v_offset1, 4);
                __m256i v_c11 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(src_base), _mm256_add_epi32(v_offset1, _mm256_set1_epi32(1)), 4);

                __m256 r00, g00, b00, a00; unpack_colors_avx2(v_c00, r00, g00, b00, a00);
                __m256 r10, g10, b10, a10; unpack_colors_avx2(v_c10, r10, g10, b10, a10);
                __m256 r01, g01, b01, a01; unpack_colors_avx2(v_c01, r01, g01, b01, a01);
                __m256 r11, g11, b11, a11; unpack_colors_avx2(v_c11, r11, g11, b11, a11);

                __m256 res_r = _mm256_fmadd_ps(r00, v_w1, _mm256_fmadd_ps(r10, v_w2, _mm256_fmadd_ps(r01, v_w3, _mm256_mul_ps(r11, v_w4))));
                __m256 res_g = _mm256_fmadd_ps(g00, v_w1, _mm256_fmadd_ps(g10, v_w2, _mm256_fmadd_ps(g01, v_w3, _mm256_mul_ps(g11, v_w4))));
                __m256 res_b = _mm256_fmadd_ps(b00, v_w1, _mm256_fmadd_ps(b10, v_w2, _mm256_fmadd_ps(b01, v_w3, _mm256_mul_ps(b11, v_w4))));
                __m256 res_a = _mm256_fmadd_ps(a00, v_w1, _mm256_fmadd_ps(a10, v_w2, _mm256_fmadd_ps(a01, v_w3, _mm256_mul_ps(a11, v_w4))));

                __m256i v_final = pack_colors_avx2(res_r, res_g, res_b, res_a);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(row_ptr + x), v_final);
            }

            for (; x <= width - 4; x += 4) {
                __m128 v_x_base = _mm_set1_ps((float)x);
                __m128 v_srcX = _mm_mul_ps(_mm_add_ps(v_x_base, v_idx_step4), v_invScaleX_sse);
                v_srcX = _mm_min_ps(v_srcX, v_width_limit_sse);

                __m128 v_floor_x = _mm_floor_ps(v_srcX);
                __m128i v_sx0 = _mm_cvtps_epi32(v_floor_x);
                __m128 v_dx = _mm_sub_ps(v_srcX, v_floor_x);
                __m128 v_inv_dx = _mm_sub_ps(v_ones_sse, v_dx);

                __m128 v_w1 = _mm_mul_ps(v_inv_dx, v_inv_dy_sse);
                __m128 v_w2 = _mm_mul_ps(v_dx, v_inv_dy_sse);
                __m128 v_w3 = _mm_mul_ps(v_inv_dx, v_dy_sse);
                __m128 v_w4 = _mm_mul_ps(v_dx, v_dy_sse);

                alignas(16) int indices[4];
                _mm_store_si128(reinterpret_cast<__m128i*>(indices), v_sx0);

                __m128i v_c00 = _mm_set_epi32(
                    reinterpret_cast<const int*>(src_base)[sy0 * src_stride + indices[3]],
                    reinterpret_cast<const int*>(src_base)[sy0 * src_stride + indices[2]],
                    reinterpret_cast<const int*>(src_base)[sy0 * src_stride + indices[1]],
                    reinterpret_cast<const int*>(src_base)[sy0 * src_stride + indices[0]]
                );

                __m128i v_c10 = _mm_set_epi32(
                    reinterpret_cast<const int*>(src_base)[sy0 * src_stride + indices[3] + 1],
                    reinterpret_cast<const int*>(src_base)[sy0 * src_stride + indices[2] + 1],
                    reinterpret_cast<const int*>(src_base)[sy0 * src_stride + indices[1] + 1],
                    reinterpret_cast<const int*>(src_base)[sy0 * src_stride + indices[0] + 1]
                );

                __m128i v_c01 = _mm_set_epi32(
                    reinterpret_cast<const int*>(src_base)[sy1 * src_stride + indices[3]],
                    reinterpret_cast<const int*>(src_base)[sy1 * src_stride + indices[2]],
                    reinterpret_cast<const int*>(src_base)[sy1 * src_stride + indices[1]],
                    reinterpret_cast<const int*>(src_base)[sy1 * src_stride + indices[0]]
                );

                __m128i v_c11 = _mm_set_epi32(
                    reinterpret_cast<const int*>(src_base)[sy1 * src_stride + indices[3] + 1],
                    reinterpret_cast<const int*>(src_base)[sy1 * src_stride + indices[2] + 1],
                    reinterpret_cast<const int*>(src_base)[sy1 * src_stride + indices[1] + 1],
                    reinterpret_cast<const int*>(src_base)[sy1 * src_stride + indices[0] + 1]
                );

                __m128 r00, g00, b00, a00; unpack_colors_sse(v_c00, r00, g00, b00, a00);
                __m128 r10, g10, b10, a10; unpack_colors_sse(v_c10, r10, g10, b10, a10);
                __m128 r01, g01, b01, a01; unpack_colors_sse(v_c01, r01, g01, b01, a01);
                __m128 r11, g11, b11, a11; unpack_colors_sse(v_c11, r11, g11, b11, a11);

                __m128 res_r = _mm_fmadd_ps(r00, v_w1, _mm_fmadd_ps(r10, v_w2, _mm_fmadd_ps(r01, v_w3, _mm_mul_ps(r11, v_w4))));
                __m128 res_g = _mm_fmadd_ps(g00, v_w1, _mm_fmadd_ps(g10, v_w2, _mm_fmadd_ps(g01, v_w3, _mm_mul_ps(g11, v_w4))));
                __m128 res_b = _mm_fmadd_ps(b00, v_w1, _mm_fmadd_ps(b10, v_w2, _mm_fmadd_ps(b01, v_w3, _mm_mul_ps(b11, v_w4))));
                __m128 res_a = _mm_fmadd_ps(a00, v_w1, _mm_fmadd_ps(a10, v_w2, _mm_fmadd_ps(a01, v_w3, _mm_mul_ps(a11, v_w4))));

                __m128i v_final = pack_colors_sse(res_r, res_g, res_b, res_a);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(row_ptr + x), v_final);
            }

            for (; x < width; ++x) {
                const float srcX = x * invScaleX;
                const int srcX0 = static_cast<int>(srcX);
                const int srcX1 = std::min(srcX0 + 1, src.width - 1);
                const float dx = srcX - srcX0;

                const Color& c00 = src.at(srcX0, sy0);
                const Color& c10 = src.at(srcX1, sy0);
                const Color& c01 = src.at(srcX0, sy1);
                const Color& c11 = src.at(srcX1, sy1);

                float w1 = (1.0f - dx) * (1.0f - dy);
                float w2 = dx * (1.0f - dy);
                float w3 = (1.0f - dx) * dy;
                float w4 = dx * dy;

                Color ret;
                ret.r = (uint8_t)(c00.r * w1 + c10.r * w2 + c01.r * w3 + c11.r * w4);
                ret.g = (uint8_t)(c00.g * w1 + c10.g * w2 + c01.g * w3 + c11.g * w4);
                ret.b = (uint8_t)(c00.b * w1 + c10.b * w2 + c01.b * w3 + c11.b * w4);
                ret.a = (uint8_t)(c00.a * w1 + c10.a * w2 + c01.a * w3 + c11.a * w4);
                row_ptr[x] = ret;
            }
        }
        return result;
    }

    Buffer scaled(const Buffer& src, float factor) {
        return scaled(src, factor, factor);
    }

    // ============================================================================
    // 旋转相关函数
    // ============================================================================

    Buffer rotated(const Buffer& src, float rotation) {
        if (!src.isValid()) return Buffer();

        float rad = rotation;
        while (rad < 0) rad += 2.0f * M_PI;
        while (rad >= 2.0f * M_PI) rad -= 2.0f * M_PI;

        const float cosA = std::cos(rad);
        const float sinA = std::sin(rad);

        const float absCosA = std::abs(cosA);
        const float absSinA = std::abs(sinA);
        const float newWidthF = src.width * absCosA + src.height * absSinA;
        const float newHeightF = src.width * absSinA + src.height * absCosA;

        constexpr float epsilon = 1e-5f;
        const int newWidth = static_cast<int>(std::ceil(newWidthF + epsilon));
        const int newHeight = static_cast<int>(std::ceil(newHeightF + epsilon));

        Buffer result(newWidth, newHeight, Color(0, 0, 0, 0));
        if (!result.isValid()) return result;

        const float srcCenterX = src.width * 0.5f;
        const float srcCenterY = src.height * 0.5f;
        const float dstCenterX = newWidth * 0.5f;
        const float dstCenterY = newHeight * 0.5f;

        const float invCosA = cosA;
        const float invSinA = sinA;

        const __m256 v_invCosA = _mm256_set1_ps(invCosA);
        const __m256 v_invSinA = _mm256_set1_ps(invSinA);
        const __m256 v_srcCenterX = _mm256_set1_ps(srcCenterX);
        const __m256 v_srcCenterY = _mm256_set1_ps(srcCenterY);
        const __m256 v_dstCenterX = _mm256_set1_ps(dstCenterX);
        const __m256 v_ones = _mm256_set1_ps(1.0f);
        const __m256 v_zeros = _mm256_setzero_ps();
        const __m256 v_255 = _mm256_set1_ps(255.0f);
        const __m256 v_inv255 = _mm256_set1_ps(1.0f / 255.0f);

        const __m256 v_srcW_limit = _mm256_set1_ps(static_cast<float>(src.width) - 0.001f);
        const __m256 v_srcH_limit = _mm256_set1_ps(static_cast<float>(src.height) - 0.001f);
        const __m256 v_neg_one = _mm256_set1_ps(-1.0f);

        const __m256i v_max_w_idx = _mm256_set1_epi32(src.width - 1);
        const __m256i v_max_h_idx = _mm256_set1_epi32(src.height - 1);
        const __m256i v_zero_idx = _mm256_setzero_si256();
        const __m256i v_transparent_color = _mm256_set1_epi32(0x00FFFFFF);

        const __m128 v_invCosA_sse = _mm_set1_ps(invCosA);
        const __m128 v_invSinA_sse = _mm_set1_ps(invSinA);
        const __m128 v_srcCenterX_sse = _mm_set1_ps(srcCenterX);
        const __m128 v_srcCenterY_sse = _mm_set1_ps(srcCenterY);
        const __m128 v_dstCenterX_sse = _mm_set1_ps(dstCenterX);
        const __m128 v_ones_sse = _mm_set1_ps(1.0f);
        const __m128 v_zeros_sse = _mm_setzero_ps();
        const __m128 v_255_sse = _mm_set1_ps(255.0f);
        const __m128 v_inv255_sse = _mm_set1_ps(1.0f / 255.0f);
        const __m128 v_srcW_limit_sse = _mm_set1_ps(static_cast<float>(src.width) - 0.001f);
        const __m128 v_srcH_limit_sse = _mm_set1_ps(static_cast<float>(src.height) - 0.001f);
        const __m128 v_neg_one_sse = _mm_set1_ps(-1.0f);
        const __m128i v_max_w_idx_sse = _mm_set1_epi32(src.width - 1);
        const __m128i v_max_h_idx_sse = _mm_set1_epi32(src.height - 1);
        const __m128i v_zero_idx_sse = _mm_setzero_si128();
        const __m128i v_transparent_color_sse = _mm_set1_epi32(0x00FFFFFF);

        const int src_stride = src.width;

        auto safe_gather_avx2 = [&](__m256i ix, __m256i iy) -> __m256i {
            __m256i mask_x = _mm256_and_si256(
                _mm256_cmpgt_epi32(ix, _mm256_set1_epi32(-1)),
                _mm256_cmpgt_epi32(_mm256_set1_epi32(src.width), ix)
            );
            __m256i mask_y = _mm256_and_si256(
                _mm256_cmpgt_epi32(iy, _mm256_set1_epi32(-1)),
                _mm256_cmpgt_epi32(_mm256_set1_epi32(src.height), iy)
            );
            __m256i valid_mask = _mm256_and_si256(mask_x, mask_y);

            __m256i safe_x = _mm256_max_epi32(v_zero_idx, _mm256_min_epi32(ix, v_max_w_idx));
            __m256i safe_y = _mm256_max_epi32(v_zero_idx, _mm256_min_epi32(iy, v_max_h_idx));

            __m256i offset = _mm256_add_epi32(
                _mm256_mullo_epi32(safe_y, _mm256_set1_epi32(src_stride)),
                safe_x
            );
            const int* sptr = reinterpret_cast<const int*>(src.color);
            __m256i colors = _mm256_i32gather_epi32(sptr, offset, 4);

            return _mm256_blendv_epi8(v_transparent_color, colors, valid_mask);
        };

        auto safe_gather_sse = [&](__m128i ix, __m128i iy) -> __m128i {
            __m128i mask_x = _mm_and_si128(
                _mm_cmpgt_epi32(ix, _mm_set1_epi32(-1)),
                _mm_cmpgt_epi32(_mm_set1_epi32(src.width), ix)
            );
            __m128i mask_y = _mm_and_si128(
                _mm_cmpgt_epi32(iy, _mm_set1_epi32(-1)),
                _mm_cmpgt_epi32(_mm_set1_epi32(src.height), iy)
            );
            __m128i valid_mask = _mm_and_si128(mask_x, mask_y);

            __m128i safe_x = _mm_max_epi32(v_zero_idx_sse, _mm_min_epi32(ix, v_max_w_idx_sse));
            __m128i safe_y = _mm_max_epi32(v_zero_idx_sse, _mm_min_epi32(iy, v_max_h_idx_sse));

            alignas(16) int idx_x[4], idx_y[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(idx_x), safe_x);
            _mm_store_si128(reinterpret_cast<__m128i*>(idx_y), safe_y);

            const int* src_data = reinterpret_cast<const int*>(src.color);
            int c0 = src_data[idx_y[0] * src_stride + idx_x[0]];
            int c1 = src_data[idx_y[1] * src_stride + idx_x[1]];
            int c2 = src_data[idx_y[2] * src_stride + idx_x[2]];
            int c3 = src_data[idx_y[3] * src_stride + idx_x[3]];

            __m128i colors = _mm_set_epi32(c3, c2, c1, c0);
            return _mm_blendv_epi8(v_transparent_color_sse, colors, valid_mask);
        };

        for (int y = 0; y < newHeight; ++y) {
            Color* destRow = result.getRow(y);
            const __m256 v_dy = _mm256_sub_ps(_mm256_set1_ps(static_cast<float>(y)), v_dstCenterX);

            const __m256 v_baseX = _mm256_fmadd_ps(v_dy, v_invSinA, v_srcCenterX);
            const __m256 v_baseY = _mm256_fnmadd_ps(v_dy, v_invCosA, v_srcCenterY);

            int x = 0;

            for (; x <= newWidth - 8; x += 8) {
                __m256 v_idx = _mm256_add_ps(_mm256_set1_ps(static_cast<float>(x)),
                    _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7));
                __m256 v_dx = _mm256_sub_ps(v_idx, v_dstCenterX);

                __m256 v_srcX = _mm256_fmadd_ps(v_dx, v_invCosA, v_baseX);
                __m256 v_srcY = _mm256_fmadd_ps(v_dx, v_invSinA, v_baseY);

                __m256 v_mask = _mm256_and_ps(
                    _mm256_cmp_ps(v_srcX, v_neg_one, _CMP_GE_OQ),
                    _mm256_cmp_ps(v_srcX, v_srcW_limit, _CMP_LT_OQ)
                );
                v_mask = _mm256_and_ps(v_mask,
                    _mm256_cmp_ps(v_srcY, v_neg_one, _CMP_GE_OQ));
                v_mask = _mm256_and_ps(v_mask,
                    _mm256_cmp_ps(v_srcY, v_srcH_limit, _CMP_LT_OQ));

                if (_mm256_movemask_ps(v_mask) == 0) continue;

                __m256 v_floorX = _mm256_floor_ps(v_srcX);
                __m256 v_floorY = _mm256_floor_ps(v_srcY);
                __m256i v_idx_x = _mm256_cvtps_epi32(v_floorX);
                __m256i v_idx_y = _mm256_cvtps_epi32(v_floorY);

                __m256 v_fx = _mm256_sub_ps(v_srcX, v_floorX);
                __m256 v_fy = _mm256_sub_ps(v_srcY, v_floorY);
                __m256 v_fx_inv = _mm256_sub_ps(v_ones, v_fx);
                __m256 v_fy_inv = _mm256_sub_ps(v_ones, v_fy);

                __m256 v_w00 = _mm256_mul_ps(v_fx_inv, v_fy_inv);
                __m256 v_w10 = _mm256_mul_ps(v_fx, v_fy_inv);
                __m256 v_w01 = _mm256_mul_ps(v_fx_inv, v_fy);
                __m256 v_w11 = _mm256_mul_ps(v_fx, v_fy);

                __m256i c00 = safe_gather_avx2(v_idx_x, v_idx_y);
                __m256i c10 = safe_gather_avx2(_mm256_add_epi32(v_idx_x, _mm256_set1_epi32(1)), v_idx_y);
                __m256i c01 = safe_gather_avx2(v_idx_x, _mm256_add_epi32(v_idx_y, _mm256_set1_epi32(1)));
                __m256i c11 = safe_gather_avx2(_mm256_add_epi32(v_idx_x, _mm256_set1_epi32(1)),
                    _mm256_add_epi32(v_idx_y, _mm256_set1_epi32(1)));

                __m256 r00, g00, b00, a00; unpack_colors_avx2(c00, r00, g00, b00, a00);
                __m256 r10, g10, b10, a10; unpack_colors_avx2(c10, r10, g10, b10, a10);
                __m256 r01, g01, b01, a01; unpack_colors_avx2(c01, r01, g01, b01, a01);
                __m256 r11, g11, b11, a11; unpack_colors_avx2(c11, r11, g11, b11, a11);

                __m256 r = _mm256_add_ps(
                    _mm256_add_ps(_mm256_mul_ps(r00, v_w00), _mm256_mul_ps(r10, v_w10)),
                    _mm256_add_ps(_mm256_mul_ps(r01, v_w01), _mm256_mul_ps(r11, v_w11))
                );
                __m256 g = _mm256_add_ps(
                    _mm256_add_ps(_mm256_mul_ps(g00, v_w00), _mm256_mul_ps(g10, v_w10)),
                    _mm256_add_ps(_mm256_mul_ps(g01, v_w01), _mm256_mul_ps(g11, v_w11))
                );
                __m256 b = _mm256_add_ps(
                    _mm256_add_ps(_mm256_mul_ps(b00, v_w00), _mm256_mul_ps(b10, v_w10)),
                    _mm256_add_ps(_mm256_mul_ps(b01, v_w01), _mm256_mul_ps(b11, v_w11))
                );
                __m256 a = _mm256_add_ps(
                    _mm256_add_ps(_mm256_mul_ps(a00, v_w00), _mm256_mul_ps(a10, v_w10)),
                    _mm256_add_ps(_mm256_mul_ps(a01, v_w01), _mm256_mul_ps(a11, v_w11))
                );

                __m256i v_dst_raw = _mm256_loadu_si256(reinterpret_cast<__m256i*>(destRow + x));
                __m256 d_r, d_g, d_b, d_a;
                unpack_colors_avx2(v_dst_raw, d_r, d_g, d_b, d_a);

                __m256 v_sa_norm = _mm256_mul_ps(a, v_inv255);
                __m256 v_da_norm = _mm256_mul_ps(d_a, v_inv255);
                __m256 v_out_a = _mm256_add_ps(v_sa_norm,
                    _mm256_mul_ps(v_da_norm, _mm256_sub_ps(v_ones, v_sa_norm)));

                __m256 v_sa_factor = _mm256_div_ps(v_sa_norm,
                    _mm256_max_ps(v_out_a, _mm256_set1_ps(0.0001f)));
                __m256 v_da_factor = _mm256_sub_ps(v_ones, v_sa_factor);

                __m256 out_r = _mm256_add_ps(_mm256_mul_ps(r, v_sa_factor),
                    _mm256_mul_ps(d_r, v_da_factor));
                __m256 out_g = _mm256_add_ps(_mm256_mul_ps(g, v_sa_factor),
                    _mm256_mul_ps(d_g, v_da_factor));
                __m256 out_b = _mm256_add_ps(_mm256_mul_ps(b, v_sa_factor),
                    _mm256_mul_ps(d_b, v_da_factor));
                __m256 out_a = _mm256_mul_ps(v_out_a, v_255);

                __m256i v_result = pack_colors_avx2(out_r, out_g, out_b, out_a);
                __m256i v_mask_i = _mm256_castps_si256(v_mask);
                __m256i v_final = _mm256_blendv_epi8(v_dst_raw, v_result, v_mask_i);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(destRow + x), v_final);
            }

            const __m128 v_dy_sse = _mm_set1_ps(static_cast<float>(y) - dstCenterY);
            const __m128 v_baseX_sse = _mm_fmadd_ps(v_dy_sse, v_invSinA_sse, v_srcCenterX_sse);
            const __m128 v_baseY_sse = _mm_fnmadd_ps(v_dy_sse, v_invCosA_sse, v_srcCenterY_sse);

            for (; x <= newWidth - 4; x += 4) {
                __m128 v_idx = _mm_add_ps(_mm_set1_ps(static_cast<float>(x)),
                    _mm_setr_ps(0, 1, 2, 3));
                __m128 v_dx = _mm_sub_ps(v_idx, v_dstCenterX_sse);

                __m128 v_srcX = _mm_fmadd_ps(v_dx, v_invCosA_sse, v_baseX_sse);
                __m128 v_srcY = _mm_fmadd_ps(v_dx, v_invSinA_sse, v_baseY_sse);

                __m128 v_mask = _mm_and_ps(
                    _mm_cmpge_ps(v_srcX, v_neg_one_sse),
                    _mm_cmplt_ps(v_srcX, v_srcW_limit_sse)
                );
                v_mask = _mm_and_ps(v_mask,
                    _mm_cmpge_ps(v_srcY, v_neg_one_sse));
                v_mask = _mm_and_ps(v_mask,
                    _mm_cmplt_ps(v_srcY, v_srcH_limit_sse));

                if (_mm_movemask_ps(v_mask) == 0) continue;

                __m128 v_floorX = _mm_floor_ps(v_srcX);
                __m128 v_floorY = _mm_floor_ps(v_srcY);
                __m128i v_idx_x = _mm_cvtps_epi32(v_floorX);
                __m128i v_idx_y = _mm_cvtps_epi32(v_floorY);

                __m128 v_fx = _mm_sub_ps(v_srcX, v_floorX);
                __m128 v_fy = _mm_sub_ps(v_srcY, v_floorY);
                __m128 v_fx_inv = _mm_sub_ps(v_ones_sse, v_fx);
                __m128 v_fy_inv = _mm_sub_ps(v_ones_sse, v_fy);

                __m128 v_w00 = _mm_mul_ps(v_fx_inv, v_fy_inv);
                __m128 v_w10 = _mm_mul_ps(v_fx, v_fy_inv);
                __m128 v_w01 = _mm_mul_ps(v_fx_inv, v_fy);
                __m128 v_w11 = _mm_mul_ps(v_fx, v_fy);

                __m128i c00 = safe_gather_sse(v_idx_x, v_idx_y);
                __m128i c10 = safe_gather_sse(_mm_add_epi32(v_idx_x, _mm_set1_epi32(1)), v_idx_y);
                __m128i c01 = safe_gather_sse(v_idx_x, _mm_add_epi32(v_idx_y, _mm_set1_epi32(1)));
                __m128i c11 = safe_gather_sse(_mm_add_epi32(v_idx_x, _mm_set1_epi32(1)),
                    _mm_add_epi32(v_idx_y, _mm_set1_epi32(1)));

                __m128 r00, g00, b00, a00; unpack_colors_sse(c00, r00, g00, b00, a00);
                __m128 r10, g10, b10, a10; unpack_colors_sse(c10, r10, g10, b10, a10);
                __m128 r01, g01, b01, a01; unpack_colors_sse(c01, r01, g01, b01, a01);
                __m128 r11, g11, b11, a11; unpack_colors_sse(c11, r11, g11, b11, a11);

                __m128 r = _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(r00, v_w00), _mm_mul_ps(r10, v_w10)),
                    _mm_add_ps(_mm_mul_ps(r01, v_w01), _mm_mul_ps(r11, v_w11))
                );
                __m128 g = _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(g00, v_w00), _mm_mul_ps(g10, v_w10)),
                    _mm_add_ps(_mm_mul_ps(g01, v_w01), _mm_mul_ps(g11, v_w11))
                );
                __m128 b = _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(b00, v_w00), _mm_mul_ps(b10, v_w10)),
                    _mm_add_ps(_mm_mul_ps(b01, v_w01), _mm_mul_ps(b11, v_w11))
                );
                __m128 a = _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(a00, v_w00), _mm_mul_ps(a10, v_w10)),
                    _mm_add_ps(_mm_mul_ps(a01, v_w01), _mm_mul_ps(a11, v_w11))
                );

                __m128i v_dst_raw = _mm_loadu_si128(reinterpret_cast<__m128i*>(destRow + x));
                __m128 d_r, d_g, d_b, d_a;
                unpack_colors_sse(v_dst_raw, d_r, d_g, d_b, d_a);

                __m128 v_sa_norm = _mm_mul_ps(a, v_inv255_sse);
                __m128 v_da_norm = _mm_mul_ps(d_a, v_inv255_sse);
                __m128 v_out_a = _mm_add_ps(v_sa_norm,
                    _mm_mul_ps(v_da_norm, _mm_sub_ps(v_ones_sse, v_sa_norm)));

                __m128 v_sa_factor = _mm_div_ps(v_sa_norm,
                    _mm_max_ps(v_out_a, _mm_set1_ps(0.0001f)));
                __m128 v_da_factor = _mm_sub_ps(v_ones_sse, v_sa_factor);

                __m128 out_r = _mm_add_ps(_mm_mul_ps(r, v_sa_factor),
                    _mm_mul_ps(d_r, v_da_factor));
                __m128 out_g = _mm_add_ps(_mm_mul_ps(g, v_sa_factor),
                    _mm_mul_ps(d_g, v_da_factor));
                __m128 out_b = _mm_add_ps(_mm_mul_ps(b, v_sa_factor),
                    _mm_mul_ps(d_b, v_da_factor));
                __m128 out_a_scaled = _mm_mul_ps(v_out_a, v_255_sse);

                __m128i v_result = pack_colors_sse(out_r, out_g, out_b, out_a_scaled);
                __m128i v_mask_i = _mm_castps_si128(v_mask);
                __m128i v_final = _mm_blendv_epi8(v_dst_raw, v_result, v_mask_i);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(destRow + x), v_final);
            }

            for (; x < newWidth; ++x) {
                const float dx = static_cast<float>(x) - dstCenterX;
                const float dy = static_cast<float>(y) - dstCenterY;

                const float srcX = dx * invCosA + dy * invSinA + srcCenterX;
                const float srcY = -dx * invSinA + dy * invCosA + srcCenterY;

                if (srcX >= -1.0f && srcX < src.width &&
                    srcY >= -1.0f && srcY < src.height) {

                    const int x0 = static_cast<int>(std::floor(srcX));
                    const int y0 = static_cast<int>(std::floor(srcY));
                    const float fx = srcX - x0;
                    const float fy = srcY - y0;

                    auto getPixelSafe = [&](int px, int py) -> Color {
                        if (px >= 0 && px < src.width && py >= 0 && py < src.height) {
                            return src.at(px, py);
                        }
                        return Color(255, 255, 255, 0);
                        };

                    Color c00 = getPixelSafe(x0, y0);
                    Color c10 = getPixelSafe(x0 + 1, y0);
                    Color c01 = getPixelSafe(x0, y0 + 1);
                    Color c11 = getPixelSafe(x0 + 1, y0 + 1);

                    const float w00 = (1.0f - fx) * (1.0f - fy);
                    const float w10 = fx * (1.0f - fy);
                    const float w01 = (1.0f - fx) * fy;
                    const float w11 = fx * fy;

                    float r = c00.r * w00 + c10.r * w10 + c01.r * w01 + c11.r * w11;
                    float g = c00.g * w00 + c10.g * w10 + c01.g * w01 + c11.g * w11;
                    float b = c00.b * w00 + c10.b * w10 + c01.b * w01 + c11.b * w11;
                    float a = c00.a * w00 + c10.a * w10 + c01.a * w01 + c11.a * w11;

                    Color& dst = destRow[x];
                    const float srcAlpha = a / 255.0f;
                    const float dstAlpha = dst.a / 255.0f;
                    const float outAlpha = srcAlpha + dstAlpha * (1.0f - srcAlpha);

                    if (outAlpha > 0.0001f) {
                        const float srcFactor = srcAlpha / outAlpha;
                        const float dstFactor = 1.0f - srcFactor;

                        dst.r = static_cast<uint8_t>(std::min(255.0f, r * srcFactor + dst.r * dstFactor));
                        dst.g = static_cast<uint8_t>(std::min(255.0f, g * srcFactor + dst.g * dstFactor));
                        dst.b = static_cast<uint8_t>(std::min(255.0f, b * srcFactor + dst.b * dstFactor));
                    }
                    dst.a = static_cast<uint8_t>(std::min(255.0f, outAlpha * 255.0f));
                }
            }
        }

        return result;
    }

    inline void calculateRotatedSize(float halfW, float halfH, float cosA, float sinA, int& outWidth, int& outHeight) {
        float x1 = halfW * cosA - halfH * sinA;
        float x2 = -halfW * cosA - halfH * sinA;
        float x3 = halfW * cosA + halfH * sinA;
        float x4 = -halfW * cosA + halfH * sinA;

        float y1 = halfW * sinA + halfH * cosA;
        float y2 = -halfW * sinA + halfH * cosA;
        float y3 = halfW * sinA - halfH * cosA;
        float y4 = -halfW * sinA - halfH * cosA;

        float minX = std::min({ x1, x2, x3, x4 });
        float maxX = std::max({ x1, x2, x3, x4 });
        float minY = std::min({ y1, y2, y3, y4 });
        float maxY = std::max({ y1, y2, y3, y4 });

        outWidth = static_cast<int>(std::ceil(maxX - minX));
        outHeight = static_cast<int>(std::ceil(maxY - minY));
    }

    inline Buffer transformedImpl(const Buffer& src, float scaleX, float scaleY, float rotation,
        int newWidth, int newHeight) {
        Buffer result(newWidth, newHeight, 0);
        if (!result.isValid()) return Buffer();

        const float centerX = newWidth * 0.5f;
        const float centerY = newHeight * 0.5f;

        if (scaleX == scaleY) {
            drawTransformed(result, src, centerX, centerY, scaleX, rotation);
        }
        else {
            drawTransformed(result, src, centerX, centerY, scaleX, scaleY, rotation);
        }
        return result;
    }

    Buffer transformed(const Buffer& src, float scale, float rotation) {
        if (!src.isValid()) return Buffer();

        const float cosA = std::cos(rotation);
        const float sinA = std::sin(rotation);
        const float halfW = src.width * scale * 0.5f;
        const float halfH = src.height * scale * 0.5f;

        int newWidth, newHeight;
        calculateRotatedSize(halfW, halfH, cosA, sinA, newWidth, newHeight);

        return transformedImpl(src, scale, scale, rotation, newWidth, newHeight);
    }

    Buffer transformed(const Buffer& src, float scaleX, float scaleY, float rotation) {
        if (!src.isValid()) return Buffer();

        const float cosA = std::cos(rotation);
        const float sinA = std::sin(rotation);
        const float halfW = src.width * scaleX * 0.5f;
        const float halfH = src.height * scaleY * 0.5f;

        int newWidth, newHeight;
        calculateRotatedSize(halfW, halfH, cosA, sinA, newWidth, newHeight);

        return transformedImpl(src, scaleX, scaleY, rotation, newWidth, newHeight);
    }

    // ============================================================================
    // 绘制函数
    // ============================================================================

    void drawScaled(Buffer& dest, const Buffer& src, float centerX, float centerY, float scaleX, float scaleY) {
        if (!dest.isValid() || !src.isValid() || scaleX <= 0.0f || scaleY <= 0.0f) return;

        const float halfW = src.width * scaleX * 0.5f;
        const float halfH = src.height * scaleY * 0.5f;

        int startX = std::max(0, static_cast<int>(centerX - halfW));
        int startY = std::max(0, static_cast<int>(centerY - halfH));
        int endX = std::min(dest.width, static_cast<int>(centerX + halfW) + 1);
        int endY = std::min(dest.height, static_cast<int>(centerY + halfH) + 1);

        if (startX >= endX || startY >= endY) return;

        const float invScaleX = 1.0f / scaleX;
        const float invScaleY = 1.0f / scaleY;
        const float srcCenterX = src.width * 0.5f;
        const float srcCenterY = src.height * 0.5f;
        const int src_stride = src.width;

        const __m256 v_invScaleX = _mm256_set1_ps(invScaleX);
        const __m256 v_idx_step8 = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256 v_srcW = _mm256_set1_ps((float)src.width - 1.05f);
        const __m256 v_ones = _mm256_set1_ps(1.0f);
        const __m256 v_255 = _mm256_set1_ps(255.0f);
        const __m256 v_inv255 = _mm256_set1_ps(1.0f / 255.0f);

        const __m128 v_invScaleX_sse = _mm_set1_ps(invScaleX);
        const __m128 v_idx_step4 = _mm_setr_ps(0, 1, 2, 3);
        const __m128 v_srcW_sse = _mm_set1_ps((float)src.width - 1.05f);
        const __m128 v_ones_sse = _mm_set1_ps(1.0f);
        const __m128 v_255_sse = _mm_set1_ps(255.0f);
        const __m128 v_inv255_sse = _mm_set1_ps(1.0f / 255.0f);

        for (int y = startY; y < endY; ++y) {
            float srcY = (y - centerY) * invScaleY + srcCenterY;

            if (srcY < 0.0f) srcY = 0.0f;
            if (srcY > src.height - 1.05f) srcY = src.height - 1.05f;

            int sy0 = static_cast<int>(srcY);
            int sy1 = (sy0 + 1 < src.height) ? sy0 + 1 : sy0;

            float dy = srcY - sy0;

            __m256 v_dy = _mm256_set1_ps(dy);
            __m256 v_inv_dy = _mm256_sub_ps(v_ones, v_dy);

            __m128 v_dy_sse = _mm_set1_ps(dy);
            __m128 v_inv_dy_sse = _mm_sub_ps(v_ones_sse, v_dy_sse);

            const Color* row0_ptr = src.color + sy0 * src_stride;
            const Color* row1_ptr = src.color + sy1 * src_stride;

            Color* destRow = dest.getRow(y);

            float startSrcX = (startX - centerX) * invScaleX + srcCenterX;

            int x = startX;

            for (; x <= endX - 8; x += 8) {
                __m256 v_sx = _mm256_fmadd_ps(v_idx_step8, v_invScaleX, _mm256_set1_ps(startSrcX));
                startSrcX += invScaleX * 8.0f;

                __m256 v_mask = _mm256_and_ps(_mm256_cmp_ps(v_sx, v_ones, _CMP_GE_OQ), _mm256_cmp_ps(v_sx, v_srcW, _CMP_LE_OQ));

                if (_mm256_movemask_ps(v_mask) == 0) continue;

                __m256 v_safe_sx = _mm256_blendv_ps(v_ones, v_sx, v_mask);
                __m256 v_flr_x = _mm256_floor_ps(v_safe_sx);
                __m256i v_idx_x = _mm256_cvtps_epi32(v_flr_x);

                __m256 v_dx = _mm256_sub_ps(v_safe_sx, v_flr_x);
                __m256 v_inv_dx = _mm256_sub_ps(v_ones, v_dx);

                __m256i c00_i = _mm256_i32gather_epi32(reinterpret_cast<const int*>(row0_ptr), v_idx_x, 4);
                __m256i c10_i = _mm256_i32gather_epi32(reinterpret_cast<const int*>(row0_ptr), _mm256_add_epi32(v_idx_x, _mm256_set1_epi32(1)), 4);
                __m256i c01_i = _mm256_i32gather_epi32(reinterpret_cast<const int*>(row1_ptr), v_idx_x, 4);
                __m256i c11_i = _mm256_i32gather_epi32(reinterpret_cast<const int*>(row1_ptr), _mm256_add_epi32(v_idx_x, _mm256_set1_epi32(1)), 4);

                __m256 s_r, s_g, s_b, s_a;
                {
                    __m256 r00, g00, b00, a00; unpack_colors_avx2(c00_i, r00, g00, b00, a00);
                    __m256 r10, g10, b10, a10; unpack_colors_avx2(c10_i, r10, g10, b10, a10);
                    __m256 r01, g01, b01, a01; unpack_colors_avx2(c01_i, r01, g01, b01, a01);
                    __m256 r11, g11, b11, a11; unpack_colors_avx2(c11_i, r11, g11, b11, a11);

                    __m256 top_r = _mm256_fmadd_ps(r10, v_dx, _mm256_mul_ps(r00, v_inv_dx));
                    __m256 top_g = _mm256_fmadd_ps(g10, v_dx, _mm256_mul_ps(g00, v_inv_dx));
                    __m256 top_b = _mm256_fmadd_ps(b10, v_dx, _mm256_mul_ps(b00, v_inv_dx));
                    __m256 top_a = _mm256_fmadd_ps(a10, v_dx, _mm256_mul_ps(a00, v_inv_dx));

                    __m256 bot_r = _mm256_fmadd_ps(r11, v_dx, _mm256_mul_ps(r01, v_inv_dx));
                    __m256 bot_g = _mm256_fmadd_ps(g11, v_dx, _mm256_mul_ps(g01, v_inv_dx));
                    __m256 bot_b = _mm256_fmadd_ps(b11, v_dx, _mm256_mul_ps(b01, v_inv_dx));
                    __m256 bot_a = _mm256_fmadd_ps(a11, v_dx, _mm256_mul_ps(a01, v_inv_dx));

                    s_r = _mm256_fmadd_ps(bot_r, v_dy, _mm256_mul_ps(top_r, v_inv_dy));
                    s_g = _mm256_fmadd_ps(bot_g, v_dy, _mm256_mul_ps(top_g, v_inv_dy));
                    s_b = _mm256_fmadd_ps(bot_b, v_dy, _mm256_mul_ps(top_b, v_inv_dy));
                    s_a = _mm256_fmadd_ps(bot_a, v_dy, _mm256_mul_ps(top_a, v_inv_dy));
                }

                __m256i v_dst_raw = _mm256_loadu_si256(reinterpret_cast<__m256i*>(destRow + x));
                __m256 d_r, d_g, d_b, d_a;
                unpack_colors_avx2(v_dst_raw, d_r, d_g, d_b, d_a);

                __m256 v_sa_norm = _mm256_mul_ps(s_a, v_inv255);
                __m256 v_inv_sa = _mm256_sub_ps(v_ones, v_sa_norm);

                __m256 out_r = _mm256_fmadd_ps(s_r, v_sa_norm, _mm256_mul_ps(d_r, v_inv_sa));
                __m256 out_g = _mm256_fmadd_ps(s_g, v_sa_norm, _mm256_mul_ps(d_g, v_inv_sa));
                __m256 out_b = _mm256_fmadd_ps(s_b, v_sa_norm, _mm256_mul_ps(d_b, v_inv_sa));
                __m256 out_a = _mm256_fmadd_ps(s_a, v_ones, _mm256_mul_ps(d_a, v_inv_sa));

                out_r = _mm256_min_ps(out_r, v_255);
                out_g = _mm256_min_ps(out_g, v_255);
                out_b = _mm256_min_ps(out_b, v_255);
                out_a = _mm256_min_ps(out_a, v_255);

                __m256i v_result = pack_colors_avx2(out_r, out_g, out_b, out_a);
                __m256i v_mask_i = _mm256_castps_si256(v_mask);
                __m256i v_final = _mm256_blendv_epi8(v_dst_raw, v_result, v_mask_i);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(destRow + x), v_final);
            }

            for (; x <= endX - 4; x += 4) {
                __m128 v_sx = _mm_fmadd_ps(v_idx_step4, v_invScaleX_sse, _mm_set1_ps(startSrcX));
                startSrcX += invScaleX * 4.0f;

                __m128 v_mask = _mm_and_ps(_mm_cmpge_ps(v_sx, v_ones_sse), _mm_cmple_ps(v_sx, v_srcW_sse));

                if (_mm_movemask_ps(v_mask) == 0) continue;

                __m128 v_safe_sx = _mm_blendv_ps(v_ones_sse, v_sx, v_mask);
                __m128 v_flr_x = _mm_floor_ps(v_safe_sx);
                __m128i v_idx_x = _mm_cvtps_epi32(v_flr_x);

                __m128 v_dx = _mm_sub_ps(v_safe_sx, v_flr_x);
                __m128 v_inv_dx = _mm_sub_ps(v_ones_sse, v_dx);

                alignas(16) int indices[4];
                _mm_store_si128(reinterpret_cast<__m128i*>(indices), v_idx_x);

                __m128i c00_i = _mm_set_epi32(
                    reinterpret_cast<const int*>(row0_ptr)[indices[3]],
                    reinterpret_cast<const int*>(row0_ptr)[indices[2]],
                    reinterpret_cast<const int*>(row0_ptr)[indices[1]],
                    reinterpret_cast<const int*>(row0_ptr)[indices[0]]
                );

                __m128i c10_i = _mm_set_epi32(
                    reinterpret_cast<const int*>(row0_ptr)[indices[3] + 1],
                    reinterpret_cast<const int*>(row0_ptr)[indices[2] + 1],
                    reinterpret_cast<const int*>(row0_ptr)[indices[1] + 1],
                    reinterpret_cast<const int*>(row0_ptr)[indices[0] + 1]
                );

                __m128i c01_i = _mm_set_epi32(
                    reinterpret_cast<const int*>(row1_ptr)[indices[3]],
                    reinterpret_cast<const int*>(row1_ptr)[indices[2]],
                    reinterpret_cast<const int*>(row1_ptr)[indices[1]],
                    reinterpret_cast<const int*>(row1_ptr)[indices[0]]
                );

                __m128i c11_i = _mm_set_epi32(
                    reinterpret_cast<const int*>(row1_ptr)[indices[3] + 1],
                    reinterpret_cast<const int*>(row1_ptr)[indices[2] + 1],
                    reinterpret_cast<const int*>(row1_ptr)[indices[1] + 1],
                    reinterpret_cast<const int*>(row1_ptr)[indices[0] + 1]
                );

                __m128 s_r, s_g, s_b, s_a;
                {
                    __m128 r00, g00, b00, a00; unpack_colors_sse(c00_i, r00, g00, b00, a00);
                    __m128 r10, g10, b10, a10; unpack_colors_sse(c10_i, r10, g10, b10, a10);
                    __m128 r01, g01, b01, a01; unpack_colors_sse(c01_i, r01, g01, b01, a01);
                    __m128 r11, g11, b11, a11; unpack_colors_sse(c11_i, r11, g11, b11, a11);

                    __m128 top_r = _mm_fmadd_ps(r10, v_dx, _mm_mul_ps(r00, v_inv_dx));
                    __m128 top_g = _mm_fmadd_ps(g10, v_dx, _mm_mul_ps(g00, v_inv_dx));
                    __m128 top_b = _mm_fmadd_ps(b10, v_dx, _mm_mul_ps(b00, v_inv_dx));
                    __m128 top_a = _mm_fmadd_ps(a10, v_dx, _mm_mul_ps(a00, v_inv_dx));

                    __m128 bot_r = _mm_fmadd_ps(r11, v_dx, _mm_mul_ps(r01, v_inv_dx));
                    __m128 bot_g = _mm_fmadd_ps(g11, v_dx, _mm_mul_ps(g01, v_inv_dx));
                    __m128 bot_b = _mm_fmadd_ps(b11, v_dx, _mm_mul_ps(b01, v_inv_dx));
                    __m128 bot_a = _mm_fmadd_ps(a11, v_dx, _mm_mul_ps(a01, v_inv_dx));

                    s_r = _mm_fmadd_ps(bot_r, v_dy_sse, _mm_mul_ps(top_r, v_inv_dy_sse));
                    s_g = _mm_fmadd_ps(bot_g, v_dy_sse, _mm_mul_ps(top_g, v_inv_dy_sse));
                    s_b = _mm_fmadd_ps(bot_b, v_dy_sse, _mm_mul_ps(top_b, v_inv_dy_sse));
                    s_a = _mm_fmadd_ps(bot_a, v_dy_sse, _mm_mul_ps(top_a, v_inv_dy_sse));
                }

                __m128i v_dst_raw = _mm_loadu_si128(reinterpret_cast<__m128i*>(destRow + x));
                __m128 d_r, d_g, d_b, d_a;
                unpack_colors_sse(v_dst_raw, d_r, d_g, d_b, d_a);

                __m128 v_sa_norm = _mm_mul_ps(s_a, v_inv255_sse);
                __m128 v_inv_sa = _mm_sub_ps(v_ones_sse, v_sa_norm);

                __m128 out_r = _mm_fmadd_ps(s_r, v_sa_norm, _mm_mul_ps(d_r, v_inv_sa));
                __m128 out_g = _mm_fmadd_ps(s_g, v_sa_norm, _mm_mul_ps(d_g, v_inv_sa));
                __m128 out_b = _mm_fmadd_ps(s_b, v_sa_norm, _mm_mul_ps(d_b, v_inv_sa));
                __m128 out_a = _mm_fmadd_ps(s_a, v_ones_sse, _mm_mul_ps(d_a, v_inv_sa));

                out_r = _mm_min_ps(out_r, v_255_sse);
                out_g = _mm_min_ps(out_g, v_255_sse);
                out_b = _mm_min_ps(out_b, v_255_sse);
                out_a = _mm_min_ps(out_a, v_255_sse);

                __m128i v_result = pack_colors_sse(out_r, out_g, out_b, out_a);
                __m128i v_mask_i = _mm_castps_si128(v_mask);
                __m128i v_final = _mm_blendv_epi8(v_dst_raw, v_result, v_mask_i);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(destRow + x), v_final);
            }

            for (; x < endX; ++x) {
                float sx = (x - centerX) * invScaleX + srcCenterX;

                if (sx >= 0.0f && sx < src.width - 1.05f) {
                    int sx0 = static_cast<int>(sx);
                    float dx = sx - sx0;

                    const Color& c0 = row0_ptr[sx0]; const Color& c1 = row0_ptr[sx0 + 1];
                    const Color& c2 = row1_ptr[sx0]; const Color& c3 = row1_ptr[sx0 + 1];

                    float w_inv_dx = 1.0f - dx;
                    float r_top = c0.r * w_inv_dx + c1.r * dx;
                    float g_top = c0.g * w_inv_dx + c1.g * dx;
                    float b_top = c0.b * w_inv_dx + c1.b * dx;
                    float a_top = c0.a * w_inv_dx + c1.a * dx;

                    float r_bot = c2.r * w_inv_dx + c3.r * dx;
                    float g_bot = c2.g * w_inv_dx + c3.g * dx;
                    float b_bot = c2.b * w_inv_dx + c3.b * dx;
                    float a_bot = c2.a * w_inv_dx + c3.a * dx;

                    float w_inv_dy = 1.0f - dy;
                    float r = r_top * w_inv_dy + r_bot * dy;
                    float g = g_top * w_inv_dy + g_bot * dy;
                    float b = b_top * w_inv_dy + b_bot * dy;
                    float a = a_top * w_inv_dy + a_bot * dy;

                    Color& dstC = destRow[x];
                    float sa = a / 255.0f;
                    float da = 1.0f - sa;

                    dstC.r = static_cast<uint8_t>(std::min(255.0f, r * sa + dstC.r * da));
                    dstC.g = static_cast<uint8_t>(std::min(255.0f, g * sa + dstC.g * da));
                    dstC.b = static_cast<uint8_t>(std::min(255.0f, b * sa + dstC.b * da));
                    dstC.a = static_cast<uint8_t>(std::min(255.0f, a + dstC.a * da));
                }
            }
        }
    }

    void drawResized(Buffer& dest, const Buffer& src, float centerX, float centerY, int width, int height) {
        if (!dest.isValid() || !src.isValid() || width <= 0 || height <= 0) return;

        int startX = std::max(0, static_cast<int>(std::round(centerX - width * 0.5f)));
        int startY = std::max(0, static_cast<int>(std::round(centerY - height * 0.5f)));
        int endX = std::min(dest.width, static_cast<int>(std::round(centerX + width * 0.5f)));
        int endY = std::min(dest.height, static_cast<int>(std::round(centerY + height * 0.5f)));

        int actualWidth = std::min(width, endX - startX);
        int actualHeight = std::min(height, endY - startY);

        if (actualWidth <= 0 || actualHeight <= 0) return;

        float invScaleX = static_cast<float>(src.width) / actualWidth;
        float invScaleY = static_cast<float>(src.height) / actualHeight;

        const float srcW_limit = static_cast<float>(src.width) - 1.05f;
        const float srcH_limit = static_cast<float>(src.height) - 1.05f;
        const int src_stride = src.width;

        const __m256 v_invScaleX = _mm256_set1_ps(invScaleX);
        const __m256 v_invScaleY = _mm256_set1_ps(invScaleY);
        const __m256 v_ones = _mm256_set1_ps(1.0f);
        const __m256 v_idx_step8 = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256 v_srcW_limit = _mm256_set1_ps(srcW_limit);
        const __m256 v_srcH_limit = _mm256_set1_ps(srcH_limit);
        const __m256 v_255 = _mm256_set1_ps(255.0f);
        const __m256 v_inv255 = _mm256_set1_ps(1.0f / 255.0f);

        const __m128 v_invScaleX_sse = _mm_set1_ps(invScaleX);
        const __m128 v_invScaleY_sse = _mm_set1_ps(invScaleY);
        const __m128 v_idx_step4 = _mm_setr_ps(0, 1, 2, 3);
        const __m128 v_srcW_limit_sse = _mm_set1_ps(srcW_limit);
        const __m128 v_srcH_limit_sse = _mm_set1_ps(srcH_limit);
        const __m128 v_ones_sse = _mm_set1_ps(1.0f);
        const __m128 v_255_sse = _mm_set1_ps(255.0f);
        const __m128 v_inv255_sse = _mm_set1_ps(1.0f / 255.0f);

        for (int y = 0; y < actualHeight; ++y) {
            int destY = startY + y;
            if (destY >= dest.height) break;

            float srcY = y * invScaleY;
            if (srcY < 0.0f) srcY = 0.0f;
            if (srcY > srcH_limit) srcY = srcH_limit;

            int sy0 = static_cast<int>(srcY);
            int sy1 = (sy0 + 1 < src.height) ? sy0 + 1 : sy0;
            float dy = srcY - sy0;

            __m256 v_dy = _mm256_set1_ps(dy);
            __m256 v_inv_dy = _mm256_sub_ps(v_ones, v_dy);

            __m128 v_dy_sse = _mm_set1_ps(dy);
            __m128 v_inv_dy_sse = _mm_sub_ps(v_ones_sse, v_dy_sse);

            Color* destRow = dest.getRow(destY);
            const Color* row0_ptr = src.color + sy0 * src_stride;
            const Color* row1_ptr = src.color + sy1 * src_stride;

            int x = 0;
            for (; x <= actualWidth - 8; x += 8) {
                int destX = startX + x;
                if (destX + 8 > dest.width) break;

                __m256 v_srcX = _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps((float)x), v_idx_step8), v_invScaleX);

                __m256 v_mask = _mm256_and_ps(
                    _mm256_cmp_ps(v_srcX, v_ones, _CMP_GE_OQ),
                    _mm256_cmp_ps(v_srcX, v_srcW_limit, _CMP_LE_OQ)
                );

                if (_mm256_movemask_ps(v_mask) == 0) continue;

                __m256 v_safe_sx = _mm256_blendv_ps(v_ones, v_srcX, v_mask);
                __m256 v_floor_x = _mm256_floor_ps(v_safe_sx);
                __m256i v_sx0 = _mm256_cvtps_epi32(v_floor_x);

                __m256 v_dx = _mm256_sub_ps(v_safe_sx, v_floor_x);
                __m256 v_inv_dx = _mm256_sub_ps(v_ones, v_dx);

                __m256 v_w1 = _mm256_mul_ps(v_inv_dx, v_inv_dy);
                __m256 v_w2 = _mm256_mul_ps(v_dx, v_inv_dy);
                __m256 v_w3 = _mm256_mul_ps(v_inv_dx, v_dy);
                __m256 v_w4 = _mm256_mul_ps(v_dx, v_dy);

                __m256i v_offset0 = _mm256_add_epi32(_mm256_set1_epi32(sy0 * src_stride), v_sx0);
                __m256i v_offset1 = _mm256_add_epi32(_mm256_set1_epi32(sy1 * src_stride), v_sx0);

                __m256i c00_i = _mm256_i32gather_epi32(reinterpret_cast<const int*>(row0_ptr), v_offset0, 4);
                __m256i c10_i = _mm256_i32gather_epi32(reinterpret_cast<const int*>(row0_ptr),
                    _mm256_add_epi32(v_offset0, _mm256_set1_epi32(1)), 4);
                __m256i c01_i = _mm256_i32gather_epi32(reinterpret_cast<const int*>(row1_ptr), v_offset0, 4);
                __m256i c11_i = _mm256_i32gather_epi32(reinterpret_cast<const int*>(row1_ptr),
                    _mm256_add_epi32(v_offset0, _mm256_set1_epi32(1)), 4);

                __m256 s_r, s_g, s_b, s_a;
                {
                    __m256 r00, g00, b00, a00; unpack_colors_avx2(c00_i, r00, g00, b00, a00);
                    __m256 r10, g10, b10, a10; unpack_colors_avx2(c10_i, r10, g10, b10, a10);
                    __m256 r01, g01, b01, a01; unpack_colors_avx2(c01_i, r01, g01, b01, a01);
                    __m256 r11, g11, b11, a11; unpack_colors_avx2(c11_i, r11, g11, b11, a11);

                    __m256 top_r = _mm256_fmadd_ps(r10, v_dx, _mm256_mul_ps(r00, v_inv_dx));
                    __m256 top_g = _mm256_fmadd_ps(g10, v_dx, _mm256_mul_ps(g00, v_inv_dx));
                    __m256 top_b = _mm256_fmadd_ps(b10, v_dx, _mm256_mul_ps(b00, v_inv_dx));
                    __m256 top_a = _mm256_fmadd_ps(a10, v_dx, _mm256_mul_ps(a00, v_inv_dx));

                    __m256 bot_r = _mm256_fmadd_ps(r11, v_dx, _mm256_mul_ps(r01, v_inv_dx));
                    __m256 bot_g = _mm256_fmadd_ps(g11, v_dx, _mm256_mul_ps(g01, v_inv_dx));
                    __m256 bot_b = _mm256_fmadd_ps(b11, v_dx, _mm256_mul_ps(b01, v_inv_dx));
                    __m256 bot_a = _mm256_fmadd_ps(a11, v_dx, _mm256_mul_ps(a01, v_inv_dx));

                    s_r = _mm256_fmadd_ps(bot_r, v_dy, _mm256_mul_ps(top_r, v_inv_dy));
                    s_g = _mm256_fmadd_ps(bot_g, v_dy, _mm256_mul_ps(top_g, v_inv_dy));
                    s_b = _mm256_fmadd_ps(bot_b, v_dy, _mm256_mul_ps(top_b, v_inv_dy));
                    s_a = _mm256_fmadd_ps(bot_a, v_dy, _mm256_mul_ps(top_a, v_inv_dy));
                }

                __m256i v_dst_raw = _mm256_loadu_si256(reinterpret_cast<__m256i*>(destRow + destX));
                __m256 d_r, d_g, d_b, d_a;
                unpack_colors_avx2(v_dst_raw, d_r, d_g, d_b, d_a);

                __m256 v_sa_norm = _mm256_mul_ps(s_a, v_inv255);
                __m256 v_inv_sa = _mm256_sub_ps(v_ones, v_sa_norm);

                __m256 out_r = _mm256_fmadd_ps(s_r, v_sa_norm, _mm256_mul_ps(d_r, v_inv_sa));
                __m256 out_g = _mm256_fmadd_ps(s_g, v_sa_norm, _mm256_mul_ps(d_g, v_inv_sa));
                __m256 out_b = _mm256_fmadd_ps(s_b, v_sa_norm, _mm256_mul_ps(d_b, v_inv_sa));
                __m256 out_a = _mm256_fmadd_ps(s_a, v_ones, _mm256_mul_ps(d_a, v_inv_sa));

                out_r = _mm256_min_ps(out_r, v_255);
                out_g = _mm256_min_ps(out_g, v_255);
                out_b = _mm256_min_ps(out_b, v_255);
                out_a = _mm256_min_ps(out_a, v_255);

                __m256i v_result = pack_colors_avx2(out_r, out_g, out_b, out_a);
                __m256i v_mask_i = _mm256_castps_si256(v_mask);
                __m256i v_final = _mm256_blendv_epi8(v_dst_raw, v_result, v_mask_i);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(destRow + destX), v_final);
            }

            for (; x <= actualWidth - 4; x += 4) {
                int destX = startX + x;
                if (destX + 4 > dest.width) break;

                __m128 v_srcX = _mm_mul_ps(_mm_add_ps(_mm_set1_ps((float)x), v_idx_step4), v_invScaleX_sse);

                __m128 v_mask = _mm_and_ps(
                    _mm_cmpge_ps(v_srcX, v_ones_sse),
                    _mm_cmple_ps(v_srcX, v_srcW_limit_sse)
                );

                if (_mm_movemask_ps(v_mask) == 0) continue;

                __m128 v_safe_sx = _mm_blendv_ps(v_ones_sse, v_srcX, v_mask);
                __m128 v_floor_x = _mm_floor_ps(v_safe_sx);
                __m128i v_sx0 = _mm_cvtps_epi32(v_floor_x);

                __m128 v_dx = _mm_sub_ps(v_safe_sx, v_floor_x);
                __m128 v_inv_dx = _mm_sub_ps(v_ones_sse, v_dx);

                __m128 v_w1 = _mm_mul_ps(v_inv_dx, v_inv_dy_sse);
                __m128 v_w2 = _mm_mul_ps(v_dx, v_inv_dy_sse);
                __m128 v_w3 = _mm_mul_ps(v_inv_dx, v_dy_sse);
                __m128 v_w4 = _mm_mul_ps(v_dx, v_dy_sse);

                alignas(16) int indices[4];
                _mm_store_si128(reinterpret_cast<__m128i*>(indices), v_sx0);

                __m128i c00_i = _mm_set_epi32(
                    reinterpret_cast<const int*>(row0_ptr)[indices[3]],
                    reinterpret_cast<const int*>(row0_ptr)[indices[2]],
                    reinterpret_cast<const int*>(row0_ptr)[indices[1]],
                    reinterpret_cast<const int*>(row0_ptr)[indices[0]]
                );

                __m128i c10_i = _mm_set_epi32(
                    reinterpret_cast<const int*>(row0_ptr)[indices[3] + 1],
                    reinterpret_cast<const int*>(row0_ptr)[indices[2] + 1],
                    reinterpret_cast<const int*>(row0_ptr)[indices[1] + 1],
                    reinterpret_cast<const int*>(row0_ptr)[indices[0] + 1]
                );

                __m128i c01_i = _mm_set_epi32(
                    reinterpret_cast<const int*>(row1_ptr)[indices[3]],
                    reinterpret_cast<const int*>(row1_ptr)[indices[2]],
                    reinterpret_cast<const int*>(row1_ptr)[indices[1]],
                    reinterpret_cast<const int*>(row1_ptr)[indices[0]]
                );

                __m128i c11_i = _mm_set_epi32(
                    reinterpret_cast<const int*>(row1_ptr)[indices[3] + 1],
                    reinterpret_cast<const int*>(row1_ptr)[indices[2] + 1],
                    reinterpret_cast<const int*>(row1_ptr)[indices[1] + 1],
                    reinterpret_cast<const int*>(row1_ptr)[indices[0] + 1]
                );

                __m128 s_r, s_g, s_b, s_a;
                {
                    __m128 r00, g00, b00, a00; unpack_colors_sse(c00_i, r00, g00, b00, a00);
                    __m128 r10, g10, b10, a10; unpack_colors_sse(c10_i, r10, g10, b10, a10);
                    __m128 r01, g01, b01, a01; unpack_colors_sse(c01_i, r01, g01, b01, a01);
                    __m128 r11, g11, b11, a11; unpack_colors_sse(c11_i, r11, g11, b11, a11);

                    __m128 top_r = _mm_fmadd_ps(r10, v_dx, _mm_mul_ps(r00, v_inv_dx));
                    __m128 top_g = _mm_fmadd_ps(g10, v_dx, _mm_mul_ps(g00, v_inv_dx));
                    __m128 top_b = _mm_fmadd_ps(b10, v_dx, _mm_mul_ps(b00, v_inv_dx));
                    __m128 top_a = _mm_fmadd_ps(a10, v_dx, _mm_mul_ps(a00, v_inv_dx));

                    __m128 bot_r = _mm_fmadd_ps(r11, v_dx, _mm_mul_ps(r01, v_inv_dx));
                    __m128 bot_g = _mm_fmadd_ps(g11, v_dx, _mm_mul_ps(g01, v_inv_dx));
                    __m128 bot_b = _mm_fmadd_ps(b11, v_dx, _mm_mul_ps(b01, v_inv_dx));
                    __m128 bot_a = _mm_fmadd_ps(a11, v_dx, _mm_mul_ps(a01, v_inv_dx));

                    s_r = _mm_fmadd_ps(bot_r, v_dy_sse, _mm_mul_ps(top_r, v_inv_dy_sse));
                    s_g = _mm_fmadd_ps(bot_g, v_dy_sse, _mm_mul_ps(top_g, v_inv_dy_sse));
                    s_b = _mm_fmadd_ps(bot_b, v_dy_sse, _mm_mul_ps(top_b, v_inv_dy_sse));
                    s_a = _mm_fmadd_ps(bot_a, v_dy_sse, _mm_mul_ps(top_a, v_inv_dy_sse));
                }

                __m128i v_dst_raw = _mm_loadu_si128(reinterpret_cast<__m128i*>(destRow + destX));
                __m128 d_r, d_g, d_b, d_a;
                unpack_colors_sse(v_dst_raw, d_r, d_g, d_b, d_a);

                __m128 v_sa_norm = _mm_mul_ps(s_a, v_inv255_sse);
                __m128 v_inv_sa = _mm_sub_ps(v_ones_sse, v_sa_norm);

                __m128 out_r = _mm_fmadd_ps(s_r, v_sa_norm, _mm_mul_ps(d_r, v_inv_sa));
                __m128 out_g = _mm_fmadd_ps(s_g, v_sa_norm, _mm_mul_ps(d_g, v_inv_sa));
                __m128 out_b = _mm_fmadd_ps(s_b, v_sa_norm, _mm_mul_ps(d_b, v_inv_sa));
                __m128 out_a = _mm_fmadd_ps(s_a, v_ones_sse, _mm_mul_ps(d_a, v_inv_sa));

                out_r = _mm_min_ps(out_r, v_255_sse);
                out_g = _mm_min_ps(out_g, v_255_sse);
                out_b = _mm_min_ps(out_b, v_255_sse);
                out_a = _mm_min_ps(out_a, v_255_sse);

                __m128i v_result = pack_colors_sse(out_r, out_g, out_b, out_a);
                __m128i v_mask_i = _mm_castps_si128(v_mask);
                __m128i v_final = _mm_blendv_epi8(v_dst_raw, v_result, v_mask_i);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(destRow + destX), v_final);
            }

            for (; x < actualWidth; ++x) {
                int destX = startX + x;
                if (destX >= dest.width) break;

                float srcX = x * invScaleX;
                if (srcX >= 0.0f && srcX < src.width - 1.05f) {
                    int srcX0 = static_cast<int>(srcX);
                    int srcX1 = std::min(srcX0 + 1, src.width - 1);
                    float dx = srcX - srcX0;

                    const Color& c0 = row0_ptr[srcX0];
                    const Color& c1 = row0_ptr[srcX1];
                    const Color& c2 = row1_ptr[srcX0];
                    const Color& c3 = row1_ptr[srcX1];

                    float w_inv_dx = 1.0f - dx;
                    float r_top = c0.r * w_inv_dx + c1.r * dx;
                    float g_top = c0.g * w_inv_dx + c1.g * dx;
                    float b_top = c0.b * w_inv_dx + c1.b * dx;
                    float a_top = c0.a * w_inv_dx + c1.a * dx;

                    float r_bot = c2.r * w_inv_dx + c3.r * dx;
                    float g_bot = c2.g * w_inv_dx + c3.g * dx;
                    float b_bot = c2.b * w_inv_dx + c3.b * dx;
                    float a_bot = c2.a * w_inv_dx + c3.a * dx;

                    float w_inv_dy = 1.0f - dy;
                    float r = r_top * w_inv_dy + r_bot * dy;
                    float g = g_top * w_inv_dy + g_bot * dy;
                    float b = b_top * w_inv_dy + b_bot * dy;
                    float a = a_top * w_inv_dy + a_bot * dy;

                    Color& dstC = destRow[destX];
                    float sa = a / 255.0f;
                    float da = 1.0f - sa;

                    dstC.r = static_cast<uint8_t>(std::min(255.0f, r * sa + dstC.r * da));
                    dstC.g = static_cast<uint8_t>(std::min(255.0f, g * sa + dstC.g * da));
                    dstC.b = static_cast<uint8_t>(std::min(255.0f, b * sa + dstC.b * da));
                    dstC.a = static_cast<uint8_t>(std::min(255.0f, a + dstC.a * da));
                }
            }
        }
    }

    void drawScaled(Buffer& dest, const Buffer& src, float centerX, float centerY, float scale) {
        drawScaled(dest, src, centerX, centerY, scale, scale);
    }

    void drawRotated(Buffer& dest, const Buffer& src, float centerX, float centerY, float rotation) {
        if (!dest.isValid() || !src.isValid()) return;

        float rad = std::fmod(rotation, 360.0f);
        if (rad < 0) rad += 360.0f;

        if (std::abs(rad) < 0.001f || rad > 360.0f - 0.001f) {
            alphaBlend(src, dest, centerX - src.width / 2, centerY - src.height / 2);
            return;
        }
        if (std::abs(rad - 90.0f) < 0.001f) {
            drawRotated90Scaled(dest, src, centerX, centerY, 1.0f, 1.0f);
            return;
        }
        if (std::abs(rad - 180.0f) < 0.001f) {
            drawRotated180Scaled(dest, src, centerX, centerY, 1.0f, 1.0f);
            return;
        }
        if (std::abs(rad - 270.0f) < 0.001f) {
            drawRotated270Scaled(dest, src, centerX, centerY, 1.0f, 1.0f);
            return;
        }

        const float cosA = std::cos(rad);
        const float sinA = std::sin(rad);
        const float halfW = src.width * 0.5f;
        const float halfH = src.height * 0.5f;

        const float absCos = std::abs(halfW * cosA);
        const float absSin = std::abs(halfW * sinA);
        const float absCosH = std::abs(halfH * cosA);
        const float absSinH = std::abs(halfH * sinA);

        int startX = std::max(0, static_cast<int>(centerX - std::max(absCos + absSinH, halfW)));
        int startY = std::max(0, static_cast<int>(centerY - std::max(absSin + absCosH, halfH)));
        int endX = std::min(dest.width, static_cast<int>(centerX + std::max(absCos + absSinH, halfW)) + 1);
        int endY = std::min(dest.height, static_cast<int>(centerY + std::max(absSin + absCosH, halfH)) + 1);

        const float dx_step_val = cosA;
        const float dy_step_val = -sinA;

        const __m256 v_dx_step = _mm256_set1_ps(dx_step_val);
        const __m256 v_dy_step = _mm256_set1_ps(dy_step_val);
        const __m256 v_idx_step8 = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256 v_ones = _mm256_set1_ps(1.0f);
        const __m256 v_zeros = _mm256_setzero_ps();

        const __m256 v_srcW_limit = _mm256_set1_ps((float)src.width - 0.001f);
        const __m256 v_srcH_limit = _mm256_set1_ps((float)src.height - 0.001f);

        const __m256i v_max_w_idx = _mm256_set1_epi32(src.width - 1);
        const __m256i v_max_h_idx = _mm256_set1_epi32(src.height - 1);
        const __m256i v_zero_idx = _mm256_setzero_si256();
        const __m256i v_border_color = _mm256_set1_epi32(0x00FFFFFF);

        const __m256 v_255 = _mm256_set1_ps(255.0f);
        const __m256 v_inv255 = _mm256_set1_ps(1.0f / 255.0f);

        const __m128 v_dx_step_sse = _mm_set1_ps(dx_step_val);
        const __m128 v_dy_step_sse = _mm_set1_ps(dy_step_val);
        const __m128 v_idx_step4 = _mm_setr_ps(0, 1, 2, 3);
        const __m128 v_ones_sse = _mm_set1_ps(1.0f);
        const __m128 v_zeros_sse = _mm_setzero_ps();
        const __m128 v_srcW_limit_sse = _mm_set1_ps((float)src.width - 0.001f);
        const __m128 v_srcH_limit_sse = _mm_set1_ps((float)src.height - 0.001f);
        const __m128i v_max_w_idx_sse = _mm_set1_epi32(src.width - 1);
        const __m128i v_max_h_idx_sse = _mm_set1_epi32(src.height - 1);
        const __m128i v_zero_idx_sse = _mm_setzero_si128();
        const __m128i v_border_color_sse = _mm_set1_epi32(0x00FFFFFF);
        const __m128 v_255_sse = _mm_set1_ps(255.0f);
        const __m128 v_inv255_sse = _mm_set1_ps(1.0f / 255.0f);

        const int src_stride = src.width;
        const float srcCenterX = src.width * 0.5f;
        const float srcCenterY = src.height * 0.5f;

        for (int y = startY; y < endY; ++y) {
            float relY = (y - centerY);
            float startRelX = (startX - centerX);

            float currentSrcX = startRelX * cosA + relY * sinA + srcCenterX;
            float currentSrcY = -startRelX * sinA + relY * cosA + srcCenterY;

            Color* destRow = dest.getRow(y);
            int x = startX;

            for (; x <= endX - 8; x += 8) {
                __m256 v_sx = _mm256_fmadd_ps(v_idx_step8, v_dx_step, _mm256_set1_ps(currentSrcX));
                __m256 v_sy = _mm256_fmadd_ps(v_idx_step8, v_dy_step, _mm256_set1_ps(currentSrcY));

                currentSrcX += dx_step_val * 8.0f;
                currentSrcY += dy_step_val * 8.0f;

                __m256 v_mask = _mm256_and_ps(_mm256_cmp_ps(v_sx, _mm256_set1_ps(-1.0f), _CMP_GE_OQ),
                    _mm256_cmp_ps(v_sx, _mm256_add_ps(v_srcW_limit, v_ones), _CMP_LE_OQ));
                v_mask = _mm256_and_ps(v_mask, _mm256_cmp_ps(v_sy, _mm256_set1_ps(-1.0f), _CMP_GE_OQ));
                v_mask = _mm256_and_ps(v_mask, _mm256_cmp_ps(v_sy, _mm256_add_ps(v_srcH_limit, v_ones), _CMP_LE_OQ));

                if (_mm256_movemask_ps(v_mask) == 0) continue;

                __m256 v_safe_sx = _mm256_blendv_ps(v_zeros, v_sx, v_mask);
                __m256 v_safe_sy = _mm256_blendv_ps(v_zeros, v_sy, v_mask);

                __m256 v_flr_x = _mm256_floor_ps(v_safe_sx);
                __m256 v_flr_y = _mm256_floor_ps(v_safe_sy);
                __m256i v_idx_x = _mm256_cvtps_epi32(v_flr_x);
                __m256i v_idx_y = _mm256_cvtps_epi32(v_flr_y);

                __m256 v_dx = _mm256_sub_ps(v_safe_sx, v_flr_x);
                __m256 v_dy = _mm256_sub_ps(v_safe_sy, v_flr_y);
                __m256 v_inv_dx = _mm256_sub_ps(v_ones, v_dx);
                __m256 v_inv_dy = _mm256_sub_ps(v_ones, v_dy);

                __m256 v_w1 = _mm256_mul_ps(v_inv_dx, v_inv_dy);
                __m256 v_w2 = _mm256_mul_ps(v_dx, v_inv_dy);
                __m256 v_w3 = _mm256_mul_ps(v_inv_dx, v_dy);
                __m256 v_w4 = _mm256_mul_ps(v_dx, v_dy);

                auto safe_gather_avx2 = [&](__m256i ix, __m256i iy) {
                    __m256i mask_x = _mm256_and_si256(_mm256_cmpgt_epi32(ix, _mm256_set1_epi32(-1)), _mm256_cmpgt_epi32(_mm256_set1_epi32(src.width), ix));
                    __m256i mask_y = _mm256_and_si256(_mm256_cmpgt_epi32(iy, _mm256_set1_epi32(-1)), _mm256_cmpgt_epi32(_mm256_set1_epi32(src.height), iy));
                    __m256i valid_mask = _mm256_and_si256(mask_x, mask_y);

                    __m256i safe_x = _mm256_max_epi32(v_zero_idx, _mm256_min_epi32(ix, v_max_w_idx));
                    __m256i safe_y = _mm256_max_epi32(v_zero_idx, _mm256_min_epi32(iy, v_max_h_idx));

                    __m256i offset = _mm256_add_epi32(_mm256_mullo_epi32(safe_y, _mm256_set1_epi32(src_stride)), safe_x);
                    const int* sptr = reinterpret_cast<const int*>(src.color);
                    __m256i colors = _mm256_i32gather_epi32(sptr, offset, 4);

                    return _mm256_blendv_epi8(v_border_color, colors, valid_mask);
                    };

                __m256i c00 = safe_gather_avx2(v_idx_x, v_idx_y);
                __m256i c10 = safe_gather_avx2(_mm256_add_epi32(v_idx_x, _mm256_set1_epi32(1)), v_idx_y);
                __m256i c01 = safe_gather_avx2(v_idx_x, _mm256_add_epi32(v_idx_y, _mm256_set1_epi32(1)));
                __m256i c11 = safe_gather_avx2(_mm256_add_epi32(v_idx_x, _mm256_set1_epi32(1)), _mm256_add_epi32(v_idx_y, _mm256_set1_epi32(1)));

                __m256 s_r, s_g, s_b, s_a;
                {
                    __m256 r0, g0, b0, a0; unpack_colors_avx2(c00, r0, g0, b0, a0);
                    __m256 r1, g1, b1, a1; unpack_colors_avx2(c10, r1, g1, b1, a1);
                    __m256 r2, g2, b2, a2; unpack_colors_avx2(c01, r2, g2, b2, a2);
                    __m256 r3, g3, b3, a3; unpack_colors_avx2(c11, r3, g3, b3, a3);

                    s_r = _mm256_fmadd_ps(r0, v_w1, _mm256_fmadd_ps(r1, v_w2, _mm256_fmadd_ps(r2, v_w3, _mm256_mul_ps(r3, v_w4))));
                    s_g = _mm256_fmadd_ps(g0, v_w1, _mm256_fmadd_ps(g1, v_w2, _mm256_fmadd_ps(g2, v_w3, _mm256_mul_ps(g3, v_w4))));
                    s_b = _mm256_fmadd_ps(b0, v_w1, _mm256_fmadd_ps(b1, v_w2, _mm256_fmadd_ps(b2, v_w3, _mm256_mul_ps(b3, v_w4))));
                    s_a = _mm256_fmadd_ps(a0, v_w1, _mm256_fmadd_ps(a1, v_w2, _mm256_fmadd_ps(a2, v_w3, _mm256_mul_ps(a3, v_w4))));
                }

                __m256i v_dst_raw = _mm256_loadu_si256(reinterpret_cast<__m256i*>(destRow + x));
                __m256 d_r, d_g, d_b, d_a;
                unpack_colors_avx2(v_dst_raw, d_r, d_g, d_b, d_a);

                __m256 v_sa_norm = _mm256_mul_ps(s_a, v_inv255);
                __m256 v_da_norm = _mm256_mul_ps(d_a, v_inv255);
                __m256 v_out_a = _mm256_add_ps(v_sa_norm, _mm256_mul_ps(v_da_norm, _mm256_sub_ps(v_ones, v_sa_norm)));

                __m256 v_sa_factor = _mm256_div_ps(v_sa_norm, _mm256_max_ps(v_out_a, _mm256_set1_ps(0.0001f)));
                __m256 v_da_factor = _mm256_sub_ps(v_ones, v_sa_factor);

                __m256 is_transparent = _mm256_cmp_ps(v_out_a, _mm256_set1_ps(0.0001f), _CMP_LT_OQ);
                v_sa_factor = _mm256_blendv_ps(v_sa_factor, v_zeros, is_transparent);
                v_da_factor = _mm256_blendv_ps(v_da_factor, v_zeros, is_transparent);

                __m256 out_r = _mm256_fmadd_ps(s_r, v_sa_factor, _mm256_mul_ps(d_r, v_da_factor));
                __m256 out_g = _mm256_fmadd_ps(s_g, v_sa_factor, _mm256_mul_ps(d_g, v_da_factor));
                __m256 out_b = _mm256_fmadd_ps(s_b, v_sa_factor, _mm256_mul_ps(d_b, v_da_factor));
                __m256 out_a_scaled = _mm256_mul_ps(v_out_a, v_255);

                __m256i v_result = pack_colors_avx2(out_r, out_g, out_b, out_a_scaled);
                __m256i v_mask_i = _mm256_castps_si256(v_mask);
                __m256i v_final = _mm256_blendv_epi8(v_dst_raw, v_result, v_mask_i);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(destRow + x), v_final);
            }

            for (; x <= endX - 4; x += 4) {
                __m128 v_sx = _mm_fmadd_ps(v_idx_step4, v_dx_step_sse, _mm_set1_ps(currentSrcX));
                __m128 v_sy = _mm_fmadd_ps(v_idx_step4, v_dy_step_sse, _mm_set1_ps(currentSrcY));

                currentSrcX += dx_step_val * 4.0f;
                currentSrcY += dy_step_val * 4.0f;

                __m128 v_mask = _mm_and_ps(_mm_cmpge_ps(v_sx, _mm_set1_ps(-1.0f)), _mm_cmple_ps(v_sx, _mm_add_ps(v_srcW_limit_sse, v_ones_sse)));
                v_mask = _mm_and_ps(v_mask, _mm_cmpge_ps(v_sy, _mm_set1_ps(-1.0f)));
                v_mask = _mm_and_ps(v_mask, _mm_cmple_ps(v_sy, _mm_add_ps(v_srcH_limit_sse, v_ones_sse)));

                if (_mm_movemask_ps(v_mask) == 0) continue;

                __m128 v_safe_sx = _mm_blendv_ps(v_zeros_sse, v_sx, v_mask);
                __m128 v_safe_sy = _mm_blendv_ps(v_zeros_sse, v_sy, v_mask);

                __m128 v_flr_x = _mm_floor_ps(v_safe_sx);
                __m128 v_flr_y = _mm_floor_ps(v_safe_sy);
                __m128i v_idx_x = _mm_cvtps_epi32(v_flr_x);
                __m128i v_idx_y = _mm_cvtps_epi32(v_flr_y);

                __m128 v_dx = _mm_sub_ps(v_safe_sx, v_flr_x);
                __m128 v_dy = _mm_sub_ps(v_safe_sy, v_flr_y);
                __m128 v_inv_dx = _mm_sub_ps(v_ones_sse, v_dx);
                __m128 v_inv_dy = _mm_sub_ps(v_ones_sse, v_dy);

                __m128 v_w1 = _mm_mul_ps(v_inv_dx, v_inv_dy);
                __m128 v_w2 = _mm_mul_ps(v_dx, v_inv_dy);
                __m128 v_w3 = _mm_mul_ps(v_inv_dx, v_dy);
                __m128 v_w4 = _mm_mul_ps(v_dx, v_dy);

                auto safe_gather_sse = [&](__m128i ix, __m128i iy) {
                    __m128i mask_x = _mm_and_si128(_mm_cmpgt_epi32(ix, _mm_set1_epi32(-1)), _mm_cmpgt_epi32(_mm_set1_epi32(src.width), ix));
                    __m128i mask_y = _mm_and_si128(_mm_cmpgt_epi32(iy, _mm_set1_epi32(-1)), _mm_cmpgt_epi32(_mm_set1_epi32(src.height), iy));
                    __m128i valid_mask = _mm_and_si128(mask_x, mask_y);

                    __m128i safe_x = _mm_max_epi32(v_zero_idx_sse, _mm_min_epi32(ix, v_max_w_idx_sse));
                    __m128i safe_y = _mm_max_epi32(v_zero_idx_sse, _mm_min_epi32(iy, v_max_h_idx_sse));

                    alignas(16) int idx_x_buf[4], idx_y_buf[4];
                    _mm_store_si128(reinterpret_cast<__m128i*>(idx_x_buf), safe_x);
                    _mm_store_si128(reinterpret_cast<__m128i*>(idx_y_buf), safe_y);

                    int c0 = reinterpret_cast<const int*>(src.color)[idx_y_buf[0] * src_stride + idx_x_buf[0]];
                    int c1 = reinterpret_cast<const int*>(src.color)[idx_y_buf[1] * src_stride + idx_x_buf[1]];
                    int c2 = reinterpret_cast<const int*>(src.color)[idx_y_buf[2] * src_stride + idx_x_buf[2]];
                    int c3 = reinterpret_cast<const int*>(src.color)[idx_y_buf[3] * src_stride + idx_x_buf[3]];

                    __m128i colors = _mm_set_epi32(c3, c2, c1, c0);
                    return _mm_blendv_epi8(v_border_color_sse, colors, valid_mask);
                    };

                __m128i c00 = safe_gather_sse(v_idx_x, v_idx_y);
                __m128i c10 = safe_gather_sse(_mm_add_epi32(v_idx_x, _mm_set1_epi32(1)), v_idx_y);
                __m128i c01 = safe_gather_sse(v_idx_x, _mm_add_epi32(v_idx_y, _mm_set1_epi32(1)));
                __m128i c11 = safe_gather_sse(_mm_add_epi32(v_idx_x, _mm_set1_epi32(1)), _mm_add_epi32(v_idx_y, _mm_set1_epi32(1)));

                __m128 s_r, s_g, s_b, s_a;
                {
                    __m128 r0, g0, b0, a0; unpack_colors_sse(c00, r0, g0, b0, a0);
                    __m128 r1, g1, b1, a1; unpack_colors_sse(c10, r1, g1, b1, a1);
                    __m128 r2, g2, b2, a2; unpack_colors_sse(c01, r2, g2, b2, a2);
                    __m128 r3, g3, b3, a3; unpack_colors_sse(c11, r3, g3, b3, a3);

                    s_r = _mm_fmadd_ps(r0, v_w1, _mm_fmadd_ps(r1, v_w2, _mm_fmadd_ps(r2, v_w3, _mm_mul_ps(r3, v_w4))));
                    s_g = _mm_fmadd_ps(g0, v_w1, _mm_fmadd_ps(g1, v_w2, _mm_fmadd_ps(g2, v_w3, _mm_mul_ps(g3, v_w4))));
                    s_b = _mm_fmadd_ps(b0, v_w1, _mm_fmadd_ps(b1, v_w2, _mm_fmadd_ps(b2, v_w3, _mm_mul_ps(b3, v_w4))));
                    s_a = _mm_fmadd_ps(a0, v_w1, _mm_fmadd_ps(a1, v_w2, _mm_fmadd_ps(a2, v_w3, _mm_mul_ps(a3, v_w4))));
                }

                __m128i v_dst_raw = _mm_loadu_si128(reinterpret_cast<__m128i*>(destRow + x));
                __m128 d_r, d_g, d_b, d_a;
                unpack_colors_sse(v_dst_raw, d_r, d_g, d_b, d_a);

                __m128 v_sa_norm = _mm_mul_ps(s_a, v_inv255_sse);
                __m128 v_da_norm = _mm_mul_ps(d_a, v_inv255_sse);
                __m128 v_out_a = _mm_add_ps(v_sa_norm, _mm_mul_ps(v_da_norm, _mm_sub_ps(v_ones_sse, v_sa_norm)));
                __m128 v_sa_factor = _mm_div_ps(v_sa_norm, _mm_max_ps(v_out_a, _mm_set1_ps(0.0001f)));
                __m128 v_da_factor = _mm_sub_ps(v_ones_sse, v_sa_factor);

                __m128 is_transparent = _mm_cmplt_ps(v_out_a, _mm_set1_ps(0.0001f));
                v_sa_factor = _mm_blendv_ps(v_sa_factor, v_zeros_sse, is_transparent);
                v_da_factor = _mm_blendv_ps(v_da_factor, v_zeros_sse, is_transparent);

                __m128 out_r = _mm_fmadd_ps(s_r, v_sa_factor, _mm_mul_ps(d_r, v_da_factor));
                __m128 out_g = _mm_fmadd_ps(s_g, v_sa_factor, _mm_mul_ps(d_g, v_da_factor));
                __m128 out_b = _mm_fmadd_ps(s_b, v_sa_factor, _mm_mul_ps(d_b, v_da_factor));
                __m128 out_a_scaled = _mm_mul_ps(v_out_a, v_255_sse);

                __m128i v_result = pack_colors_sse(out_r, out_g, out_b, out_a_scaled);
                __m128i v_mask_i = _mm_castps_si128(v_mask);
                __m128i v_final = _mm_blendv_epi8(v_dst_raw, v_result, v_mask_i);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(destRow + x), v_final);
            }

            for (; x < endX; ++x) {
                float relX = (x - centerX);
                float srcX = relX * cosA + relY * sinA + srcCenterX;
                float srcY = -relX * sinA + relY * cosA + srcCenterY;

                if (srcX > -1.0f && srcX < src.width && srcY > -1.0f && srcY < src.height) {
                    int sx = static_cast<int>(std::floor(srcX));
                    int sy = static_cast<int>(std::floor(srcY));

                    float dx = srcX - sx;
                    float dy = srcY - sy;

                    auto getPixelSafe = [&](int px, int py) -> Color {
                        if (px >= 0 && px < src.width && py >= 0 && py < src.height) {
                            return src.at(px, py);
                        }
                        return { 255, 255, 255, 0 };
                        };

                    Color c00 = getPixelSafe(sx, sy);
                    Color c10 = getPixelSafe(sx + 1, sy);
                    Color c01 = getPixelSafe(sx, sy + 1);
                    Color c11 = getPixelSafe(sx + 1, sy + 1);

                    float w1 = (1.0f - dx) * (1.0f - dy);
                    float w2 = dx * (1.0f - dy);
                    float w3 = (1.0f - dx) * dy;
                    float w4 = dx * dy;

                    float r = c00.r * w1 + c10.r * w2 + c01.r * w3 + c11.r * w4;
                    float g = c00.g * w1 + c10.g * w2 + c01.g * w3 + c11.g * w4;
                    float b = c00.b * w1 + c10.b * w2 + c01.b * w3 + c11.b * w4;
                    float a = c00.a * w1 + c10.a * w2 + c01.a * w3 + c11.a * w4;

                    Color& dstC = destRow[x];
                    float sa = a / 255.0f;
                    float da = dstC.a / 255.0f;
                    float out_a = sa + da * (1.0f - sa);

                    if (out_a > 0.0001f) {
                        float sa_factor = sa / out_a;
                        float da_factor = 1.0f - sa_factor;

                        dstC.r = static_cast<uint8_t>(std::min(255.0f, r * sa_factor + dstC.r * da_factor));
                        dstC.g = static_cast<uint8_t>(std::min(255.0f, g * sa_factor + dstC.g * da_factor));
                        dstC.b = static_cast<uint8_t>(std::min(255.0f, b * sa_factor + dstC.b * da_factor));
                    }
                    dstC.a = static_cast<uint8_t>(std::min(255.0f, out_a * 255.0f));
                }
            }
        }
    }

    void drawTransformed(Buffer& dest, const Buffer& src, float centerX, float centerY, float scale, float rotation) {
        if (!dest.isValid() || !src.isValid() || scale <= 0.0f) return;

        float rad = std::fmod(rotation, 360.0f);
        if (rad < 0) rad += 360.0f;

        const float EPSILON = 0.001f;

        if (std::abs(rad) < EPSILON || std::abs(rad) > 360.0f - EPSILON) {
            drawScaled(dest, src, centerX, centerY, scale, scale);
            return;
        }
        else if (std::abs(rad - 90.0f) < EPSILON) {
            drawRotated90Scaled(dest, src, centerX, centerY, scale, scale);
            return;
        }
        else if (std::abs(rad - 180.0f) < EPSILON) {
            drawRotated180Scaled(dest, src, centerX, centerY, scale, scale);
            return;
        }
        else if (std::abs(rad - 270.0f) < EPSILON) {
            drawRotated270Scaled(dest, src, centerX, centerY, scale, scale);
            return;
        }

        const float cosA = std::cos(rad);
        const float sinA = std::sin(rad);
        const float halfW = src.width * scale * 0.5f;
        const float halfH = src.height * scale * 0.5f;

        const float absCos = std::abs(halfW * cosA);
        const float absSin = std::abs(halfW * sinA);
        const float absCosH = std::abs(halfH * cosA);
        const float absSinH = std::abs(halfH * sinA);

        int startX = std::max(0, static_cast<int>(centerX - std::max(absCos + absSinH, halfW)));
        int startY = std::max(0, static_cast<int>(centerY - std::max(absSin + absCosH, halfH)));
        int endX = std::min(dest.width, static_cast<int>(centerX + std::max(absCos + absSinH, halfW)) + 1);
        int endY = std::min(dest.height, static_cast<int>(centerY + std::max(absSin + absCosH, halfH)) + 1);

        const float invScale = 1.0f / scale;
        const float dx_step_val = cosA * invScale;
        const float dy_step_val = -sinA * invScale;

        const __m256 v_dx_step = _mm256_set1_ps(dx_step_val);
        const __m256 v_dy_step = _mm256_set1_ps(dy_step_val);
        const __m256 v_ones = _mm256_set1_ps(1.0f);
        const __m256 v_zeros = _mm256_setzero_ps();
        const __m256 v_idx_step8 = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);

        const __m256 v_srcW_limit = _mm256_set1_ps((float)src.width - 0.001f);
        const __m256 v_srcH_limit = _mm256_set1_ps((float)src.height - 0.001f);

        const __m256i v_max_w_idx = _mm256_set1_epi32(src.width - 1);
        const __m256i v_max_h_idx = _mm256_set1_epi32(src.height - 1);
        const __m256i v_zero_idx = _mm256_setzero_si256();
        const __m256i v_border_color = _mm256_set1_epi32(0x00FFFFFF);

        const __m256 v_255 = _mm256_set1_ps(255.0f);
        const __m256 v_inv255 = _mm256_set1_ps(1.0f / 255.0f);

        const __m128 v_dx_step_sse = _mm_set1_ps(dx_step_val);
        const __m128 v_dy_step_sse = _mm_set1_ps(dy_step_val);
        const __m128 v_idx_step4 = _mm_setr_ps(0, 1, 2, 3);
        const __m128 v_ones_sse = _mm_set1_ps(1.0f);
        const __m128 v_zeros_sse = _mm_setzero_ps();
        const __m128 v_srcW_limit_sse = _mm_set1_ps((float)src.width - 0.001f);
        const __m128 v_srcH_limit_sse = _mm_set1_ps((float)src.height - 0.001f);
        const __m128i v_max_w_idx_sse = _mm_set1_epi32(src.width - 1);
        const __m128i v_max_h_idx_sse = _mm_set1_epi32(src.height - 1);
        const __m128i v_zero_idx_sse = _mm_setzero_si128();
        const __m128i v_border_color_sse = _mm_set1_epi32(0x00FFFFFF);
        const __m128 v_255_sse = _mm_set1_ps(255.0f);
        const __m128 v_inv255_sse = _mm_set1_ps(1.0f / 255.0f);

        const int src_stride = src.width;
        const float srcCenterX = src.width * 0.5f;
        const float srcCenterY = src.height * 0.5f;

        for (int y = startY; y < endY; ++y) {
            float relY = (y - centerY);
            float startRelX = (startX - centerX);

            float currentSrcX = (startRelX * cosA + relY * sinA) * invScale + srcCenterX;
            float currentSrcY = (-startRelX * sinA + relY * cosA) * invScale + srcCenterY;

            Color* destRow = dest.getRow(y);
            int x = startX;

            for (; x <= endX - 8; x += 8) {
                __m256 v_sx = _mm256_fmadd_ps(v_idx_step8, v_dx_step, _mm256_set1_ps(currentSrcX));
                __m256 v_sy = _mm256_fmadd_ps(v_idx_step8, v_dy_step, _mm256_set1_ps(currentSrcY));

                currentSrcX += dx_step_val * 8.0f;
                currentSrcY += dy_step_val * 8.0f;

                __m256 v_mask = _mm256_and_ps(_mm256_cmp_ps(v_sx, _mm256_set1_ps(-1.0f), _CMP_GE_OQ),
                    _mm256_cmp_ps(v_sx, _mm256_add_ps(v_srcW_limit, v_ones), _CMP_LE_OQ));
                v_mask = _mm256_and_ps(v_mask, _mm256_cmp_ps(v_sy, _mm256_set1_ps(-1.0f), _CMP_GE_OQ));
                v_mask = _mm256_and_ps(v_mask, _mm256_cmp_ps(v_sy, _mm256_add_ps(v_srcH_limit, v_ones), _CMP_LE_OQ));

                if (_mm256_movemask_ps(v_mask) == 0) continue;

                __m256 v_safe_sx = _mm256_blendv_ps(v_zeros, v_sx, v_mask);
                __m256 v_safe_sy = _mm256_blendv_ps(v_zeros, v_sy, v_mask);

                __m256 v_flr_x = _mm256_floor_ps(v_safe_sx);
                __m256 v_flr_y = _mm256_floor_ps(v_safe_sy);
                __m256i v_idx_x = _mm256_cvtps_epi32(v_flr_x);
                __m256i v_idx_y = _mm256_cvtps_epi32(v_flr_y);

                __m256 v_dx = _mm256_sub_ps(v_safe_sx, v_flr_x);
                __m256 v_dy = _mm256_sub_ps(v_safe_sy, v_flr_y);
                __m256 v_inv_dx = _mm256_sub_ps(v_ones, v_dx);
                __m256 v_inv_dy = _mm256_sub_ps(v_ones, v_dy);

                __m256 v_w1 = _mm256_mul_ps(v_inv_dx, v_inv_dy);
                __m256 v_w2 = _mm256_mul_ps(v_dx, v_inv_dy);
                __m256 v_w3 = _mm256_mul_ps(v_inv_dx, v_dy);
                __m256 v_w4 = _mm256_mul_ps(v_dx, v_dy);

                auto safe_gather_avx2 = [&](__m256i ix, __m256i iy) {
                    __m256i mask_x = _mm256_and_si256(_mm256_cmpgt_epi32(ix, _mm256_set1_epi32(-1)),
                        _mm256_cmpgt_epi32(_mm256_set1_epi32(src.width), ix));
                    __m256i mask_y = _mm256_and_si256(_mm256_cmpgt_epi32(iy, _mm256_set1_epi32(-1)),
                        _mm256_cmpgt_epi32(_mm256_set1_epi32(src.height), iy));
                    __m256i valid_mask = _mm256_and_si256(mask_x, mask_y);

                    __m256i safe_x = _mm256_max_epi32(v_zero_idx, _mm256_min_epi32(ix, v_max_w_idx));
                    __m256i safe_y = _mm256_max_epi32(v_zero_idx, _mm256_min_epi32(iy, v_max_h_idx));

                    __m256i offset = _mm256_add_epi32(_mm256_mullo_epi32(safe_y, _mm256_set1_epi32(src_stride)), safe_x);
                    const int* sptr = reinterpret_cast<const int*>(src.color);
                    __m256i colors = _mm256_i32gather_epi32(sptr, offset, 4);

                    return _mm256_blendv_epi8(v_border_color, colors, valid_mask);
                    };

                __m256i c00 = safe_gather_avx2(v_idx_x, v_idx_y);
                __m256i c10 = safe_gather_avx2(_mm256_add_epi32(v_idx_x, _mm256_set1_epi32(1)), v_idx_y);
                __m256i c01 = safe_gather_avx2(v_idx_x, _mm256_add_epi32(v_idx_y, _mm256_set1_epi32(1)));
                __m256i c11 = safe_gather_avx2(_mm256_add_epi32(v_idx_x, _mm256_set1_epi32(1)),
                    _mm256_add_epi32(v_idx_y, _mm256_set1_epi32(1)));

                __m256 s_r, s_g, s_b, s_a;
                {
                    __m256 r0, g0, b0, a0; unpack_colors_avx2(c00, r0, g0, b0, a0);
                    __m256 r1, g1, b1, a1; unpack_colors_avx2(c10, r1, g1, b1, a1);
                    __m256 r2, g2, b2, a2; unpack_colors_avx2(c01, r2, g2, b2, a2);
                    __m256 r3, g3, b3, a3; unpack_colors_avx2(c11, r3, g3, b3, a3);

                    s_r = _mm256_fmadd_ps(r0, v_w1, _mm256_fmadd_ps(r1, v_w2,
                        _mm256_fmadd_ps(r2, v_w3, _mm256_mul_ps(r3, v_w4))));
                    s_g = _mm256_fmadd_ps(g0, v_w1, _mm256_fmadd_ps(g1, v_w2,
                        _mm256_fmadd_ps(g2, v_w3, _mm256_mul_ps(g3, v_w4))));
                    s_b = _mm256_fmadd_ps(b0, v_w1, _mm256_fmadd_ps(b1, v_w2,
                        _mm256_fmadd_ps(b2, v_w3, _mm256_mul_ps(b3, v_w4))));
                    s_a = _mm256_fmadd_ps(a0, v_w1, _mm256_fmadd_ps(a1, v_w2,
                        _mm256_fmadd_ps(a2, v_w3, _mm256_mul_ps(a3, v_w4))));
                }

                __m256i v_dst_raw = _mm256_loadu_si256(reinterpret_cast<__m256i*>(destRow + x));
                __m256 d_r, d_g, d_b, d_a;
                unpack_colors_avx2(v_dst_raw, d_r, d_g, d_b, d_a);

                __m256 v_sa_norm = _mm256_mul_ps(s_a, v_inv255);
                __m256 v_da_norm = _mm256_mul_ps(d_a, v_inv255);
                __m256 v_out_a = _mm256_add_ps(v_sa_norm,
                    _mm256_mul_ps(v_da_norm, _mm256_sub_ps(v_ones, v_sa_norm)));

                __m256 v_sa_factor = _mm256_div_ps(v_sa_norm,
                    _mm256_max_ps(v_out_a, _mm256_set1_ps(0.0001f)));
                __m256 v_da_factor = _mm256_sub_ps(v_ones, v_sa_factor);

                __m256 is_transparent = _mm256_cmp_ps(v_out_a,
                    _mm256_set1_ps(0.0001f), _CMP_LT_OQ);
                v_sa_factor = _mm256_blendv_ps(v_sa_factor, v_zeros, is_transparent);
                v_da_factor = _mm256_blendv_ps(v_da_factor, v_zeros, is_transparent);

                __m256 out_r = _mm256_fmadd_ps(s_r, v_sa_factor, _mm256_mul_ps(d_r, v_da_factor));
                __m256 out_g = _mm256_fmadd_ps(s_g, v_sa_factor, _mm256_mul_ps(d_g, v_da_factor));
                __m256 out_b = _mm256_fmadd_ps(s_b, v_sa_factor, _mm256_mul_ps(d_b, v_da_factor));
                __m256 out_a_scaled = _mm256_mul_ps(v_out_a, v_255);

                __m256i v_result = pack_colors_avx2(out_r, out_g, out_b, out_a_scaled);
                __m256i v_mask_i = _mm256_castps_si256(v_mask);
                __m256i v_final = _mm256_blendv_epi8(v_dst_raw, v_result, v_mask_i);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(destRow + x), v_final);
            }

            for (; x <= endX - 4; x += 4) {
                __m128 v_sx = _mm_fmadd_ps(v_idx_step4, v_dx_step_sse, _mm_set1_ps(currentSrcX));
                __m128 v_sy = _mm_fmadd_ps(v_idx_step4, v_dy_step_sse, _mm_set1_ps(currentSrcY));

                currentSrcX += dx_step_val * 4.0f;
                currentSrcY += dy_step_val * 4.0f;

                __m128 v_mask = _mm_and_ps(_mm_cmpge_ps(v_sx, _mm_set1_ps(-1.0f)),
                    _mm_cmple_ps(v_sx, _mm_add_ps(v_srcW_limit_sse, v_ones_sse)));
                v_mask = _mm_and_ps(v_mask, _mm_cmpge_ps(v_sy, _mm_set1_ps(-1.0f)));
                v_mask = _mm_and_ps(v_mask, _mm_cmple_ps(v_sy, _mm_add_ps(v_srcH_limit_sse, v_ones_sse)));

                if (_mm_movemask_ps(v_mask) == 0) continue;

                __m128 v_safe_sx = _mm_blendv_ps(v_zeros_sse, v_sx, v_mask);
                __m128 v_safe_sy = _mm_blendv_ps(v_zeros_sse, v_sy, v_mask);

                __m128 v_flr_x = _mm_floor_ps(v_safe_sx);
                __m128 v_flr_y = _mm_floor_ps(v_safe_sy);
                __m128i v_idx_x = _mm_cvtps_epi32(v_flr_x);
                __m128i v_idx_y = _mm_cvtps_epi32(v_flr_y);

                __m128 v_dx = _mm_sub_ps(v_safe_sx, v_flr_x);
                __m128 v_dy = _mm_sub_ps(v_safe_sy, v_flr_y);
                __m128 v_inv_dx = _mm_sub_ps(v_ones_sse, v_dx);
                __m128 v_inv_dy = _mm_sub_ps(v_ones_sse, v_dy);

                __m128 v_w1 = _mm_mul_ps(v_inv_dx, v_inv_dy);
                __m128 v_w2 = _mm_mul_ps(v_dx, v_inv_dy);
                __m128 v_w3 = _mm_mul_ps(v_inv_dx, v_dy);
                __m128 v_w4 = _mm_mul_ps(v_dx, v_dy);

                auto safe_gather_sse = [&](__m128i ix, __m128i iy) {
                    __m128i mask_x = _mm_and_si128(_mm_cmpgt_epi32(ix, _mm_set1_epi32(-1)),
                        _mm_cmpgt_epi32(_mm_set1_epi32(src.width), ix));
                    __m128i mask_y = _mm_and_si128(_mm_cmpgt_epi32(iy, _mm_set1_epi32(-1)),
                        _mm_cmpgt_epi32(_mm_set1_epi32(src.height), iy));
                    __m128i valid_mask = _mm_and_si128(mask_x, mask_y);

                    __m128i safe_x = _mm_max_epi32(v_zero_idx_sse, _mm_min_epi32(ix, v_max_w_idx_sse));
                    __m128i safe_y = _mm_max_epi32(v_zero_idx_sse, _mm_min_epi32(iy, v_max_h_idx_sse));

                    alignas(16) int idx_x_buf[4], idx_y_buf[4];
                    _mm_store_si128(reinterpret_cast<__m128i*>(idx_x_buf), safe_x);
                    _mm_store_si128(reinterpret_cast<__m128i*>(idx_y_buf), safe_y);

                    int c0 = reinterpret_cast<const int*>(src.color)[idx_y_buf[0] * src_stride + idx_x_buf[0]];
                    int c1 = reinterpret_cast<const int*>(src.color)[idx_y_buf[1] * src_stride + idx_x_buf[1]];
                    int c2 = reinterpret_cast<const int*>(src.color)[idx_y_buf[2] * src_stride + idx_x_buf[2]];
                    int c3 = reinterpret_cast<const int*>(src.color)[idx_y_buf[3] * src_stride + idx_x_buf[3]];

                    __m128i colors = _mm_set_epi32(c3, c2, c1, c0);
                    return _mm_blendv_epi8(v_border_color_sse, colors, valid_mask);
                    };

                __m128i c00 = safe_gather_sse(v_idx_x, v_idx_y);
                __m128i c10 = safe_gather_sse(_mm_add_epi32(v_idx_x, _mm_set1_epi32(1)), v_idx_y);
                __m128i c01 = safe_gather_sse(v_idx_x, _mm_add_epi32(v_idx_y, _mm_set1_epi32(1)));
                __m128i c11 = safe_gather_sse(_mm_add_epi32(v_idx_x, _mm_set1_epi32(1)),
                    _mm_add_epi32(v_idx_y, _mm_set1_epi32(1)));

                __m128 s_r, s_g, s_b, s_a;
                {
                    __m128 r0, g0, b0, a0; unpack_colors_sse(c00, r0, g0, b0, a0);
                    __m128 r1, g1, b1, a1; unpack_colors_sse(c10, r1, g1, b1, a1);
                    __m128 r2, g2, b2, a2; unpack_colors_sse(c01, r2, g2, b2, a2);
                    __m128 r3, g3, b3, a3; unpack_colors_sse(c11, r3, g3, b3, a3);

                    s_r = _mm_fmadd_ps(r0, v_w1, _mm_fmadd_ps(r1, v_w2,
                        _mm_fmadd_ps(r2, v_w3, _mm_mul_ps(r3, v_w4))));
                    s_g = _mm_fmadd_ps(g0, v_w1, _mm_fmadd_ps(g1, v_w2,
                        _mm_fmadd_ps(g2, v_w3, _mm_mul_ps(g3, v_w4))));
                    s_b = _mm_fmadd_ps(b0, v_w1, _mm_fmadd_ps(b1, v_w2,
                        _mm_fmadd_ps(b2, v_w3, _mm_mul_ps(b3, v_w4))));
                    s_a = _mm_fmadd_ps(a0, v_w1, _mm_fmadd_ps(a1, v_w2,
                        _mm_fmadd_ps(a2, v_w3, _mm_mul_ps(a3, v_w4))));
                }

                __m128i v_dst_raw = _mm_loadu_si128(reinterpret_cast<__m128i*>(destRow + x));
                __m128 d_r, d_g, d_b, d_a;
                unpack_colors_sse(v_dst_raw, d_r, d_g, d_b, d_a);

                __m128 v_sa_norm = _mm_mul_ps(s_a, v_inv255_sse);
                __m128 v_da_norm = _mm_mul_ps(d_a, v_inv255_sse);
                __m128 v_out_a = _mm_add_ps(v_sa_norm,
                    _mm_mul_ps(v_da_norm, _mm_sub_ps(v_ones_sse, v_sa_norm)));

                __m128 v_sa_factor = _mm_div_ps(v_sa_norm,
                    _mm_max_ps(v_out_a, _mm_set1_ps(0.0001f)));
                __m128 v_da_factor = _mm_sub_ps(v_ones_sse, v_sa_factor);

                __m128 is_transparent = _mm_cmplt_ps(v_out_a, _mm_set1_ps(0.0001f));
                v_sa_factor = _mm_blendv_ps(v_sa_factor, v_zeros_sse, is_transparent);
                v_da_factor = _mm_blendv_ps(v_da_factor, v_zeros_sse, is_transparent);

                __m128 out_r = _mm_fmadd_ps(s_r, v_sa_factor, _mm_mul_ps(d_r, v_da_factor));
                __m128 out_g = _mm_fmadd_ps(s_g, v_sa_factor, _mm_mul_ps(d_g, v_da_factor));
                __m128 out_b = _mm_fmadd_ps(s_b, v_sa_factor, _mm_mul_ps(d_b, v_da_factor));
                __m128 out_a_scaled = _mm_mul_ps(v_out_a, v_255_sse);

                __m128i v_result = pack_colors_sse(out_r, out_g, out_b, out_a_scaled);
                __m128i v_mask_i = _mm_castps_si128(v_mask);
                __m128i v_final = _mm_blendv_epi8(v_dst_raw, v_result, v_mask_i);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(destRow + x), v_final);
            }

            for (; x < endX; ++x) {
                const float relX = (x - centerX);
                const float srcX = (relX * cosA + relY * sinA) * invScale + srcCenterX;
                const float srcY = (-relX * sinA + relY * cosA) * invScale + srcCenterY;

                if (srcX > -1.0f && srcX < src.width && srcY > -1.0f && srcY < src.height) {
                    int sx = static_cast<int>(std::floor(srcX));
                    int sy = static_cast<int>(std::floor(srcY));
                    float dx = srcX - sx;
                    float dy = srcY - sy;

                    auto getPixelSafe = [&](int px, int py) -> Color {
                        if (px >= 0 && px < src.width && py >= 0 && py < src.height) {
                            return src.at(px, py);
                        }
                        return { 255, 255, 255, 0 };
                        };

                    Color c00 = getPixelSafe(sx, sy);
                    Color c10 = getPixelSafe(sx + 1, sy);
                    Color c01 = getPixelSafe(sx, sy + 1);
                    Color c11 = getPixelSafe(sx + 1, sy + 1);

                    float w1 = (1.0f - dx) * (1.0f - dy);
                    float w2 = dx * (1.0f - dy);
                    float w3 = (1.0f - dx) * dy;
                    float w4 = dx * dy;

                    float r = c00.r * w1 + c10.r * w2 + c01.r * w3 + c11.r * w4;
                    float g = c00.g * w1 + c10.g * w2 + c01.g * w3 + c11.g * w4;
                    float b = c00.b * w1 + c10.b * w2 + c01.b * w3 + c11.b * w4;
                    float a = c00.a * w1 + c10.a * w2 + c01.a * w3 + c11.a * w4;

                    Color& dstC = destRow[x];
                    float sa = a / 255.0f;
                    float da = dstC.a / 255.0f;
                    float out_a = sa + da * (1.0f - sa);

                    if (out_a > 0.0001f) {
                        float sa_factor = sa / out_a;
                        float da_factor = 1.0f - sa_factor;

                        dstC.r = static_cast<uint8_t>(std::min(255.0f, r * sa_factor + dstC.r * da_factor));
                        dstC.g = static_cast<uint8_t>(std::min(255.0f, g * sa_factor + dstC.g * da_factor));
                        dstC.b = static_cast<uint8_t>(std::min(255.0f, b * sa_factor + dstC.b * da_factor));
                    }
                    dstC.a = static_cast<uint8_t>(std::min(255.0f, out_a * 255.0f));
                }
            }
        }
    }

    void drawTransformed(Buffer& dest, const Buffer& src, float centerX, float centerY, float scaleX, float scaleY, float rotation) {
        if (!dest.isValid() || !src.isValid() || scaleX <= 0.0f || scaleY <= 0.0f) return;

        float rad = std::fmod(rotation, 360.0f);
        if (rad < 0) rad += 360.0f;

        const float EPSILON = 0.001f;

        if (std::abs(rad) < EPSILON || std::abs(rad) > 360.0f - EPSILON) {
            drawScaled(dest, src, centerX, centerY, scaleX, scaleY);
            return;
        }
        else if (std::abs(rad - 90.0f) < EPSILON) {
            drawRotated90Scaled(dest, src, centerX, centerY, scaleX, scaleY);
            return;
        }
        else if (std::abs(rad - 180.0f) < EPSILON) {
            drawRotated180Scaled(dest, src, centerX, centerY, scaleX, scaleY);
            return;
        }
        else if (std::abs(rad - 270.0f) < EPSILON) {
            drawRotated270Scaled(dest, src, centerX, centerY, scaleX, scaleY);
            return;
        }

        const float cosA = std::cos(rad);
        const float sinA = std::sin(rad);

        const float halfW = src.width * scaleX * 0.5f;
        const float halfH = src.height * scaleY * 0.5f;

        const float boundX = std::abs(halfW * cosA) + std::abs(halfH * sinA);
        const float boundY = std::abs(halfW * sinA) + std::abs(halfH * cosA);

        int startX = std::max(0, static_cast<int>(centerX - boundX));
        int startY = std::max(0, static_cast<int>(centerY - boundY));
        int endX = std::min(dest.width, static_cast<int>(centerX + boundX) + 1);
        int endY = std::min(dest.height, static_cast<int>(centerY + boundY) + 1);

        if (startX >= endX || startY >= endY) return;

        const float invScaleX = 1.0f / scaleX;
        const float invScaleY = 1.0f / scaleY;

        const float dx_step_src_x = cosA * invScaleX;
        const float dx_step_src_y = -sinA * invScaleY;

        const __m256 v_dx_step_x = _mm256_set1_ps(dx_step_src_x);
        const __m256 v_dx_step_y = _mm256_set1_ps(dx_step_src_y);
        const __m256 v_ones = _mm256_set1_ps(1.0f);
        const __m256 v_zeros = _mm256_setzero_ps();
        const __m256 v_idx_step8 = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);

        const __m256 v_srcW_limit = _mm256_set1_ps((float)src.width - 0.001f);
        const __m256 v_srcH_limit = _mm256_set1_ps((float)src.height - 0.001f);

        const __m256i v_max_w_idx = _mm256_set1_epi32(src.width - 1);
        const __m256i v_max_h_idx = _mm256_set1_epi32(src.height - 1);
        const __m256i v_zero_idx = _mm256_setzero_si256();
        const __m256i v_border_color = _mm256_set1_epi32(0x00FFFFFF);

        const __m256 v_255 = _mm256_set1_ps(255.0f);
        const __m256 v_inv255 = _mm256_set1_ps(1.0f / 255.0f);

        const __m128 v_dx_step_x_sse = _mm_set1_ps(dx_step_src_x);
        const __m128 v_dx_step_y_sse = _mm_set1_ps(dx_step_src_y);
        const __m128 v_idx_step4 = _mm_setr_ps(0, 1, 2, 3);
        const __m128 v_ones_sse = _mm_set1_ps(1.0f);
        const __m128 v_zeros_sse = _mm_setzero_ps();
        const __m128 v_srcW_limit_sse = _mm_set1_ps((float)src.width - 0.001f);
        const __m128 v_srcH_limit_sse = _mm_set1_ps((float)src.height - 0.001f);
        const __m128i v_max_w_idx_sse = _mm_set1_epi32(src.width - 1);
        const __m128i v_max_h_idx_sse = _mm_set1_epi32(src.height - 1);
        const __m128i v_zero_idx_sse = _mm_setzero_si128();
        const __m128i v_border_color_sse = _mm_set1_epi32(0x00FFFFFF);
        const __m128 v_255_sse = _mm_set1_ps(255.0f);
        const __m128 v_inv255_sse = _mm_set1_ps(1.0f / 255.0f);

        const int src_stride = src.width;
        const float srcCenterX = src.width * 0.5f;
        const float srcCenterY = src.height * 0.5f;

        for (int y = startY; y < endY; ++y) {
            float relY = (y - centerY);
            float startRelX = (startX - centerX);

            float currentSrcX = (startRelX * cosA + relY * sinA) * invScaleX + srcCenterX;
            float currentSrcY = (-startRelX * sinA + relY * cosA) * invScaleY + srcCenterY;

            Color* destRow = dest.getRow(y);
            int x = startX;

            for (; x <= endX - 8; x += 8) {
                __m256 v_sx = _mm256_fmadd_ps(v_idx_step8, v_dx_step_x, _mm256_set1_ps(currentSrcX));
                __m256 v_sy = _mm256_fmadd_ps(v_idx_step8, v_dx_step_y, _mm256_set1_ps(currentSrcY));

                currentSrcX += dx_step_src_x * 8.0f;
                currentSrcY += dx_step_src_y * 8.0f;

                __m256 v_mask = _mm256_and_ps(_mm256_cmp_ps(v_sx, _mm256_set1_ps(-1.0f), _CMP_GE_OQ),
                    _mm256_cmp_ps(v_sx, _mm256_add_ps(v_srcW_limit, v_ones), _CMP_LE_OQ));
                v_mask = _mm256_and_ps(v_mask, _mm256_cmp_ps(v_sy, _mm256_set1_ps(-1.0f), _CMP_GE_OQ));
                v_mask = _mm256_and_ps(v_mask, _mm256_cmp_ps(v_sy, _mm256_add_ps(v_srcH_limit, v_ones), _CMP_LE_OQ));

                if (_mm256_movemask_ps(v_mask) == 0) continue;

                __m256 v_safe_sx = _mm256_blendv_ps(v_zeros, v_sx, v_mask);
                __m256 v_safe_sy = _mm256_blendv_ps(v_zeros, v_sy, v_mask);

                __m256 v_flr_x = _mm256_floor_ps(v_safe_sx);
                __m256 v_flr_y = _mm256_floor_ps(v_safe_sy);
                __m256i v_idx_x = _mm256_cvtps_epi32(v_flr_x);
                __m256i v_idx_y = _mm256_cvtps_epi32(v_flr_y);

                __m256 v_dx = _mm256_sub_ps(v_safe_sx, v_flr_x);
                __m256 v_dy = _mm256_sub_ps(v_safe_sy, v_flr_y);
                __m256 v_inv_dx = _mm256_sub_ps(v_ones, v_dx);
                __m256 v_inv_dy = _mm256_sub_ps(v_ones, v_dy);

                __m256 v_w1 = _mm256_mul_ps(v_inv_dx, v_inv_dy);
                __m256 v_w2 = _mm256_mul_ps(v_dx, v_inv_dy);
                __m256 v_w3 = _mm256_mul_ps(v_inv_dx, v_dy);
                __m256 v_w4 = _mm256_mul_ps(v_dx, v_dy);

                auto safe_gather_avx2 = [&](__m256i ix, __m256i iy) {
                    __m256i mask_x = _mm256_and_si256(_mm256_cmpgt_epi32(ix, _mm256_set1_epi32(-1)),
                        _mm256_cmpgt_epi32(_mm256_set1_epi32(src.width), ix));
                    __m256i mask_y = _mm256_and_si256(_mm256_cmpgt_epi32(iy, _mm256_set1_epi32(-1)),
                        _mm256_cmpgt_epi32(_mm256_set1_epi32(src.height), iy));
                    __m256i valid_mask = _mm256_and_si256(mask_x, mask_y);

                    __m256i safe_x = _mm256_max_epi32(v_zero_idx, _mm256_min_epi32(ix, v_max_w_idx));
                    __m256i safe_y = _mm256_max_epi32(v_zero_idx, _mm256_min_epi32(iy, v_max_h_idx));

                    __m256i offset = _mm256_add_epi32(_mm256_mullo_epi32(safe_y, _mm256_set1_epi32(src_stride)), safe_x);
                    const int* sptr = reinterpret_cast<const int*>(src.color);
                    __m256i colors = _mm256_i32gather_epi32(sptr, offset, 4);

                    return _mm256_blendv_epi8(v_border_color, colors, valid_mask);
                    };

                __m256i c00 = safe_gather_avx2(v_idx_x, v_idx_y);
                __m256i c10 = safe_gather_avx2(_mm256_add_epi32(v_idx_x, _mm256_set1_epi32(1)), v_idx_y);
                __m256i c01 = safe_gather_avx2(v_idx_x, _mm256_add_epi32(v_idx_y, _mm256_set1_epi32(1)));
                __m256i c11 = safe_gather_avx2(_mm256_add_epi32(v_idx_x, _mm256_set1_epi32(1)),
                    _mm256_add_epi32(v_idx_y, _mm256_set1_epi32(1)));

                __m256 s_r, s_g, s_b, s_a;
                {
                    __m256 r0, g0, b0, a0; unpack_colors_avx2(c00, r0, g0, b0, a0);
                    __m256 r1, g1, b1, a1; unpack_colors_avx2(c10, r1, g1, b1, a1);
                    __m256 r2, g2, b2, a2; unpack_colors_avx2(c01, r2, g2, b2, a2);
                    __m256 r3, g3, b3, a3; unpack_colors_avx2(c11, r3, g3, b3, a3);

                    s_r = _mm256_fmadd_ps(r0, v_w1, _mm256_fmadd_ps(r1, v_w2,
                        _mm256_fmadd_ps(r2, v_w3, _mm256_mul_ps(r3, v_w4))));
                    s_g = _mm256_fmadd_ps(g0, v_w1, _mm256_fmadd_ps(g1, v_w2,
                        _mm256_fmadd_ps(g2, v_w3, _mm256_mul_ps(g3, v_w4))));
                    s_b = _mm256_fmadd_ps(b0, v_w1, _mm256_fmadd_ps(b1, v_w2,
                        _mm256_fmadd_ps(b2, v_w3, _mm256_mul_ps(b3, v_w4))));
                    s_a = _mm256_fmadd_ps(a0, v_w1, _mm256_fmadd_ps(a1, v_w2,
                        _mm256_fmadd_ps(a2, v_w3, _mm256_mul_ps(a3, v_w4))));
                }

                __m256i v_dst_raw = _mm256_loadu_si256(reinterpret_cast<__m256i*>(destRow + x));
                __m256 d_r, d_g, d_b, d_a;
                unpack_colors_avx2(v_dst_raw, d_r, d_g, d_b, d_a);

                __m256 v_sa_norm = _mm256_mul_ps(s_a, v_inv255);
                __m256 v_da_norm = _mm256_mul_ps(d_a, v_inv255);
                __m256 v_out_a = _mm256_add_ps(v_sa_norm,
                    _mm256_mul_ps(v_da_norm, _mm256_sub_ps(v_ones, v_sa_norm)));

                __m256 v_sa_factor = _mm256_div_ps(v_sa_norm,
                    _mm256_max_ps(v_out_a, _mm256_set1_ps(0.0001f)));
                __m256 v_da_factor = _mm256_sub_ps(v_ones, v_sa_factor);

                __m256 is_transparent = _mm256_cmp_ps(v_out_a,
                    _mm256_set1_ps(0.0001f), _CMP_LT_OQ);
                v_sa_factor = _mm256_blendv_ps(v_sa_factor, v_zeros, is_transparent);
                v_da_factor = _mm256_blendv_ps(v_da_factor, v_zeros, is_transparent);

                __m256 out_r = _mm256_fmadd_ps(s_r, v_sa_factor, _mm256_mul_ps(d_r, v_da_factor));
                __m256 out_g = _mm256_fmadd_ps(s_g, v_sa_factor, _mm256_mul_ps(d_g, v_da_factor));
                __m256 out_b = _mm256_fmadd_ps(s_b, v_sa_factor, _mm256_mul_ps(d_b, v_da_factor));
                __m256 out_a_scaled = _mm256_mul_ps(v_out_a, v_255);

                __m256i v_result = pack_colors_avx2(out_r, out_g, out_b, out_a_scaled);
                __m256i v_mask_i = _mm256_castps_si256(v_mask);
                __m256i v_final = _mm256_blendv_epi8(v_dst_raw, v_result, v_mask_i);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(destRow + x), v_final);
            }

            for (; x <= endX - 4; x += 4) {
                __m128 v_sx = _mm_fmadd_ps(v_idx_step4, v_dx_step_x_sse, _mm_set1_ps(currentSrcX));
                __m128 v_sy = _mm_fmadd_ps(v_idx_step4, v_dx_step_y_sse, _mm_set1_ps(currentSrcY));

                currentSrcX += dx_step_src_x * 4.0f;
                currentSrcY += dx_step_src_y * 4.0f;

                __m128 v_mask = _mm_and_ps(_mm_cmpge_ps(v_sx, _mm_set1_ps(-1.0f)),
                    _mm_cmple_ps(v_sx, _mm_add_ps(v_srcW_limit_sse, v_ones_sse)));
                v_mask = _mm_and_ps(v_mask, _mm_cmpge_ps(v_sy, _mm_set1_ps(-1.0f)));
                v_mask = _mm_and_ps(v_mask, _mm_cmple_ps(v_sy, _mm_add_ps(v_srcH_limit_sse, v_ones_sse)));

                if (_mm_movemask_ps(v_mask) == 0) continue;

                __m128 v_safe_sx = _mm_blendv_ps(v_zeros_sse, v_sx, v_mask);
                __m128 v_safe_sy = _mm_blendv_ps(v_zeros_sse, v_sy, v_mask);

                __m128 v_flr_x = _mm_floor_ps(v_safe_sx);
                __m128 v_flr_y = _mm_floor_ps(v_safe_sy);
                __m128i v_idx_x = _mm_cvtps_epi32(v_flr_x);
                __m128i v_idx_y = _mm_cvtps_epi32(v_flr_y);

                __m128 v_dx = _mm_sub_ps(v_safe_sx, v_flr_x);
                __m128 v_dy = _mm_sub_ps(v_safe_sy, v_flr_y);
                __m128 v_inv_dx = _mm_sub_ps(v_ones_sse, v_dx);
                __m128 v_inv_dy = _mm_sub_ps(v_ones_sse, v_dy);

                __m128 v_w1 = _mm_mul_ps(v_inv_dx, v_inv_dy);
                __m128 v_w2 = _mm_mul_ps(v_dx, v_inv_dy);
                __m128 v_w3 = _mm_mul_ps(v_inv_dx, v_dy);
                __m128 v_w4 = _mm_mul_ps(v_dx, v_dy);

                auto safe_gather_sse = [&](__m128i ix, __m128i iy) {
                    __m128i mask_x = _mm_and_si128(_mm_cmpgt_epi32(ix, _mm_set1_epi32(-1)),
                        _mm_cmpgt_epi32(_mm_set1_epi32(src.width), ix));
                    __m128i mask_y = _mm_and_si128(_mm_cmpgt_epi32(iy, _mm_set1_epi32(-1)),
                        _mm_cmpgt_epi32(_mm_set1_epi32(src.height), iy));
                    __m128i valid_mask = _mm_and_si128(mask_x, mask_y);

                    __m128i safe_x = _mm_max_epi32(v_zero_idx_sse, _mm_min_epi32(ix, v_max_w_idx_sse));
                    __m128i safe_y = _mm_max_epi32(v_zero_idx_sse, _mm_min_epi32(iy, v_max_h_idx_sse));

                    alignas(16) int idx_x_buf[4], idx_y_buf[4];
                    _mm_store_si128(reinterpret_cast<__m128i*>(idx_x_buf), safe_x);
                    _mm_store_si128(reinterpret_cast<__m128i*>(idx_y_buf), safe_y);

                    int c0 = reinterpret_cast<const int*>(src.color)[idx_y_buf[0] * src_stride + idx_x_buf[0]];
                    int c1 = reinterpret_cast<const int*>(src.color)[idx_y_buf[1] * src_stride + idx_x_buf[1]];
                    int c2 = reinterpret_cast<const int*>(src.color)[idx_y_buf[2] * src_stride + idx_x_buf[2]];
                    int c3 = reinterpret_cast<const int*>(src.color)[idx_y_buf[3] * src_stride + idx_x_buf[3]];

                    __m128i colors = _mm_set_epi32(c3, c2, c1, c0);
                    return _mm_blendv_epi8(v_border_color_sse, colors, valid_mask);
                    };

                __m128i c00 = safe_gather_sse(v_idx_x, v_idx_y);
                __m128i c10 = safe_gather_sse(_mm_add_epi32(v_idx_x, _mm_set1_epi32(1)), v_idx_y);
                __m128i c01 = safe_gather_sse(v_idx_x, _mm_add_epi32(v_idx_y, _mm_set1_epi32(1)));
                __m128i c11 = safe_gather_sse(_mm_add_epi32(v_idx_x, _mm_set1_epi32(1)),
                    _mm_add_epi32(v_idx_y, _mm_set1_epi32(1)));

                __m128 s_r, s_g, s_b, s_a;
                {
                    __m128 r0, g0, b0, a0; unpack_colors_sse(c00, r0, g0, b0, a0);
                    __m128 r1, g1, b1, a1; unpack_colors_sse(c10, r1, g1, b1, a1);
                    __m128 r2, g2, b2, a2; unpack_colors_sse(c01, r2, g2, b2, a2);
                    __m128 r3, g3, b3, a3; unpack_colors_sse(c11, r3, g3, b3, a3);

                    s_r = _mm_fmadd_ps(r0, v_w1, _mm_fmadd_ps(r1, v_w2,
                        _mm_fmadd_ps(r2, v_w3, _mm_mul_ps(r3, v_w4))));
                    s_g = _mm_fmadd_ps(g0, v_w1, _mm_fmadd_ps(g1, v_w2,
                        _mm_fmadd_ps(g2, v_w3, _mm_mul_ps(g3, v_w4))));
                    s_b = _mm_fmadd_ps(b0, v_w1, _mm_fmadd_ps(b1, v_w2,
                        _mm_fmadd_ps(b2, v_w3, _mm_mul_ps(b3, v_w4))));
                    s_a = _mm_fmadd_ps(a0, v_w1, _mm_fmadd_ps(a1, v_w2,
                        _mm_fmadd_ps(a2, v_w3, _mm_mul_ps(a3, v_w4))));
                }

                __m128i v_dst_raw = _mm_loadu_si128(reinterpret_cast<__m128i*>(destRow + x));
                __m128 d_r, d_g, d_b, d_a;
                unpack_colors_sse(v_dst_raw, d_r, d_g, d_b, d_a);

                __m128 v_sa_norm = _mm_mul_ps(s_a, v_inv255_sse);
                __m128 v_da_norm = _mm_mul_ps(d_a, v_inv255_sse);
                __m128 v_out_a = _mm_add_ps(v_sa_norm,
                    _mm_mul_ps(v_da_norm, _mm_sub_ps(v_ones_sse, v_sa_norm)));

                __m128 v_sa_factor = _mm_div_ps(v_sa_norm,
                    _mm_max_ps(v_out_a, _mm_set1_ps(0.0001f)));
                __m128 v_da_factor = _mm_sub_ps(v_ones_sse, v_sa_factor);

                __m128 is_transparent = _mm_cmplt_ps(v_out_a, _mm_set1_ps(0.0001f));
                v_sa_factor = _mm_blendv_ps(v_sa_factor, v_zeros_sse, is_transparent);
                v_da_factor = _mm_blendv_ps(v_da_factor, v_zeros_sse, is_transparent);

                __m128 out_r = _mm_fmadd_ps(s_r, v_sa_factor, _mm_mul_ps(d_r, v_da_factor));
                __m128 out_g = _mm_fmadd_ps(s_g, v_sa_factor, _mm_mul_ps(d_g, v_da_factor));
                __m128 out_b = _mm_fmadd_ps(s_b, v_sa_factor, _mm_mul_ps(d_b, v_da_factor));
                __m128 out_a_scaled = _mm_mul_ps(v_out_a, v_255_sse);

                __m128i v_result = pack_colors_sse(out_r, out_g, out_b, out_a_scaled);
                __m128i v_mask_i = _mm_castps_si128(v_mask);
                __m128i v_final = _mm_blendv_epi8(v_dst_raw, v_result, v_mask_i);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(destRow + x), v_final);
            }

            for (; x < endX; ++x) {
                const float relX = (x - centerX);
                const float srcX = (relX * cosA + relY * sinA) * invScaleX + srcCenterX;
                const float srcY = (-relX * sinA + relY * cosA) * invScaleY + srcCenterY;

                if (srcX > -1.0f && srcX < src.width && srcY > -1.0f && srcY < src.height) {
                    int sx = static_cast<int>(std::floor(srcX));
                    int sy = static_cast<int>(std::floor(srcY));
                    float dx = srcX - sx;
                    float dy = srcY - sy;

                    auto getPixelSafe = [&](int px, int py) -> Color {
                        if (px >= 0 && px < src.width && py >= 0 && py < src.height) {
                            return src.at(px, py);
                        }
                        return { 255, 255, 255, 0 };
                        };

                    Color c00 = getPixelSafe(sx, sy);
                    Color c10 = getPixelSafe(sx + 1, sy);
                    Color c01 = getPixelSafe(sx, sy + 1);
                    Color c11 = getPixelSafe(sx + 1, sy + 1);

                    float w1 = (1.0f - dx) * (1.0f - dy);
                    float w2 = dx * (1.0f - dy);
                    float w3 = (1.0f - dx) * dy;
                    float w4 = dx * dy;

                    float r = c00.r * w1 + c10.r * w2 + c01.r * w3 + c11.r * w4;
                    float g = c00.g * w1 + c10.g * w2 + c01.g * w3 + c11.g * w4;
                    float b = c00.b * w1 + c10.b * w2 + c01.b * w3 + c11.b * w4;
                    float a = c00.a * w1 + c10.a * w2 + c01.a * w3 + c11.a * w4;

                    Color& dstC = destRow[x];
                    float sa = a / 255.0f;
                    float da = dstC.a / 255.0f;
                    float out_a = sa + da * (1.0f - sa);

                    if (out_a > 0.0001f) {
                        float sa_factor = sa / out_a;
                        float da_factor = 1.0f - sa_factor;

                        dstC.r = static_cast<uint8_t>(std::min(255.0f, r * sa_factor + dstC.r * da_factor));
                        dstC.g = static_cast<uint8_t>(std::min(255.0f, g * sa_factor + dstC.g * da_factor));
                        dstC.b = static_cast<uint8_t>(std::min(255.0f, b * sa_factor + dstC.b * da_factor));
                    }
                    dstC.a = static_cast<uint8_t>(std::min(255.0f, out_a * 255.0f));
                }
            }
        }
    }

    // ============================================================================
    // 快速通道旋转实现
    // ============================================================================

    void drawRotated180Scaled(Buffer& dest, const Buffer& src, float centerX, float centerY, float scaleX, float scaleY) {
        if (!dest.isValid() || !src.isValid() || scaleX <= 0.0f || scaleY <= 0.0f) return;

        const float invScaleX = 1.0f / scaleX;
        const float invScaleY = 1.0f / scaleY;
        const float srcCenterX = src.width * 0.5f;
        const float srcCenterY = src.height * 0.5f;

        const float halfW = src.width * scaleX * 0.5f;
        const float halfH = src.height * scaleY * 0.5f;

        int startX = std::max(0, static_cast<int>(centerX - halfW));
        int startY = std::max(0, static_cast<int>(centerY - halfH));
        int endX = std::min(dest.width, static_cast<int>(centerX + halfW) + 1);
        int endY = std::min(dest.height, static_cast<int>(centerY + halfH) + 1);

        if (startX >= endX || startY >= endY) return;

        const float srcW_limit = (float)src.width - 1.05f;
        const float srcH_limit = (float)src.height - 1.05f;

        const __m256 v_dx_step = _mm256_set1_ps(-invScaleX);
        const __m256 v_idx_step8 = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256 v_ones = _mm256_set1_ps(1.0f);
        const __m256 v_srcW_limit = _mm256_set1_ps(srcW_limit);
        const __m256 v_255 = _mm256_set1_ps(255.0f);
        const __m256 v_inv255 = _mm256_set1_ps(1.0f / 255.0f);
        const int src_stride = src.width;

        const __m128 v_dx_step_sse = _mm_set1_ps(-invScaleX);
        const __m128 v_idx_step4 = _mm_setr_ps(0, 1, 2, 3);
        const __m128 v_ones_sse = _mm_set1_ps(1.0f);
        const __m128 v_srcW_limit_sse = _mm_set1_ps(srcW_limit);
        const __m128 v_255_sse = _mm_set1_ps(255.0f);
        const __m128 v_inv255_sse = _mm_set1_ps(1.0f / 255.0f);

        for (int y = startY; y < endY; ++y) {
            float srcY = srcCenterY + (centerY - y) * invScaleY;

            if (srcY < 0.0f || srcY > srcH_limit) continue;

            int sy0 = static_cast<int>(srcY);
            float dy = srcY - sy0;
            int sy1 = sy0 + 1;

            __m256 v_dy = _mm256_set1_ps(dy);
            __m256 v_inv_dy = _mm256_sub_ps(v_ones, v_dy);

            __m128 v_dy_sse = _mm_set1_ps(dy);
            __m128 v_inv_dy_sse = _mm_sub_ps(v_ones_sse, v_dy_sse);

            Color* destRow = dest.getRow(y);
            const int* src_ptr = reinterpret_cast<const int*>(src.color);

            float startSrcX = srcCenterX + (centerX - startX) * invScaleX;

            int x = startX;

            for (; x <= endX - 8; x += 8) {
                __m256 v_sx = _mm256_fmadd_ps(v_idx_step8, v_dx_step, _mm256_set1_ps(startSrcX));
                startSrcX -= invScaleX * 8.0f;

                __m256 v_mask = _mm256_and_ps(_mm256_cmp_ps(v_sx, v_ones, _CMP_GE_OQ), _mm256_cmp_ps(v_sx, v_srcW_limit, _CMP_LE_OQ));
                if (_mm256_movemask_ps(v_mask) == 0) continue;

                __m256 v_safe_sx = _mm256_blendv_ps(v_ones, v_sx, v_mask);
                __m256 v_flr_x = _mm256_floor_ps(v_safe_sx);
                __m256i v_idx_x = _mm256_cvtps_epi32(v_flr_x);

                __m256 v_dx = _mm256_sub_ps(v_safe_sx, v_flr_x);
                __m256 v_inv_dx = _mm256_sub_ps(v_ones, v_dx);

                __m256 v_w1 = _mm256_mul_ps(v_inv_dx, v_inv_dy);
                __m256 v_w2 = _mm256_mul_ps(v_dx, v_inv_dy);
                __m256 v_w3 = _mm256_mul_ps(v_inv_dx, v_dy);
                __m256 v_w4 = _mm256_mul_ps(v_dx, v_dy);

                __m256i v_offset0 = _mm256_add_epi32(_mm256_set1_epi32(sy0 * src_stride), v_idx_x);
                __m256i v_offset1 = _mm256_add_epi32(_mm256_set1_epi32(sy1 * src_stride), v_idx_x);

                __m256i c00 = _mm256_i32gather_epi32(src_ptr, v_offset0, 4);
                __m256i c10 = _mm256_i32gather_epi32(src_ptr, _mm256_add_epi32(v_offset0, _mm256_set1_epi32(1)), 4);
                __m256i c01 = _mm256_i32gather_epi32(src_ptr, v_offset1, 4);
                __m256i c11 = _mm256_i32gather_epi32(src_ptr, _mm256_add_epi32(v_offset1, _mm256_set1_epi32(1)), 4);

                __m256 s_r, s_g, s_b, s_a;
                {
                    __m256 r0, g0, b0, a0; unpack_colors_avx2(c00, r0, g0, b0, a0);
                    __m256 r1, g1, b1, a1; unpack_colors_avx2(c10, r1, g1, b1, a1);
                    __m256 r2, g2, b2, a2; unpack_colors_avx2(c01, r2, g2, b2, a2);
                    __m256 r3, g3, b3, a3; unpack_colors_avx2(c11, r3, g3, b3, a3);
                    s_r = _mm256_fmadd_ps(r0, v_w1, _mm256_fmadd_ps(r1, v_w2, _mm256_fmadd_ps(r2, v_w3, _mm256_mul_ps(r3, v_w4))));
                    s_g = _mm256_fmadd_ps(g0, v_w1, _mm256_fmadd_ps(g1, v_w2, _mm256_fmadd_ps(g2, v_w3, _mm256_mul_ps(g3, v_w4))));
                    s_b = _mm256_fmadd_ps(b0, v_w1, _mm256_fmadd_ps(b1, v_w2, _mm256_fmadd_ps(b2, v_w3, _mm256_mul_ps(b3, v_w4))));
                    s_a = _mm256_fmadd_ps(a0, v_w1, _mm256_fmadd_ps(a1, v_w2, _mm256_fmadd_ps(a2, v_w3, _mm256_mul_ps(a3, v_w4))));
                }

                __m256i v_dst_raw = _mm256_loadu_si256(reinterpret_cast<__m256i*>(destRow + x));
                __m256 d_r, d_g, d_b, d_a;
                unpack_colors_avx2(v_dst_raw, d_r, d_g, d_b, d_a);

                __m256 v_sa_norm = _mm256_mul_ps(s_a, v_inv255);
                __m256 v_da_norm = _mm256_mul_ps(d_a, v_inv255);
                __m256 v_out_a = _mm256_add_ps(v_sa_norm, _mm256_mul_ps(v_da_norm, _mm256_sub_ps(v_ones, v_sa_norm)));
                __m256 v_sa_factor = _mm256_div_ps(v_sa_norm, _mm256_max_ps(v_out_a, _mm256_set1_ps(0.001f)));
                __m256 v_da_factor = _mm256_sub_ps(v_ones, v_sa_factor);

                __m256 out_r = _mm256_fmadd_ps(s_r, v_sa_factor, _mm256_mul_ps(d_r, v_da_factor));
                __m256 out_g = _mm256_fmadd_ps(s_g, v_sa_factor, _mm256_mul_ps(d_g, v_da_factor));
                __m256 out_b = _mm256_fmadd_ps(s_b, v_sa_factor, _mm256_mul_ps(d_b, v_da_factor));
                __m256 out_a_scaled = _mm256_mul_ps(v_out_a, v_255);

                out_r = _mm256_min_ps(_mm256_max_ps(out_r, v_ones), v_255);
                out_g = _mm256_min_ps(_mm256_max_ps(out_g, v_ones), v_255);
                out_b = _mm256_min_ps(_mm256_max_ps(out_b, v_ones), v_255);
                out_a_scaled = _mm256_min_ps(_mm256_max_ps(out_a_scaled, v_ones), v_255);

                __m256i v_res = pack_colors_avx2(out_r, out_g, out_b, out_a_scaled);
                __m256i v_mask_i = _mm256_castps_si256(v_mask);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(destRow + x), _mm256_blendv_epi8(v_dst_raw, v_res, v_mask_i));
            }

            for (; x <= endX - 4; x += 4) {
                __m128 v_sx = _mm_fmadd_ps(v_idx_step4, v_dx_step_sse, _mm_set1_ps(startSrcX));
                startSrcX -= invScaleX * 4.0f;

                __m128 v_mask = _mm_and_ps(_mm_cmpge_ps(v_sx, v_ones_sse), _mm_cmple_ps(v_sx, v_srcW_limit_sse));
                if (_mm_movemask_ps(v_mask) == 0) continue;

                __m128 v_safe_sx = _mm_blendv_ps(v_ones_sse, v_sx, v_mask);
                __m128 v_flr_x = _mm_floor_ps(v_safe_sx);
                __m128i v_idx_x = _mm_cvtps_epi32(v_flr_x);

                __m128 v_dx = _mm_sub_ps(v_safe_sx, v_flr_x);
                __m128 v_inv_dx = _mm_sub_ps(v_ones_sse, v_dx);

                __m128 v_w1 = _mm_mul_ps(v_inv_dx, v_inv_dy_sse);
                __m128 v_w2 = _mm_mul_ps(v_dx, v_inv_dy_sse);
                __m128 v_w3 = _mm_mul_ps(v_inv_dx, v_dy_sse);
                __m128 v_w4 = _mm_mul_ps(v_dx, v_dy_sse);

                alignas(16) int indices_x[4];
                _mm_store_si128(reinterpret_cast<__m128i*>(indices_x), v_idx_x);

                __m128i c00 = _mm_set_epi32(
                    src_ptr[sy0 * src_stride + indices_x[3]],
                    src_ptr[sy0 * src_stride + indices_x[2]],
                    src_ptr[sy0 * src_stride + indices_x[1]],
                    src_ptr[sy0 * src_stride + indices_x[0]]
                );

                __m128i c10 = _mm_set_epi32(
                    src_ptr[sy0 * src_stride + indices_x[3] + 1],
                    src_ptr[sy0 * src_stride + indices_x[2] + 1],
                    src_ptr[sy0 * src_stride + indices_x[1] + 1],
                    src_ptr[sy0 * src_stride + indices_x[0] + 1]
                );

                __m128i c01 = _mm_set_epi32(
                    src_ptr[sy1 * src_stride + indices_x[3]],
                    src_ptr[sy1 * src_stride + indices_x[2]],
                    src_ptr[sy1 * src_stride + indices_x[1]],
                    src_ptr[sy1 * src_stride + indices_x[0]]
                );

                __m128i c11 = _mm_set_epi32(
                    src_ptr[sy1 * src_stride + indices_x[3] + 1],
                    src_ptr[sy1 * src_stride + indices_x[2] + 1],
                    src_ptr[sy1 * src_stride + indices_x[1] + 1],
                    src_ptr[sy1 * src_stride + indices_x[0] + 1]
                );

                __m128 s_r, s_g, s_b, s_a;
                {
                    __m128 r0, g0, b0, a0; unpack_colors_sse(c00, r0, g0, b0, a0);
                    __m128 r1, g1, b1, a1; unpack_colors_sse(c10, r1, g1, b1, a1);
                    __m128 r2, g2, b2, a2; unpack_colors_sse(c01, r2, g2, b2, a2);
                    __m128 r3, g3, b3, a3; unpack_colors_sse(c11, r3, g3, b3, a3);
                    s_r = _mm_fmadd_ps(r0, v_w1, _mm_fmadd_ps(r1, v_w2, _mm_fmadd_ps(r2, v_w3, _mm_mul_ps(r3, v_w4))));
                    s_g = _mm_fmadd_ps(g0, v_w1, _mm_fmadd_ps(g1, v_w2, _mm_fmadd_ps(g2, v_w3, _mm_mul_ps(g3, v_w4))));
                    s_b = _mm_fmadd_ps(b0, v_w1, _mm_fmadd_ps(b1, v_w2, _mm_fmadd_ps(b2, v_w3, _mm_mul_ps(b3, v_w4))));
                    s_a = _mm_fmadd_ps(a0, v_w1, _mm_fmadd_ps(a1, v_w2, _mm_fmadd_ps(a2, v_w3, _mm_mul_ps(a3, v_w4))));
                }

                __m128i v_dst_raw = _mm_loadu_si128(reinterpret_cast<__m128i*>(destRow + x));
                __m128 d_r, d_g, d_b, d_a;
                unpack_colors_sse(v_dst_raw, d_r, d_g, d_b, d_a);

                __m128 v_sa_norm = _mm_mul_ps(s_a, v_inv255_sse);
                __m128 v_da_norm = _mm_mul_ps(d_a, v_inv255_sse);
                __m128 v_out_a = _mm_add_ps(v_sa_norm, _mm_mul_ps(v_da_norm, _mm_sub_ps(v_ones_sse, v_sa_norm)));
                __m128 v_sa_factor = _mm_div_ps(v_sa_norm, _mm_max_ps(v_out_a, _mm_set1_ps(0.001f)));
                __m128 v_da_factor = _mm_sub_ps(v_ones_sse, v_sa_factor);

                __m128 out_r = _mm_fmadd_ps(s_r, v_sa_factor, _mm_mul_ps(d_r, v_da_factor));
                __m128 out_g = _mm_fmadd_ps(s_g, v_sa_factor, _mm_mul_ps(d_g, v_da_factor));
                __m128 out_b = _mm_fmadd_ps(s_b, v_sa_factor, _mm_mul_ps(d_b, v_da_factor));
                __m128 out_a_scaled = _mm_mul_ps(v_out_a, v_255_sse);

                out_r = _mm_min_ps(_mm_max_ps(out_r, v_ones_sse), v_255_sse);
                out_g = _mm_min_ps(_mm_max_ps(out_g, v_ones_sse), v_255_sse);
                out_b = _mm_min_ps(_mm_max_ps(out_b, v_ones_sse), v_255_sse);
                out_a_scaled = _mm_min_ps(_mm_max_ps(out_a_scaled, v_ones_sse), v_255_sse);

                __m128i v_res = pack_colors_sse(out_r, out_g, out_b, out_a_scaled);
                __m128i v_mask_i = _mm_castps_si128(v_mask);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(destRow + x), _mm_blendv_epi8(v_dst_raw, v_res, v_mask_i));
            }

            for (; x < endX; ++x) {
                float srcX = srcCenterX + (centerX - x) * invScaleX;

                if (srcX >= 0.0f && srcX <= srcW_limit && srcY >= 0.0f && srcY <= srcH_limit) {
                    int sx0 = static_cast<int>(srcX);
                    int sy0 = static_cast<int>(srcY);
                    float dx = srcX - sx0;
                    float dy = srcY - sy0;

                    const Color& c00 = src.at(sx0, sy0);
                    const Color& c10 = src.at(sx0 + 1, sy0);
                    const Color& c01 = src.at(sx0, sy0 + 1);
                    const Color& c11 = src.at(sx0 + 1, sy0 + 1);

                    float w1 = (1.0f - dx) * (1.0f - dy);
                    float w2 = dx * (1.0f - dy);
                    float w3 = (1.0f - dx) * dy;
                    float w4 = dx * dy;

                    float r = c00.r * w1 + c10.r * w2 + c01.r * w3 + c11.r * w4;
                    float g = c00.g * w1 + c10.g * w2 + c01.g * w3 + c11.g * w4;
                    float b = c00.b * w1 + c10.b * w2 + c01.b * w3 + c11.b * w4;
                    float a = c00.a * w1 + c10.a * w2 + c01.a * w3 + c11.a * w4;

                    Color& d = destRow[x];
                    float sa = a / 255.0f;
                    float da = d.a / 255.0f;
                    float out_a = sa + da * (1.0f - sa);
                    if (out_a > 0.001f) {
                        float sa_factor = sa / out_a;
                        float da_factor = 1.0f - sa_factor;
                        d.r = static_cast<uint8_t>(r * sa_factor + d.r * da_factor);
                        d.g = static_cast<uint8_t>(g * sa_factor + d.g * da_factor);
                        d.b = static_cast<uint8_t>(b * sa_factor + d.b * da_factor);
                    }
                    d.a = static_cast<uint8_t>(out_a * 255.0f);
                }
            }
        }
    }

    template<bool IS_90_DEG>
    inline void drawOrthogonalRotated(Buffer& dest, const Buffer& src, float centerX, float centerY, float scaleX, float scaleY) {
        if (!dest.isValid() || !src.isValid()) return;

        const float halfW = (IS_90_DEG ? src.height : src.height) * scaleX * 0.5f;
        const float halfH = (IS_90_DEG ? src.width : src.width) * scaleY * 0.5f;

        int startX = std::max(0, static_cast<int>(centerX - halfW));
        int startY = std::max(0, static_cast<int>(centerY - halfH));
        int endX = std::min(dest.width, static_cast<int>(centerX + halfW) + 1);
        int endY = std::min(dest.height, static_cast<int>(centerY + halfH) + 1);
        if (startX >= endX || startY >= endY) return;

        const float invScaleX = 1.0f / scaleX;
        const float invScaleY = 1.0f / scaleY;
        const float srcCenterX = src.width * 0.5f;
        const float srcCenterY = src.height * 0.5f;

        const float srcW_limit = (float)src.width - 1.05f;
        const float srcH_limit = (float)src.height - 1.05f;

        const __m256 v_idx_step8 = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256 v_ones = _mm256_set1_ps(1.0f);
        const __m256 v_255 = _mm256_set1_ps(255.0f);
        const __m256 v_inv255 = _mm256_set1_ps(1.0f / 255.0f);
        const __m256 v_srcH_limit = _mm256_set1_ps(srcH_limit);

        float dy_step_val = IS_90_DEG ? -invScaleX : invScaleX;
        __m256 v_dy_step = _mm256_set1_ps(dy_step_val);

        const __m128 v_idx_step4 = _mm_setr_ps(0, 1, 2, 3);
        const __m128 v_ones_sse = _mm_set1_ps(1.0f);
        const __m128 v_255_sse = _mm_set1_ps(255.0f);
        const __m128 v_inv255_sse = _mm_set1_ps(1.0f / 255.0f);
        const __m128 v_srcH_limit_sse = _mm_set1_ps(srcH_limit);

        const int src_stride = src.width;
        const int* src_ptr = reinterpret_cast<const int*>(src.color);

        for (int y = startY; y < endY; ++y) {
            float srcX;
            if (IS_90_DEG)
                srcX = srcCenterX + (y - centerY) * invScaleY;
            else
                srcX = srcCenterX - (y - centerY) * invScaleY;

            if (srcX < 0.0f || srcX > srcW_limit) continue;

            int sx0 = static_cast<int>(srcX);
            float dx = srcX - sx0;
            __m256 v_dx = _mm256_set1_ps(dx);
            __m256 v_inv_dx = _mm256_sub_ps(v_ones, v_dx);

            __m128 v_dx_sse = _mm_set1_ps(dx);
            __m128 v_inv_dx_sse = _mm_sub_ps(v_ones_sse, v_dx_sse);

            int col_offset0 = sx0;
            int col_offset1 = sx0 + 1;

            Color* destRow = dest.getRow(y);

            float startSrcY;
            if (IS_90_DEG)
                startSrcY = srcCenterY - (startX - centerX) * invScaleX;
            else
                startSrcY = srcCenterY + (startX - centerX) * invScaleX;

            int x = startX;

            for (; x <= endX - 8; x += 8) {
                __m256 v_sy = _mm256_fmadd_ps(v_idx_step8, v_dy_step, _mm256_set1_ps(startSrcY));
                startSrcY += dy_step_val * 8.0f;

                __m256 v_mask = _mm256_and_ps(_mm256_cmp_ps(v_sy, v_ones, _CMP_GE_OQ), _mm256_cmp_ps(v_sy, v_srcH_limit, _CMP_LE_OQ));
                if (_mm256_movemask_ps(v_mask) == 0) continue;

                __m256 v_safe_sy = _mm256_blendv_ps(v_ones, v_sy, v_mask);
                __m256 v_flr_y = _mm256_floor_ps(v_safe_sy);
                __m256i v_idx_y = _mm256_cvtps_epi32(v_flr_y);

                __m256 v_dy = _mm256_sub_ps(v_safe_sy, v_flr_y);
                __m256 v_inv_dy = _mm256_sub_ps(v_ones, v_dy);

                __m256 v_w1 = _mm256_mul_ps(v_inv_dx, v_inv_dy);
                __m256 v_w2 = _mm256_mul_ps(v_dx, v_inv_dy);
                __m256 v_w3 = _mm256_mul_ps(v_inv_dx, v_dy);
                __m256 v_w4 = _mm256_mul_ps(v_dx, v_dy);

                __m256i v_row_base = _mm256_mullo_epi32(v_idx_y, _mm256_set1_epi32(src_stride));

                __m256i v_idx00 = _mm256_add_epi32(v_row_base, _mm256_set1_epi32(col_offset0));
                __m256i v_idx10 = _mm256_add_epi32(v_row_base, _mm256_set1_epi32(col_offset1));
                __m256i v_idx01 = _mm256_add_epi32(v_idx00, _mm256_set1_epi32(src_stride));
                __m256i v_idx11 = _mm256_add_epi32(v_idx10, _mm256_set1_epi32(src_stride));

                __m256i c00 = _mm256_i32gather_epi32(src_ptr, v_idx00, 4);
                __m256i c10 = _mm256_i32gather_epi32(src_ptr, v_idx10, 4);
                __m256i c01 = _mm256_i32gather_epi32(src_ptr, v_idx01, 4);
                __m256i c11 = _mm256_i32gather_epi32(src_ptr, v_idx11, 4);

                __m256 s_r, s_g, s_b, s_a;
                {
                    __m256 r0, g0, b0, a0; unpack_colors_avx2(c00, r0, g0, b0, a0);
                    __m256 r1, g1, b1, a1; unpack_colors_avx2(c10, r1, g1, b1, a1);
                    __m256 r2, g2, b2, a2; unpack_colors_avx2(c01, r2, g2, b2, a2);
                    __m256 r3, g3, b3, a3; unpack_colors_avx2(c11, r3, g3, b3, a3);
                    s_r = _mm256_fmadd_ps(r0, v_w1, _mm256_fmadd_ps(r1, v_w2, _mm256_fmadd_ps(r2, v_w3, _mm256_mul_ps(r3, v_w4))));
                    s_g = _mm256_fmadd_ps(g0, v_w1, _mm256_fmadd_ps(g1, v_w2, _mm256_fmadd_ps(g2, v_w3, _mm256_mul_ps(g3, v_w4))));
                    s_b = _mm256_fmadd_ps(b0, v_w1, _mm256_fmadd_ps(b1, v_w2, _mm256_fmadd_ps(b2, v_w3, _mm256_mul_ps(b3, v_w4))));
                    s_a = _mm256_fmadd_ps(a0, v_w1, _mm256_fmadd_ps(a1, v_w2, _mm256_fmadd_ps(a2, v_w3, _mm256_mul_ps(a3, v_w4))));
                }

                __m256i v_dst_raw = _mm256_loadu_si256(reinterpret_cast<__m256i*>(destRow + x));
                __m256 d_r, d_g, d_b, d_a;
                unpack_colors_avx2(v_dst_raw, d_r, d_g, d_b, d_a);

                __m256 v_sa_norm = _mm256_mul_ps(s_a, v_inv255);
                __m256 v_da_norm = _mm256_mul_ps(d_a, v_inv255);
                __m256 v_out_a = _mm256_add_ps(v_sa_norm, _mm256_mul_ps(v_da_norm, _mm256_sub_ps(v_ones, v_sa_norm)));
                __m256 v_sa_factor = _mm256_div_ps(v_sa_norm, _mm256_max_ps(v_out_a, _mm256_set1_ps(0.001f)));
                __m256 v_da_factor = _mm256_sub_ps(v_ones, v_sa_factor);

                __m256 out_r = _mm256_fmadd_ps(s_r, v_sa_factor, _mm256_mul_ps(d_r, v_da_factor));
                __m256 out_g = _mm256_fmadd_ps(s_g, v_sa_factor, _mm256_mul_ps(d_g, v_da_factor));
                __m256 out_b = _mm256_fmadd_ps(s_b, v_sa_factor, _mm256_mul_ps(d_b, v_da_factor));
                __m256 out_a_scaled = _mm256_mul_ps(v_out_a, v_255);

                out_r = _mm256_min_ps(_mm256_max_ps(out_r, v_ones), v_255);
                out_g = _mm256_min_ps(_mm256_max_ps(out_g, v_ones), v_255);
                out_b = _mm256_min_ps(_mm256_max_ps(out_b, v_ones), v_255);
                out_a_scaled = _mm256_min_ps(_mm256_max_ps(out_a_scaled, v_ones), v_255);

                __m256i v_res = pack_colors_avx2(out_r, out_g, out_b, out_a_scaled);
                __m256i v_mask_i = _mm256_castps_si256(v_mask);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(destRow + x), _mm256_blendv_epi8(v_dst_raw, v_res, v_mask_i));
            }

            __m128 v_dy_step_sse = _mm_set1_ps(dy_step_val);
            for (; x <= endX - 4; x += 4) {
                __m128 v_sy = _mm_fmadd_ps(v_idx_step4, v_dy_step_sse, _mm_set1_ps(startSrcY));
                startSrcY += dy_step_val * 4.0f;

                __m128 v_mask = _mm_and_ps(_mm_cmpge_ps(v_sy, v_ones_sse), _mm_cmple_ps(v_sy, v_srcH_limit_sse));
                if (_mm_movemask_ps(v_mask) == 0) continue;

                __m128 v_safe_sy = _mm_blendv_ps(v_ones_sse, v_sy, v_mask);
                __m128 v_flr_y = _mm_floor_ps(v_safe_sy);
                __m128i v_idx_y = _mm_cvtps_epi32(v_flr_y);

                __m128 v_dy = _mm_sub_ps(v_safe_sy, v_flr_y);
                __m128 v_inv_dy = _mm_sub_ps(v_ones_sse, v_dy);

                __m128 v_w1 = _mm_mul_ps(v_inv_dx_sse, v_inv_dy);
                __m128 v_w2 = _mm_mul_ps(v_dx_sse, v_inv_dy);
                __m128 v_w3 = _mm_mul_ps(v_inv_dx_sse, v_dy);
                __m128 v_w4 = _mm_mul_ps(v_dx_sse, v_dy);

                alignas(16) int indices_y[4];
                _mm_store_si128(reinterpret_cast<__m128i*>(indices_y), v_idx_y);

                __m128i c00 = _mm_set_epi32(
                    src_ptr[(indices_y[3]) * src_stride + col_offset0],
                    src_ptr[(indices_y[2]) * src_stride + col_offset0],
                    src_ptr[(indices_y[1]) * src_stride + col_offset0],
                    src_ptr[(indices_y[0]) * src_stride + col_offset0]
                );

                __m128i c10 = _mm_set_epi32(
                    src_ptr[(indices_y[3]) * src_stride + col_offset1],
                    src_ptr[(indices_y[2]) * src_stride + col_offset1],
                    src_ptr[(indices_y[1]) * src_stride + col_offset1],
                    src_ptr[(indices_y[0]) * src_stride + col_offset1]
                );

                __m128i c01 = _mm_set_epi32(
                    src_ptr[(indices_y[3] + 1) * src_stride + col_offset0],
                    src_ptr[(indices_y[2] + 1) * src_stride + col_offset0],
                    src_ptr[(indices_y[1] + 1) * src_stride + col_offset0],
                    src_ptr[(indices_y[0] + 1) * src_stride + col_offset0]
                );

                __m128i c11 = _mm_set_epi32(
                    src_ptr[(indices_y[3] + 1) * src_stride + col_offset1],
                    src_ptr[(indices_y[2] + 1) * src_stride + col_offset1],
                    src_ptr[(indices_y[1] + 1) * src_stride + col_offset1],
                    src_ptr[(indices_y[0] + 1) * src_stride + col_offset1]
                );

                __m128 s_r, s_g, s_b, s_a;
                {
                    __m128 r0, g0, b0, a0; unpack_colors_sse(c00, r0, g0, b0, a0);
                    __m128 r1, g1, b1, a1; unpack_colors_sse(c10, r1, g1, b1, a1);
                    __m128 r2, g2, b2, a2; unpack_colors_sse(c01, r2, g2, b2, a2);
                    __m128 r3, g3, b3, a3; unpack_colors_sse(c11, r3, g3, b3, a3);
                    s_r = _mm_fmadd_ps(r0, v_w1, _mm_fmadd_ps(r1, v_w2, _mm_fmadd_ps(r2, v_w3, _mm_mul_ps(r3, v_w4))));
                    s_g = _mm_fmadd_ps(g0, v_w1, _mm_fmadd_ps(g1, v_w2, _mm_fmadd_ps(g2, v_w3, _mm_mul_ps(g3, v_w4))));
                    s_b = _mm_fmadd_ps(b0, v_w1, _mm_fmadd_ps(b1, v_w2, _mm_fmadd_ps(b2, v_w3, _mm_mul_ps(b3, v_w4))));
                    s_a = _mm_fmadd_ps(a0, v_w1, _mm_fmadd_ps(a1, v_w2, _mm_fmadd_ps(a2, v_w3, _mm_mul_ps(a3, v_w4))));
                }

                __m128i v_dst_raw = _mm_loadu_si128(reinterpret_cast<__m128i*>(destRow + x));
                __m128 d_r, d_g, d_b, d_a;
                unpack_colors_sse(v_dst_raw, d_r, d_g, d_b, d_a);

                __m128 v_sa_norm = _mm_mul_ps(s_a, v_inv255_sse);
                __m128 v_da_norm = _mm_mul_ps(d_a, v_inv255_sse);
                __m128 v_out_a = _mm_add_ps(v_sa_norm, _mm_mul_ps(v_da_norm, _mm_sub_ps(v_ones_sse, v_sa_norm)));
                __m128 v_sa_factor = _mm_div_ps(v_sa_norm, _mm_max_ps(v_out_a, _mm_set1_ps(0.001f)));
                __m128 v_da_factor = _mm_sub_ps(v_ones_sse, v_sa_factor);

                __m128 out_r = _mm_fmadd_ps(s_r, v_sa_factor, _mm_mul_ps(d_r, v_da_factor));
                __m128 out_g = _mm_fmadd_ps(s_g, v_sa_factor, _mm_mul_ps(d_g, v_da_factor));
                __m128 out_b = _mm_fmadd_ps(s_b, v_sa_factor, _mm_mul_ps(d_b, v_da_factor));
                __m128 out_a_scaled = _mm_mul_ps(v_out_a, v_255_sse);

                out_r = _mm_min_ps(_mm_max_ps(out_r, v_ones_sse), v_255_sse);
                out_g = _mm_min_ps(_mm_max_ps(out_g, v_ones_sse), v_255_sse);
                out_b = _mm_min_ps(_mm_max_ps(out_b, v_ones_sse), v_255_sse);
                out_a_scaled = _mm_min_ps(_mm_max_ps(out_a_scaled, v_ones_sse), v_255_sse);

                __m128i v_res = pack_colors_sse(out_r, out_g, out_b, out_a_scaled);
                __m128i v_mask_i = _mm_castps_si128(v_mask);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(destRow + x), _mm_blendv_epi8(v_dst_raw, v_res, v_mask_i));
            }

            for (; x < endX; ++x) {
                float sy = startSrcY + (x - startX) * dy_step_val;
                if (sy >= 0 && sy < srcH_limit) {
                    int sy0 = (int)sy;
                    float dy = sy - sy0;

                    const Color& c00 = src.color[sy0 * src_stride + col_offset0];
                    const Color& c10 = src.color[sy0 * src_stride + col_offset1];
                    const Color& c01 = src.color[(sy0 + 1) * src_stride + col_offset0];
                    const Color& c11 = src.color[(sy0 + 1) * src_stride + col_offset1];

                    float w1 = (1.0f - dx) * (1.0f - dy);
                    float w2 = dx * (1.0f - dy);
                    float w3 = (1.0f - dx) * dy;
                    float w4 = dx * dy;

                    float r = c00.r * w1 + c10.r * w2 + c01.r * w3 + c11.r * w4;
                    float g = c00.g * w1 + c10.g * w2 + c01.g * w3 + c11.g * w4;
                    float b = c00.b * w1 + c10.b * w2 + c01.b * w3 + c11.b * w4;
                    float a = c00.a * w1 + c10.a * w2 + c01.a * w3 + c11.a * w4;

                    Color& d = destRow[x];
                    float sa = a / 255.0f;
                    float da = d.a / 255.0f;
                    float out_a = sa + da * (1.0f - sa);
                    if (out_a > 0.001f) {
                        float sa_factor = sa / out_a;
                        float da_factor = 1.0f - sa_factor;
                        d.r = static_cast<uint8_t>(r * sa_factor + d.r * da_factor);
                        d.g = static_cast<uint8_t>(g * sa_factor + d.g * da_factor);
                        d.b = static_cast<uint8_t>(b * sa_factor + d.b * da_factor);
                    }
                    d.a = static_cast<uint8_t>(out_a * 255.0f);
                }
            }
        }
    }

    void drawRotated90Scaled(Buffer& dest, const Buffer& src, float centerX, float centerY, float scaleX, float scaleY) {
        drawOrthogonalRotated<true>(dest, src, centerX, centerY, scaleX, scaleY);
    }

    void drawRotated270Scaled(Buffer& dest, const Buffer& src, float centerX, float centerY, float scaleX, float scaleY) {
        drawOrthogonalRotated<false>(dest, src, centerX, centerY, scaleX, scaleY);
    }

} // namespace pa2d