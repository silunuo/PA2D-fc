#pragma once
// ============================================================
// compiler_compat.h
// 编译器兼容层 ── 统一 MSVC 与 GCC/MinGW-w64 的差异
//
// 在每个需要兼容的 .cpp 文件顶部 #include 此头文件
// ============================================================

// 内联提示
#ifdef _MSC_VER
    #define PA2D_FORCEINLINE __forceinline
    #define PA2D_NOINLINE    __declspec(noinline)
#else
    #define PA2D_FORCEINLINE __attribute__((always_inline)) inline
    #define PA2D_NOINLINE    __attribute__((noinline))
#endif

// ── 安全字符串函数 ───────────────────────────────────────────
// MSVC 提供 wcsncpy_s / sprintf_s 等"安全"版本；
// GCC/MinGW 通过 __STDC_WANT_LIB_EXT1__=1 可以支持部分，
// 但为了最大兼容性，这里提供统一宏。
#ifndef _MSC_VER
    #include <cstring>
    #include <cwchar>
    #include <cstdio>
    #ifndef wcsncpy_s
        inline int wcsncpy_s(wchar_t* dst, size_t dstSize,
                             const wchar_t* src, size_t count) {
            if (!dst || dstSize == 0) return 1;
            size_t n = (count < dstSize - 1) ? count : dstSize - 1;
            wcsncpy(dst, src, n);       // 复制最多 n 个宽字符
            dst[n] = L'\0';             // 确保 null 终止
            return 0;
        }
    #endif

    // sprintf_s → snprintf（参数顺序相同，行为等价）
    #ifndef sprintf_s
        #define sprintf_s(buf, size, fmt, ...) \
            snprintf((buf), (size), (fmt), ##__VA_ARGS__)
    #endif

    // sscanf_s → sscanf（GCC 下 sscanf_s 不存在）
    #ifndef sscanf_s
        #define sscanf_s sscanf
    #endif
#endif

// ── AVX2 / SIMD ──────────────────────────────────────────────
// MSVC 和 GCC 的 intrinsic 头文件名相同，均为 <immintrin.h>
// 但 GCC 需要额外的编译选项 -mavx2（在 CMakeLists.txt 中设置）
#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
    #include <immintrin.h>
    #define PA2D_HAS_AVX2 1
#else
    #define PA2D_HAS_AVX2 0
#endif

// ── 诊断警告抑制 ─────────────────────────────────────────────
// 某些 Win32 宏在 GCC 下会产生"redefine"警告，统一在此处理
#ifndef NOMINMAX
    #define NOMINMAX   // 防止 Windows.h 覆盖 std::min / std::max
#endif
