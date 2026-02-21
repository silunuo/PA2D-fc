#pragma once
#ifdef _MSC_VER
    #define PA2D_FORCEINLINE __forceinline
    #define PA2D_NOINLINE    __declspec(noinline)
#else
    #define PA2D_FORCEINLINE __attribute__((always_inline)) inline
    #define PA2D_NOINLINE    __attribute__((noinline))
#endif

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

#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
    #include <immintrin.h>
    #define PA2D_HAS_AVX2 1
#else
    #define PA2D_HAS_AVX2 0
#endif

#ifndef NOMINMAX
    #define NOMINMAX   // 防止 Windows.h 覆盖 std::min / std::max
#endif
