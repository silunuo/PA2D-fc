#include "../include/filter.h"
#include "../include/compiler_compat.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

// ============================================================
// filter.cpp  ── PA2D 滤镜系统实现
//
// 实现策略：
//   · 所有滤镜先计算实际作用矩形（resolveRect）
//   · 逐像素运算尽量用整数算术，避免浮点循环开销
//   · 卷积类滤镜（模糊/锐化）使用可分离一维卷积优化
//   · Alpha 通道默认不受大多数滤镜影响（特殊标注除外）
// ============================================================

namespace pa2d {

namespace {

// 将 FilterRect 解析为实际像素坐标，并裁剪到 buf 范围内
struct Rect4 { int x0, y0, x1, y1; }; // [x0,x1) × [y0,y1)

Rect4 resolveRect(const Buffer& buf, const FilterRect& r) {
    int x0 = r.x;
    int y0 = r.y;
    int x1 = (r.w < 0) ? buf.width  : (r.x + r.w);
    int y1 = (r.h < 0) ? buf.height : (r.y + r.h);
    // 裁剪到缓冲区边界
    x0 = std::max(0, std::min(x0, buf.width));
    y0 = std::max(0, std::min(y0, buf.height));
    x1 = std::max(x0, std::min(x1, buf.width));
    y1 = std::max(y0, std::min(y1, buf.height));
    return {x0, y0, x1, y1};
}

// clamp 到 [0, 255]
PA2D_FORCEINLINE uint8_t clamp8(int v) {
    return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}
PA2D_FORCEINLINE uint8_t clamp8f(float v) {
    return (uint8_t)(v < 0.f ? 0 : (v > 255.f ? 255 : (int)v));
}

// 将 Color 的 RGB 解包为浮点
PA2D_FORCEINLINE void unpackRGBf(const Color& c,
                                  float& r, float& g, float& b) {
    r = c.r; g = c.g; b = c.b;
}

// 构建一维高斯核（归一化）
std::vector<float> makeGaussianKernel(float sigma) {
    int radius = (int)std::ceil(3.f * sigma);
    int size   = radius * 2 + 1;
    std::vector<float> kernel(size);
    float sum = 0.f;
    for (int i = 0; i < size; ++i) {
        float x = (float)(i - radius);
        kernel[i] = std::exp(-(x * x) / (2.f * sigma * sigma));
        sum += kernel[i];
    }

    for (float& v : kernel) v /= sum;
    return kernel;
}


void convolveH(const Buffer& src, Buffer& dst,
               const std::vector<float>& kernel,
               const Rect4& rc) {
    int radius = (int)(kernel.size() / 2);
    for (int y = rc.y0; y < rc.y1; ++y) {
        const Color* srcRow = src.getRow(y);
        Color*       dstRow = dst.getRow(y);
        for (int x = rc.x0; x < rc.x1; ++x) {
            float r = 0, g = 0, b = 0;
            for (int k = -(int)radius; k <= (int)radius; ++k) {
                int sx = std::max(rc.x0, std::min(rc.x1 - 1, x + k));
                float w = kernel[k + radius];
                r += srcRow[sx].r * w;
                g += srcRow[sx].g * w;
                b += srcRow[sx].b * w;
            }
            dstRow[x].r = clamp8f(r);
            dstRow[x].g = clamp8f(g);
            dstRow[x].b = clamp8f(b);
            dstRow[x].a = srcRow[x].a; // 保留 alpha
        }
    }
}

// 垂直方向一维卷积
void convolveV(const Buffer& src, Buffer& dst,
               const std::vector<float>& kernel,
               const Rect4& rc) {
    int radius = (int)(kernel.size() / 2);
    for (int y = rc.y0; y < rc.y1; ++y) {
        Color* dstRow = dst.getRow(y);
        for (int x = rc.x0; x < rc.x1; ++x) {
            float r = 0, g = 0, b = 0;
            for (int k = -(int)radius; k <= (int)radius; ++k) {
                int sy = std::max(rc.y0, std::min(rc.y1 - 1, y + k));
                float w  = kernel[k + radius];
                const Color& sc = src.at(x, sy);
                r += sc.r * w;
                g += sc.g * w;
                b += sc.b * w;
            }
            dstRow[x].r = clamp8f(r);
            dstRow[x].g = clamp8f(g);
            dstRow[x].b = clamp8f(b);
            dstRow[x].a = src.at(x, y).a;
        }
    }
}

void copyRegion(const Buffer& src, Buffer& dst, const Rect4& rc) {
    for (int y = rc.y0; y < rc.y1; ++y) {
        const Color* s = src.getRow(y) + rc.x0;
        Color*       d = dst.getRow(y) + rc.x0;
        std::memcpy(d, s, (rc.x1 - rc.x0) * sizeof(Color));
    }
}

PA2D_FORCEINLINE float luminance(const Color& c) {
    return c.r * 0.299f + c.g * 0.587f + c.b * 0.114f;
}
// RGB → HSV（h: 0~360, s/v: 0~1）
void rgbToHSV(float r, float g, float b,
              float& h, float& s, float& v) {
    r /= 255.f; g /= 255.f; b /= 255.f;
    float mx = std::max({r, g, b});
    float mn = std::min({r, g, b});
    float delta = mx - mn;
    v = mx;
    s = (mx > 1e-6f) ? (delta / mx) : 0.f;
    if (delta < 1e-6f) { h = 0.f; return; }
    if      (mx == r) h = 60.f * (std::fmod((g - b) / delta, 6.f));
    else if (mx == g) h = 60.f * ((b - r) / delta + 2.f);
    else              h = 60.f * ((r - g) / delta + 4.f);
    if (h < 0.f) h += 360.f;
}
// HSV → RGB（输出 0~255）
void hsvToRGB(float h, float s, float v,
              float& r, float& g, float& b) {
    if (s < 1e-6f) { r = g = b = v * 255.f; return; }
    float sector = h / 60.f;
    int   i = (int)sector % 6;
    float f = sector - (int)sector;
    float p = v * (1.f - s);
    float q = v * (1.f - f * s);
    float t = v * (1.f - (1.f - f) * s);
    switch (i) {
        case 0: r=v; g=t; b=p; break;
        case 1: r=q; g=v; b=p; break;
        case 2: r=p; g=v; b=t; break;
        case 3: r=p; g=q; b=v; break;
        case 4: r=t; g=p; b=v; break;
        default:r=v; g=p; b=q; break;
    }
    r *= 255.f; g *= 255.f; b *= 255.f;
}

} 

// filter 命名空间实现
namespace filter {

// 灰度化
// 公式：gray = 0.299R + 0.587G + 0.114B（ITU-R BT.601）
void grayscale(Buffer& buf, const FilterRect& region) {
    auto rc = resolveRect(buf, region);
    for (int y = rc.y0; y < rc.y1; ++y) {
        Color* row = buf.getRow(y);
        for (int x = rc.x0; x < rc.x1; ++x) {
            uint8_t g = clamp8f(luminance(row[x]));
            row[x].r = row[x].g = row[x].b = g;
        }
    }
}

// 反色
// 对每个通道取 255 - v；alpha 不变
void invert(Buffer& buf, const FilterRect& region) {
    auto rc = resolveRect(buf, region);
    for (int y = rc.y0; y < rc.y1; ++y) {
        Color* row = buf.getRow(y);
        for (int x = rc.x0; x < rc.x1; ++x) {
            row[x].r = 255 - row[x].r;
            row[x].g = 255 - row[x].g;
            row[x].b = 255 - row[x].b;
        }
    }
}

// 亮度调节
// 新值 = 原值 * factor
void brightness(Buffer& buf, float factor, const FilterRect& region) {
    auto rc = resolveRect(buf, region);
    for (int y = rc.y0; y < rc.y1; ++y) {
        Color* row = buf.getRow(y);
        for (int x = rc.x0; x < rc.x1; ++x) {
            row[x].r = clamp8f(row[x].r * factor);
            row[x].g = clamp8f(row[x].g * factor);
            row[x].b = clamp8f(row[x].b * factor);
        }
    }
}

// 对比度调节
// 以 128 为中心：新值 = (原值 - 128) * factor + 128
void contrast(Buffer& buf, float factor, const FilterRect& region) {
    auto rc = resolveRect(buf, region);
    for (int y = rc.y0; y < rc.y1; ++y) {
        Color* row = buf.getRow(y);
        for (int x = rc.x0; x < rc.x1; ++x) {
            row[x].r = clamp8f((row[x].r - 128.f) * factor + 128.f);
            row[x].g = clamp8f((row[x].g - 128.f) * factor + 128.f);
            row[x].b = clamp8f((row[x].b - 128.f) * factor + 128.f);
        }
    }
}

// 饱和度调节
// 在 HSV 空间调整 S 分量
void saturation(Buffer& buf, float factor, const FilterRect& region) {
    auto rc = resolveRect(buf, region);
    for (int y = rc.y0; y < rc.y1; ++y) {
        Color* row = buf.getRow(y);
        for (int x = rc.x0; x < rc.x1; ++x) {
            float r = row[x].r, g = row[x].g, b = row[x].b;
            float h, s, v;
            rgbToHSV(r, g, b, h, s, v);
            s = std::min(1.f, s * factor);
            hsvToRGB(h, s, v, r, g, b);
            row[x].r = clamp8f(r);
            row[x].g = clamp8f(g);
            row[x].b = clamp8f(b);
        }
    }
}

// 色相旋转
// 在 HSV 空间把 H 加上 angle 度
void hueRotate(Buffer& buf, float angle, const FilterRect& region) {
    auto rc = resolveRect(buf, region);
    for (int y = rc.y0; y < rc.y1; ++y) {
        Color* row = buf.getRow(y);
        for (int x = rc.x0; x < rc.x1; ++x) {
            float r = row[x].r, g = row[x].g, b = row[x].b;
            float h, s, v;
            rgbToHSV(r, g, b, h, s, v);
            h = std::fmod(h + angle, 360.f);
            if (h < 0.f) h += 360.f;
            hsvToRGB(h, s, v, r, g, b);
            row[x].r = clamp8f(r);
            row[x].g = clamp8f(g);
            row[x].b = clamp8f(b);
        }
    }
}

// 颜色叠加
// 新值 = 原值 * (1 - intensity) + 目标颜色 * intensity
void colorTint(Buffer& buf, const Color& tint, float intensity,
               const FilterRect& region) {
    auto   rc = resolveRect(buf, region);
    float  inv = 1.f - intensity;
    for (int y = rc.y0; y < rc.y1; ++y) {
        Color* row = buf.getRow(y);
        for (int x = rc.x0; x < rc.x1; ++x) {
            row[x].r = clamp8f(row[x].r * inv + tint.r * intensity);
            row[x].g = clamp8f(row[x].g * inv + tint.g * intensity);
            row[x].b = clamp8f(row[x].b * inv + tint.b * intensity);
        }
    }
}

// 怀旧滤镜
// 先灰度化，再用怀旧色调矩阵叠加
void sepia(Buffer& buf, float intensity, const FilterRect& region) {
    auto rc = resolveRect(buf, region);
    for (int y = rc.y0; y < rc.y1; ++y) {
        Color* row = buf.getRow(y);
        for (int x = rc.x0; x < rc.x1; ++x) {
            float r = row[x].r, g = row[x].g, b = row[x].b;
            // 怀旧色调矩阵（来自 CSS sepia 标准）
            float sr = r * 0.393f + g * 0.769f + b * 0.189f;
            float sg = r * 0.349f + g * 0.686f + b * 0.168f;
            float sb = r * 0.272f + g * 0.534f + b * 0.131f;
            // 与原色插值
            row[x].r = clamp8f(r * (1.f - intensity) + sr * intensity);
            row[x].g = clamp8f(g * (1.f - intensity) + sg * intensity);
            row[x].b = clamp8f(b * (1.f - intensity) + sb * intensity);
        }
    }
}

// Alpha 透明度调节
void opacity(Buffer& buf, float factor, const FilterRect& region) {
    auto rc = resolveRect(buf, region);
    for (int y = rc.y0; y < rc.y1; ++y) {
        Color* row = buf.getRow(y);
        for (int x = rc.x0; x < rc.x1; ++x) {
            row[x].a = clamp8f(row[x].a * factor);
        }
    }
}

// 盒式模糊
// 使用两次可分离的一维均值卷积（水平 + 垂直），
// 速度比二维卷积快 O(radius) 倍
void boxBlur(Buffer& buf, int radius, const FilterRect& region) {
    if (radius <= 0) return;
    auto rc = resolveRect(buf, region);

    // 用均匀权重的一维核
    int   size = radius * 2 + 1;
    float w    = 1.f / size;
    std::vector<float> kernel(size, w);

    // 水平卷积：buf → tmp
    Buffer tmp(buf.width, buf.height);
    copyRegion(buf, tmp, rc);    // 把区域外像素也拷过来避免边界颜色错乱
    convolveH(buf, tmp, kernel, rc);

    // 垂直卷积：tmp → buf
    convolveV(tmp, buf, kernel, rc);
}

// 高斯模糊
// 可分离高斯核，质量高于盒式模糊
void gaussianBlur(Buffer& buf, float sigma, const FilterRect& region) {
    if (sigma <= 0.f) return;
    auto  rc     = resolveRect(buf, region);
    auto  kernel = makeGaussianKernel(sigma);

    Buffer tmp(buf.width, buf.height);
    copyRegion(buf, tmp, rc);
    convolveH(buf, tmp, kernel, rc);
    convolveV(tmp, buf, kernel, rc);
}

// 径向模糊
// 沿以 (cx,cy) 为圆心的方向进行采样累加
void radialBlur(Buffer& buf, float strength, int cx, int cy,
                const FilterRect& region) {
    auto rc = resolveRect(buf, region);
    int  ocx = (cx < 0) ? buf.width  / 2 : cx;
    int  ocy = (cy < 0) ? buf.height / 2 : cy;

    Buffer src = buf; // 先备份原始图像（避免读写冲突）

    const int SAMPLES = 8; // 采样数，越多越平滑，越慢
    for (int y = rc.y0; y < rc.y1; ++y) {
        Color* dst = buf.getRow(y);
        for (int x = rc.x0; x < rc.x1; ++x) {
            float dx = (float)(x - ocx);
            float dy = (float)(y - ocy);
            float ar = 0, ag = 0, ab = 0;
            for (int s = 0; s < SAMPLES; ++s) {
                // 沿径向方向按 strength 缩放采样坐标
                float t  = (float)s / SAMPLES * strength;
                int   sx = std::max(rc.x0, std::min(rc.x1 - 1,
                                    (int)(ocx + dx * (1.f - t))));
                int   sy = std::max(rc.y0, std::min(rc.y1 - 1,
                                    (int)(ocy + dy * (1.f - t))));
                const Color& sc = src.at(sx, sy);
                ar += sc.r; ag += sc.g; ab += sc.b;
            }
            dst[x].r = clamp8f(ar / SAMPLES);
            dst[x].g = clamp8f(ag / SAMPLES);
            dst[x].b = clamp8f(ab / SAMPLES);
        }
    }
}

// 锐化（Unsharp Mask）
// 原图 + strength × (原图 - 高斯模糊版)
void sharpen(Buffer& buf, float strength, const FilterRect& region) {
    auto  rc     = resolveRect(buf, region);
    Buffer blurred = buf;
    // 用轻度高斯模糊生成"模糊版"
    gaussianBlur(blurred, 1.2f, region);

    for (int y = rc.y0; y < rc.y1; ++y) {
        Color*       dst = buf.getRow(y);
        const Color* src = blurred.getRow(y);
        for (int x = rc.x0; x < rc.x1; ++x) {
            // 原图 + strength × (原图 - 模糊图) = 锐化结果
            dst[x].r = clamp8f(dst[x].r + strength * (dst[x].r - src[x].r));
            dst[x].g = clamp8f(dst[x].g + strength * (dst[x].g - src[x].g));
            dst[x].b = clamp8f(dst[x].b + strength * (dst[x].b - src[x].b));
        }
    }
}

// 边缘检测（Sobel 算子）
// 计算水平/垂直梯度的模，结果为灰度图
void edgeDetect(Buffer& buf, const FilterRect& region) {
    auto   rc  = resolveRect(buf, region);
    Buffer src = buf; // 备份原图

    // Sobel 3×3 核
    // Gx:  -1  0  1        Gy: -1 -2 -1
    //      -2  0  2              0  0  0
    //      -1  0  1             +1 +2 +1
    const int kx[3][3] = { {-1,0,1},{-2,0,2},{-1,0,1} };
    const int ky[3][3] = { {-1,-2,-1},{0,0,0},{1,2,1} };

    for (int y = rc.y0; y < rc.y1; ++y) {
        Color* dst = buf.getRow(y);
        for (int x = rc.x0; x < rc.x1; ++x) {
            float gx = 0, gy = 0;
            for (int ky2 = -1; ky2 <= 1; ++ky2) {
                for (int kx2 = -1; kx2 <= 1; ++kx2) {
                    int sx = std::max(rc.x0, std::min(rc.x1-1, x+kx2));
                    int sy = std::max(rc.y0, std::min(rc.y1-1, y+ky2));
                    float lum = luminance(src.at(sx, sy));
                    gx += lum * kx[ky2+1][kx2+1];
                    gy += lum * ky[ky2+1][kx2+1];
                }
            }
            uint8_t edge = clamp8f(std::sqrt(gx*gx + gy*gy));
            dst[x].r = dst[x].g = dst[x].b = edge;
        }
    }
}

// 浮雕效果
// 从 angle 方向的相邻像素差值 + 128 作为灰度输出
void emboss(Buffer& buf, float angle, float strength,
            const FilterRect& region) {
    auto   rc    = resolveRect(buf, region);
    Buffer src   = buf;
    float  rad   = angle * (3.14159265f / 180.f);
    int    dx    = (int)(std::cos(rad) + 0.5f);  // 光源 x 偏移
    int    dy    = (int)(-std::sin(rad) + 0.5f); // 光源 y 偏移（屏幕 y 向下）

    for (int y = rc.y0; y < rc.y1; ++y) {
        Color* dst = buf.getRow(y);
        for (int x = rc.x0; x < rc.x1; ++x) {
            // 取光源方向上下两个像素的亮度差
            int sx1 = std::max(rc.x0, std::min(rc.x1-1, x - dx));
            int sy1 = std::max(rc.y0, std::min(rc.y1-1, y - dy));
            int sx2 = std::max(rc.x0, std::min(rc.x1-1, x + dx));
            int sy2 = std::max(rc.y0, std::min(rc.y1-1, y + dy));

            float diff = (luminance(src.at(sx1,sy1)) -
                          luminance(src.at(sx2,sy2))) * strength;
            uint8_t v  = clamp8f(diff + 128.f);
            dst[x].r = dst[x].g = dst[x].b = v;
        }
    }
}

// 发光效果
// 算法：
//   1. 提取超过 threshold 亮度的像素 → 亮斑层
//   2. 对亮斑层进行高斯模糊
//   3. 将模糊后的亮斑叠加回原图（加法混合）
void bloom(Buffer& buf, float threshold, int radius, float intensity,
           const FilterRect& region) {
    auto  rc    = resolveRect(buf, region);
    float thresh = threshold * 255.f; // 转为 0~255 范围

    // Step 1：提取亮斑层
    Buffer bright(buf.width, buf.height, Color(0, 0, 0, 0));
    for (int y = rc.y0; y < rc.y1; ++y) {
        const Color* src = buf.getRow(y);
        Color*       dst = bright.getRow(y);
        for (int x = rc.x0; x < rc.x1; ++x) {
            float lum = luminance(src[x]);
            if (lum >= thresh) {
                // 亮斑强度按超出阈值比例缩放，减少硬边
                float t = std::min(1.f, (lum - thresh) / (255.f - thresh + 1e-6f));
                dst[x].r = clamp8f(src[x].r * t);
                dst[x].g = clamp8f(src[x].g * t);
                dst[x].b = clamp8f(src[x].b * t);
                dst[x].a = 255;
            }
        }
    }

    // Step 2：对亮斑层进行高斯模糊
    float sigma = radius / 3.f;
    gaussianBlur(bright, sigma, region);

    // Step 3：叠加回原图（加法混合，超出 255 截断）
    for (int y = rc.y0; y < rc.y1; ++y) {
        Color*       dst = buf.getRow(y);
        const Color* br  = bright.getRow(y);
        for (int x = rc.x0; x < rc.x1; ++x) {
            dst[x].r = clamp8(dst[x].r + (int)(br[x].r * intensity));
            dst[x].g = clamp8(dst[x].g + (int)(br[x].g * intensity));
            dst[x].b = clamp8(dst[x].b + (int)(br[x].b * intensity));
        }
    }
}

// 晕影
// 以图像中心为圆心，边缘越远越暗
// 使用平滑 smoothstep 过渡避免硬边
void vignette(Buffer& buf, float strength, float innerRadius,
              const FilterRect& region) {
    auto rc  = resolveRect(buf, region);
    float cx = (rc.x0 + rc.x1) * 0.5f;
    float cy = (rc.y0 + rc.y1) * 0.5f;
    // 最大距离（取到图像角落的距离，归一化用）
    float maxR = std::sqrt((float)((rc.x1-rc.x0)*(rc.x1-rc.x0) +
                                   (rc.y1-rc.y0)*(rc.y1-rc.y0))) * 0.5f;

    for (int y = rc.y0; y < rc.y1; ++y) {
        Color* row = buf.getRow(y);
        for (int x = rc.x0; x < rc.x1; ++x) {
            float dx   = (x - cx) / maxR;
            float dy   = (y - cy) / maxR;
            float dist = std::sqrt(dx*dx + dy*dy); // 0（中心） ~ 1+（角落）

            // smoothstep：在 innerRadius ~ 1.0 之间平滑过渡
            float t = (dist - innerRadius) / (1.f - innerRadius);
            t = std::max(0.f, std::min(1.f, t));
            float fade = t * t * (3.f - 2.f * t); // smoothstep 曲线
            float factor = 1.f - fade * strength;

            row[x].r = clamp8f(row[x].r * factor);
            row[x].g = clamp8f(row[x].g * factor);
            row[x].b = clamp8f(row[x].b * factor);
        }
    }
}

// 像素化/马赛克
// 将图像分成 blockSize × blockSize 的块，
// 每块取均色覆盖
void pixelate(Buffer& buf, int blockSize, const FilterRect& region) {
    if (blockSize <= 1) return;
    auto rc = resolveRect(buf, region);

    for (int by = rc.y0; by < rc.y1; by += blockSize) {
        for (int bx = rc.x0; bx < rc.x1; bx += blockSize) {
            // 计算当前块的实际范围
            int ex = std::min(bx + blockSize, rc.x1);
            int ey = std::min(by + blockSize, rc.y1);
            int count = (ex - bx) * (ey - by);
            if (count == 0) continue;

            // 计算块内均色
            long sr = 0, sg = 0, sb = 0, sa = 0;
            for (int y = by; y < ey; ++y) {
                const Color* row = buf.getRow(y);
                for (int x = bx; x < ex; ++x) {
                    sr += row[x].r; sg += row[x].g;
                    sb += row[x].b; sa += row[x].a;
                }
            }
            Color avg;
            avg.r = (uint8_t)(sr / count);
            avg.g = (uint8_t)(sg / count);
            avg.b = (uint8_t)(sb / count);
            avg.a = (uint8_t)(sa / count);

            // 用均色填充整个块
            for (int y = by; y < ey; ++y) {
                Color* row = buf.getRow(y);
                for (int x = bx; x < ex; ++x) row[x] = avg;
            }
        }
    }
}

// 扫描线|模拟效果
// 每隔 spacing 行叠加一条暗色横条（模拟 CRT 屏幕）
void scanlines(Buffer& buf, float strength, int spacing,
               const FilterRect& region) {
    if (spacing < 1) spacing = 1;
    auto  rc   = resolveRect(buf, region);
    float dark = 1.f - strength; // 暗条像素的亮度系数

    for (int y = rc.y0; y < rc.y1; ++y) {
        // 只对偶数行（或间隔行）变暗
        if ((y / spacing) % 2 == 0) continue;
        Color* row = buf.getRow(y);
        for (int x = rc.x0; x < rc.x1; ++x) {
            row[x].r = clamp8f(row[x].r * dark);
            row[x].g = clamp8f(row[x].g * dark);
            row[x].b = clamp8f(row[x].b * dark);
        }
    }
}

} // namespace filter

// Filter::apply 实现（将高层描述对象派发到底层函数）
void Filter::apply(Buffer& buf) const {
    using T = Filter::Type;
    switch (type) {
    case T::Grayscale:   filter::grayscale(buf, region); break;
    case T::Invert:      filter::invert(buf, region); break;
    case T::Brightness:  filter::brightness(buf, f1, region); break;
    case T::Contrast:    filter::contrast(buf, f1, region); break;
    case T::Saturation:  filter::saturation(buf, f1, region); break;
    case T::HueRotate:   filter::hueRotate(buf, f1, region); break;
    case T::ColorTint:   filter::colorTint(buf, color, f1, region); break;
    case T::Sepia:       filter::sepia(buf, f1, region); break;
    case T::Opacity:     filter::opacity(buf, f1, region); break;
    case T::BoxBlur:     filter::boxBlur(buf, i1, region); break;
    case T::GaussianBlur:filter::gaussianBlur(buf, f1, region); break;
    case T::RadialBlur:  filter::radialBlur(buf, f1, -1, -1, region); break;
    case T::Sharpen:     filter::sharpen(buf, f1, region); break;
    case T::EdgeDetect:  filter::edgeDetect(buf, region); break;
    case T::Emboss:      filter::emboss(buf, f1, f2, region); break;
    case T::Bloom:       filter::bloom(buf, f1, i1, f2, region); break;
    case T::Vignette:    filter::vignette(buf, f1, f2, region); break;
    case T::Pixelate:    filter::pixelate(buf, i1, region); break;
    case T::Scanlines:   filter::scanlines(buf, f1, i1, region); break;
    }
}

} // namespace pa2d
