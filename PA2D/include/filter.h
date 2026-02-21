#pragma once
#include "buffer.h"
#include "color.h"

namespace pa2d {

// ── 滤镜作用区域（可选，默认整个缓冲区）─────────────────────
struct FilterRect {
    int x = 0, y = 0, w = -1, h = -1; // -1 表示延伸到边界
};

namespace filter {


    // 灰度化：将彩色图像转为灰度（保留 alpha）
    // 使用 ITU-R BT.601 加权公式：gray = 0.299R + 0.587G + 0.114B
    void grayscale(Buffer& buf, const FilterRect& region = {});

    // 反色：每个通道取补值（255 - 值），不影响 alpha
    void invert(Buffer& buf, const FilterRect& region = {});

    // 亮度调节：factor > 1 增亮，< 1 变暗，= 1 不变
    // 范围建议：[0.0f, 3.0f]
    void brightness(Buffer& buf, float factor, const FilterRect& region = {});

    // 对比度调节：factor > 1 增强，< 1 减弱，= 1 不变
    // 使用以 128 为中心的线性变换
    void contrast(Buffer& buf, float factor, const FilterRect& region = {});

    // 饱和度调节：factor = 0 → 灰度，= 1 → 不变，> 1 → 过饱和
    void saturation(Buffer& buf, float factor, const FilterRect& region = {});

    // 色调旋转：angle 为 HSV 色相旋转角度（0~360 度）
    void hueRotate(Buffer& buf, float angle, const FilterRect& region = {});

    // 颜色叠加：将指定颜色与图像混合（intensity = 0~1）
    void colorTint(Buffer& buf, const Color& tint, float intensity = 0.5f,
                   const FilterRect& region = {});

    // 棕褐色滤镜（怀旧/复古效果）intensity = 0~1，1 = 完全棕褐
    void sepia(Buffer& buf, float intensity = 1.0f, const FilterRect& region = {});

    // Alpha 通道调节：factor = 0 → 完全透明，= 1 → 不变
    void opacity(Buffer& buf, float factor, const FilterRect& region = {});


    // 盒式模糊（Box Blur）：速度极快，适合大半径
    // radius：模糊半径（像素），建议 1~20
    void boxBlur(Buffer& buf, int radius = 3, const FilterRect& region = {});

    // 高斯模糊：质量更高，使用可分离卷积（两次一维高斯）
    // sigma：标准差，越大越模糊；radius 通常取 ceil(3*sigma)
    void gaussianBlur(Buffer& buf, float sigma = 1.5f, const FilterRect& region = {});

    // 径向（径向）模糊：以中心点向外缩放模糊，产生"速度感"
    // cx/cy：中心坐标（-1 表示图像中心）；strength：模糊强度 0~1
    void radialBlur(Buffer& buf, float strength = 0.3f,
                    int cx = -1, int cy = -1, const FilterRect& region = {});


    // 锐化（Unsharp Mask）：减去轻度模糊版本再叠加
    // strength：锐化强度 0~3
    void sharpen(Buffer& buf, float strength = 1.0f, const FilterRect& region = {});

    // 边缘检测（Sobel 算子）：突出边缘，背景变黑
    // 结果为灰度图
    void edgeDetect(Buffer& buf, const FilterRect& region = {});

    // 浮雕效果：产生立体压印感
    // angle：光源方向（度），strength：强度 0~3
    void emboss(Buffer& buf, float angle = 135.0f, float strength = 1.0f,
                const FilterRect& region = {});

    void bloom(Buffer& buf, float threshold = 0.65f, int radius = 8,
               float intensity = 0.8f, const FilterRect& region = {});

    void vignette(Buffer& buf, float strength = 0.5f, float innerRadius = 0.4f,
                  const FilterRect& region = {});

    void pixelate(Buffer& buf, int blockSize = 8, const FilterRect& region = {});

    void scanlines(Buffer& buf, float strength = 0.3f, int spacing = 2,
                   const FilterRect& region = {});

} // namespace filter

struct Filter {
    // 滤镜类型枚举（内部使用）
    enum class Type {
        Grayscale, Invert,
        Brightness, Contrast, Saturation, HueRotate,
        ColorTint, Sepia, Opacity,
        BoxBlur, GaussianBlur, RadialBlur,
        Sharpen, EdgeDetect, Emboss,
        Bloom, Vignette, Pixelate, Scanlines
    };

    Type     type;
    float    f1 = 0, f2 = 0, f3 = 0;   // 浮点参数（含义因滤镜而异）
    int      i1 = 0, i2 = 0;            // 整型参数
    Color    color = 0;                  // 颜色参数（用于 ColorTint）
    FilterRect region;                   // 作用区域

    // ── 工厂函数（静态创建接口，风格与 PA2D 一致）─────────

    static Filter Grayscale()                                { return {Type::Grayscale}; }
    static Filter Invert()                                   { return {Type::Invert}; }
    static Filter Brightness(float factor)                   { return {Type::Brightness, factor}; }
    static Filter Contrast(float factor)                     { return {Type::Contrast,   factor}; }
    static Filter Saturation(float factor)                   { return {Type::Saturation, factor}; }
    static Filter HueRotate(float angle)                     { return {Type::HueRotate,  angle}; }
    static Filter Sepia(float intensity = 1.0f)              { return {Type::Sepia, intensity}; }
    static Filter Opacity(float factor)                      { return {Type::Opacity, factor}; }
    static Filter ColorTint(const Color& c, float intensity) {
        Filter f{Type::ColorTint, intensity};
        f.color = c;
        return f;
    }
    static Filter BoxBlur(int radius = 3)                    { Filter f{Type::BoxBlur};    f.i1 = radius; return f; }
    static Filter GaussianBlur(float sigma = 1.5f)           { return {Type::GaussianBlur, sigma}; }
    static Filter RadialBlur(float strength = 0.3f)          { return {Type::RadialBlur,   strength}; }
    static Filter Sharpen(float strength = 1.0f)             { return {Type::Sharpen,      strength}; }
    static Filter EdgeDetect()                               { return {Type::EdgeDetect}; }
    static Filter Emboss(float angle = 135.0f, float strength = 1.0f) {
        return {Type::Emboss, angle, strength};
    }
    static Filter Bloom(float threshold = 0.65f, int radius = 8, float intensity = 0.8f) {
        Filter f{Type::Bloom, threshold, intensity};
        f.i1 = radius;
        return f;
    }
    static Filter Vignette(float strength = 0.5f, float innerRadius = 0.4f) {
        return {Type::Vignette, strength, innerRadius};
    }
    static Filter Pixelate(int blockSize = 8)                { Filter f{Type::Pixelate};   f.i1 = blockSize; return f; }
    static Filter Scanlines(float strength = 0.3f, int spacing = 2) {
        Filter f{Type::Scanlines, strength};
        f.i1 = spacing;
        return f;
    }

    // 限制作用区域（支持链式：Filter::Blur(3).in({x, y, w, h})）
    Filter& in(const FilterRect& r) { region = r; return *this; }

    // 将此 Filter 描述应用到 Buffer
    void apply(Buffer& buf) const;
};

} // namespace pa2d
