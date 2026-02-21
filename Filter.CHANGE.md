# 添加滤镜模块的改动说明

## 概览

在 `extensions/` 模块下新增图像滤镜系统，提供多种后处理滤镜。

---

## 新增文件

### `extensions/filter/filter.h`
滤镜系统的对外接口头文件，包含：
- `pa2d::filter::` 命名空间下的滤镜函数
- `FilterRect` 结构体，用于限定滤镜生效的局部区域
- `Filter` 描述符类，支持链式调用 `.in(rect).apply(buffer)`

### `extensions/filter/filter.cpp`
滤镜的具体实现，包含：
- 可分离卷积优化（两趟1D卷积，复杂度 O(r) 而非 O(r²)）
- RGB ↔ HSV 色彩空间转换（用于色相/饱和度滤镜）
- 高斯核动态生成
- `Filter::apply()` 统一分发调度

---

## 修改文件

### `extensions/CMakeLists.txt`
- GLOB 范围追加 `filter/filter.cpp`
- 追加 `target_include_directories`，使 `#include "filter/filter.h"` 路径可正确解析

---

## 滤镜列表（共19种）

| 分类 | 滤镜 |
|------|------|
| 色彩调整 | `grayscale` `invert` `brightness` `contrast` `saturation` `hueRotate` `sepia` `colorTint` `opacity` |
| 模糊 | `boxBlur` `gaussianBlur` `radialBlur` |
| 锐化/边缘 | `sharpen` `edgeDetect` `emboss` |
| 特效 | `bloom` `vignette` `pixelate` `scanlines` |

---

## 使用方式

```cpp
#include "filter/filter.h"
using namespace pa2d;

// 直接调用命名空间函数
filter::gaussianBlur(canvas.getBuffer(), 2.0f);

// 使用描述符对象，限定局部区域
Filter::Bloom(0.65f, 8).in({0, 0, 400, 300}).apply(canvas.getBuffer());

// 叠加滤镜（怀旧效果）
filter::sepia(canvas.getBuffer(), 0.85f);
filter::vignette(canvas.getBuffer(), 0.6f);
filter::scanlines(canvas.getBuffer(), 0.2f);
```

---

## 跨编译器

  - `extensions/compiler/include/compiler_compat.h` — 跨编译器兼容层，统一 MSVC 与 GCC/MinGW-w64 差异 
  （`PA2D_FORCEINLINE`、`wcsncpy_s` 包装、`sprintf_s` 映射、`NOMINMAX` 保护）