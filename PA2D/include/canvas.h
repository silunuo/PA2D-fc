#pragma once
#include "color.h"
#include "buffer.h"
#include "style.h"
#include "draw.h"
#include "buffer_blender.h"
#include "image_loader.h"
#include "draw_text.h"
#include "geometry.h"
#include <vector>
namespace pa2d {
    class Canvas {
    private:
        Buffer buffer_;
        Canvas& drawImpl(const Line& line, const Style& style);
        Canvas& drawImpl(const Ray& ray, const Style& style);
        Canvas& drawImpl(const Triangle& triangle, const Style& style);
        Canvas& drawImpl(const Rect& rect, const Style& style);
        Canvas& drawImpl(const Circle& circle, const Style& style);
        Canvas& drawImpl(const Elliptic& ellipse, const Style& style);
        Canvas& drawImpl(const Sector& sector, const Style& style);
        Canvas& drawImpl(const Points& points, const Style& style);
        Canvas& drawImpl(const Polygon& polygon, const Style& style);
    public:
        // 构造函数
        Canvas();
        Canvas(int width, int height, Color background = 0xFFFFFFFF);
        Canvas(const Buffer& rhs);
        Canvas(Buffer&& rhs) noexcept;
        Canvas(const Canvas& other);
        Canvas(Canvas&& other) noexcept;
        Canvas& operator=(Canvas&& rhs) noexcept;
        Canvas& operator=(Buffer&& rhs) noexcept;
        Canvas(const char* filePath);
        Canvas(int resourceID);
        Canvas& operator=(const Canvas& rhs);
        Canvas& operator=(const Buffer& rhs);
        // 基础属性访问
        int width() const;
        int height() const;
        bool isValid() const;
        const Buffer& getBuffer() const;
        Buffer& getBuffer();
        // 像素访问
        Color& at(int x, int y);
        const Color& at(int x, int y) const;
        // 图像操作
        bool loadImage(const char* filePath, int width = -1, int height = -1);
        bool loadImage(int resourceID, int width = -1, int height = -1);
        Canvas& clear(Color color = 0xFFFFFFFF);
        Canvas& blit(const Canvas& src, int DstX = 0, int DstY = 0);
        Canvas& crop(int x, int y, int width, int height);
        Canvas& resize(int width, int height);
        Canvas& resizeBuffer(int width, int height, uint32_t clearColor = 0xFFFFFFFF);
        // 混合操作
        Canvas& blend(const Canvas& src, int x = 0, int y = 0, int alpha = 255, int Mode = 0);
        Canvas& alphaBlend(const Canvas& src, int dstX = 0, int dstY = 0, int alpha = 255);
        Canvas& addBlend(const Canvas& src, int dstX = 0, int dstY = 0, int alpha = 255);
        Canvas& multiplyBlend(const Canvas& src, int dstX = 0, int dstY = 0, int alpha = 255);
        Canvas& screenBlend(const Canvas& src, int dstX = 0, int dstY = 0, int alpha = 255);
        Canvas& overlayBlend(const Canvas& src, int dstX = 0, int dstY = 0, int alpha = 255);
        Canvas& destAlphaBlend(const Canvas& src, int dstX = 0, int dstY = 0, int alpha = 255);
        // 变换操作
        Canvas& draw(const Canvas& src, float centerX, float centerY, int alpha = 255);
        Canvas& drawRotated(const Canvas& src, float centerX, float centerY, float rotation);
        Canvas& drawScaled(const Canvas& src, float centerX, float centerY, float scaleX, float scaleY);
        Canvas& drawResized(const Canvas& src, float centerX, float centerY, int width, int height);
        Canvas& drawScaled(const Canvas& src, float centerX, float centerY, float scale);
        Canvas& drawTransformed(const Canvas& src, float centerX, float centerY, float scale, float rotation);
        Canvas& drawTransformed(const Canvas& src, float centerX, float centerY, float scaleX, float scaleY, float rotation);
        // 创建变换后的副本
        Canvas cropped(int x, int y, int width, int height) const;
        Canvas scaled(float scaleX, float scaleY) const;
        Canvas resized(int width, int height) const;
        Canvas scaled(float scale) const;
        Canvas rotated(float rotation) const;
        Canvas transformed(float scale, float rotation) const;
        Canvas transformed(float scaleX, float scaleY, float rotation) const;
        // 基础绘制方法
        Canvas& rect(float x, float y, float width, float height, const Style& style);
        Canvas& rect(float centerX, float centerY, float width, float height, float angle, const Style& style);
        Canvas& circle(float centerX, float centerY, float radius, const Style& style);
        Canvas& ellipse(float cx, float cy, float width, float height, const Style& style);
        Canvas& ellipse(float cx, float cy, float width, float height, float angle, const Style& style);
        Canvas& triangle(float ax, float ay, float bx, float by, float cx, float cy, const Style& style);
        Canvas& sector(float cx, float cy, float radius, float startAngleDeg, float endAngleDeg, const Style& style);
        Canvas& line(float x0, float y0, float x1, float y1, const Style& style);
        Canvas& polyline(const std::vector<Point>& points, const Style& style, bool closed = false);
        Canvas& polygon(const std::vector<Point>& vertices, const Style& style);
        // 智能绘制函数
        Canvas& draw(const Shape& shape, const Style& style);
        // 文本绘制方法
        Canvas& text(int x, int y, const std::wstring& text, int fontSize = 16, const Color& color = 0xFF000000, FontStyle style = FontStyle::Regular, const std::wstring& fontName = L"Microsoft YaHei");
        Canvas& text(int x, int y, const std::string& text, int fontSize = 16, const Color& color = 0xFF000000, FontStyle style = FontStyle::Regular, const std::string& fontName = "Microsoft YaHei");
        Canvas& textCentered(int centerX, int centerY, const std::wstring& text, int fontSize = 16, const Color& color = 0xFF000000, FontStyle style = FontStyle::Regular, const std::wstring& fontName = L"Microsoft YaHei");
        Canvas& textCentered(int centerX, int centerY, const std::string& text, int fontSize = 16, const Color& color = 0xFF000000, FontStyle style = FontStyle::Regular, const std::string& fontName = "Microsoft YaHei");
        Canvas& textInRect(int rectX, int rectY, int rectWidth, int rectHeight, const std::wstring& text, int fontSize = 16, const Color& color = 0xFF000000, FontStyle style = FontStyle::Regular, const std::wstring& fontName = L"Microsoft YaHei");
        Canvas& textInRect(int rectX, int rectY, int rectWidth, int rectHeight, const std::string& text, int fontSize = 16, const Color& color = 0xFF000000, FontStyle style = FontStyle::Regular, const std::string& fontName = "Microsoft YaHei");
        Canvas& textFitRect(int rectX, int rectY, int rectWidth, int rectHeight, const std::wstring& text, int preferredFontSize = 16, const Color& color = 0xFF000000, FontStyle style = FontStyle::Regular, const std::wstring& fontName = L"Microsoft YaHei");
        Canvas& textFitRect(int rectX, int rectY, int rectWidth, int rectHeight, const std::string& text, int preferredFontSize = 16, const Color& color = 0xFF000000, FontStyle style = FontStyle::Regular, const std::string& fontName = "Microsoft YaHei");
    };
} // namespace pa2d