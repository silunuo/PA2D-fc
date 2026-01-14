#pragma once
#include "color.h"
namespace pa2d {
    struct Buffer {
        Color* color;
        int width, height;
        explicit Buffer(int width = 0, int height = 0, const Color& init_color = 0x0);
        Buffer(const Buffer& other);
        Buffer(Buffer&& other) noexcept;
        Buffer& operator=(const Buffer& other);
        Buffer& operator=(Buffer&& other) noexcept;
        ~Buffer();
        size_t size() const { return static_cast<size_t>(width) * height; }
        bool isValid() const { return color && width > 0 && height > 0; }
        Color& at(int x, int y);
        const Color& at(int x, int y) const;
        Color* getRow(int y);
        const Color* getRow(int y) const;
        void resize(int newWidth, int newHeight, const Color& clear_color = 0x0);
        void clear(const Color& clear_color = 0x0);
        explicit operator bool() const { return isValid(); }
    };
    void copy(Buffer& dest, const Buffer& src);
    Buffer crop(const Buffer& src, int x, int y, int w, int h);
    void drawScaled(Buffer& dest, const Buffer& src, float centerX, float centerY, float scaleX, float scaleY);
    void drawScaled(Buffer& dest, const Buffer& src, float centerX, float centerY, float scale);
    void drawResized(Buffer& dest, const Buffer& src, float centerX, float centerY , int width, int height);
    void drawRotated(Buffer& dest, const Buffer& src, float centerX, float centerY, float rotation);
    void drawTransformed(Buffer& dest, const Buffer& src, float centerX, float centerY, float scale, float rotation);
    void drawTransformed(Buffer& dest, const Buffer& src, float centerX, float centerY, float scaleX, float scaleY, float rotation);
    Buffer scaled(const Buffer& src, float scaleX, float scaleY);
    Buffer scaled(const Buffer& src, float factor);
    Buffer resized(const Buffer& src, int width, int height); 
    Buffer rotated(const Buffer& src, float rotation);
    Buffer transformed(const Buffer& src, float scale,float rotation);
    Buffer transformed(const Buffer& src, float scaleX, float scaleY, float rotation );
} // namespace pa2d