#pragma once
namespace pa2d {
    struct Buffer;
    void blit(const Buffer& src, Buffer& dst, int dstX = 0, int dstY = 0);
    void alphaBlend(const Buffer& src, Buffer& dst, int dstX = 0, int dstY = 0, int opacity = 255);
    void addBlend(const Buffer& src, Buffer& dst, int dstX = 0, int dstY = 0, int opacity = 255);
    void multiplyBlend(const Buffer& src, Buffer& dst, int dstX = 0, int dstY = 0, int opacity = 255);
    void screenBlend(const Buffer& src, Buffer& dst, int dstX = 0, int dstY = 0, int opacity = 255);
    void overlayBlend(const Buffer& src, Buffer& dst, int dstX = 0, int dstY = 0, int opacity = 255);
    void destAlphaBlend(const Buffer& src, Buffer& dst, int dstX = 0, int dstY = 0, int opacity = 255);
} // namespace pa2d