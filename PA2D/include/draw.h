#pragma once

#include "color.h"
#include "buffer.h"
#include "geometry/point.h"
#include <vector>

namespace pa2d {
    void line(Buffer& buffer, float startX, float startY, float endX, float endY, const Color& color, float width);
    void polyline(Buffer& buffer, const std::vector<Point>& points, const Color& color, float width, bool closed);
    void polygon(Buffer& buffer, const std::vector<Point>& vertices, const Color& fillColor, const Color& strokeColor, float strokeWidth);
    void triangle(Buffer& buffer, float x0, float y0, float x1, float y1, float x2, float y2, const Color& fillColor, const Color& strokeColor, float strokeWidth);
    void rect(Buffer& buffer, float left, float top, float width, float height, const Color& fillColor, const Color& strokeColor, float strokeWidth);
    void rect(Buffer& buffer, float centerX, float centerY, float width, float height, float angle, const Color& fillColor, const Color& strokeColor, float strokeWidth);
    void roundRect(Buffer& buffer, float left, float top, float width, float height, const Color& fillColor, const Color& strokeColor, float radius, float strokeWidth);
    void roundRect(Buffer& buffer, float centerX, float centerY, float width, float height, float angle, const Color& fillColor, const Color& strokeColor, float radius, float strokeWidth);
    void circle(Buffer& buffer, float centerX, float centerY, float radius, const Color& fillColor, const Color& strokeColor, float strokeWidth);
    void ellipse(Buffer& buffer, float centerX, float centerY, float width, float height, const Color& fillColor, const Color& strokeColor, float strokeWidth);
    void ellipse(Buffer& buffer, float centerX, float centerY, float width, float height, float angle, const Color& fillColor, const Color& strokeColor, float strokeWidth);
    void sector(Buffer& buffer, float centerX, float centerY, float radius, float startAngle, float endAngle, const Color& fillColor, const Color& strokeColor, float strokeWidth, bool arc, bool edges);
} // namespace pa2d