#include "../include/canvas.h"
#include <algorithm>
#include <cassert>

namespace pa2d {
    Canvas::Canvas() : buffer_(0, 0, Color(0x00000000)) {}

    Canvas::Canvas(int width, int height, Color background)
        : buffer_(width, height, Color(background)) {
    }

    Canvas::Canvas(const Buffer& rhs) : buffer_(rhs) {}

    Canvas::Canvas(Buffer&& rhs) noexcept : buffer_(std::move(rhs)) {}

    Canvas::Canvas(const Canvas& rhs) : buffer_(rhs.buffer_) {}

    Canvas::Canvas(Canvas&& rhs) noexcept : buffer_(std::move(rhs.buffer_)) {}

    Canvas& Canvas::operator=(Canvas&& rhs) noexcept {
        buffer_ = std::move(rhs.buffer_);
        return *this;
    }

    Canvas& Canvas::operator=(Buffer&& rhs) noexcept {
        buffer_ = std::move(rhs);
        return *this;
    }

    Canvas::Canvas(const char* filePath) : buffer_(0, 0, Color(0x00000000)) {
        loadImage(filePath);
    }

    Canvas::Canvas(int resourceID) : buffer_(0, 0, Color(0x00000000)) {
        loadImage(resourceID);
    }

    Canvas& Canvas::operator=(const Canvas& rhs) {
        buffer_ = rhs.buffer_;
        return *this;
    }

    Canvas& Canvas::operator=(const Buffer& rhs) {
        this->buffer_ = rhs;
        return *this;
    }

    int Canvas::width() const { return buffer_.width; }

    int Canvas::height() const { return buffer_.height; }

    bool Canvas::isValid() const { return buffer_.isValid(); }

    const Buffer& Canvas::getBuffer() const { return buffer_; }

    Buffer& Canvas::getBuffer() { return buffer_; }

    Color& Canvas::at(int x, int y) { return buffer_.at(x, y); }

    const Color& Canvas::at(int x, int y) const { return buffer_.at(x, y); }

    bool Canvas::loadImage(const char* filePath, int width, int height) {
        if (width < 0 || height < 0)
            return pa2d::loadImage(buffer_, filePath);
        else {
            if (pa2d::loadImage(buffer_, filePath)) {
                *this = resized(width, height);
                return true;
            }
            else {
                return false;
            }
        }
    }

    bool Canvas::loadImage(int resourceID, int width, int height) {
        if (width < 0 || height < 0)
            return pa2d::loadImage(buffer_, resourceID);
        else {
            if (pa2d::loadImage(buffer_, resourceID)) {
                *this = resized(width, height);
                return true;
            }
            else {
                return false;
            }
        }
    }

    Canvas& Canvas::clear(Color color) {
        buffer_.clear(Color(color));
        return *this;
    }

    Canvas& Canvas::blit(const Canvas& src, int DstX, int DstY) {
        pa2d::blit(src.buffer_, buffer_, DstX, DstY);
        return *this;
    }

    Canvas& Canvas::resizeBuffer(int newWidth, int newHeight, uint32_t clearColor) {
        buffer_.resize(newWidth, newHeight, Color(clearColor));
        return *this;
    }

    Canvas& Canvas::resize(int width, int height) {
        *this = resized(width, height);
        return *this;
    }

    Canvas& Canvas::blend(const Canvas& src, int x, int y, int alpha, int Mode) {
        switch (Mode) {
        case 0: return alphaBlend(src, x, y, alpha);
        case 1: return addBlend(src, x, y, alpha);
        case 2: return multiplyBlend(src, x, y, alpha);
        case 3: return screenBlend(src, x, y, alpha);
        case 4: return overlayBlend(src, x, y, alpha);
        case 5: return destAlphaBlend(src, x, y, alpha);
        default:
            break;
        }
    }

    Canvas& Canvas::alphaBlend(const Canvas& src, int dstX, int dstY, int alpha) {
        pa2d::alphaBlend(src.buffer_, buffer_, dstX, dstY, alpha);
        return *this;
    }

    Canvas& Canvas::addBlend(const Canvas& src, int dstX, int dstY, int alpha) {
        pa2d::addBlend(src.buffer_, buffer_, dstX, dstY, alpha);
        return *this;
    }

    Canvas& Canvas::multiplyBlend(const Canvas& src, int dstX, int dstY, int alpha) {
        pa2d::multiplyBlend(src.buffer_, buffer_, dstX, dstY, alpha);
        return *this;
    }

    Canvas& Canvas::screenBlend(const Canvas& src, int dstX, int dstY, int alpha) {
        pa2d::screenBlend(src.buffer_, buffer_, dstX, dstY, alpha);
        return *this;
    }

    Canvas& Canvas::overlayBlend(const Canvas& src, int dstX, int dstY, int alpha) {
        pa2d::overlayBlend(src.buffer_, buffer_, dstX, dstY, alpha);
        return *this;
    }

    Canvas& Canvas::destAlphaBlend(const Canvas& src, int dstX, int dstY, int alpha) {
        pa2d::destAlphaBlend(src.buffer_, buffer_, dstX, dstY, alpha);
        return *this;
    }

    Canvas& Canvas::crop(int x, int y, int width, int height) {
        buffer_ = pa2d::crop(buffer_, x, y, width, height);
        return *this;
    }

    Canvas& Canvas::draw(const Canvas& src, float centerX, float centerY, int alpha) {
        return alphaBlend(src, centerX - src.width() / 2, centerY - src.height() / 2, alpha);
    }

    Canvas& Canvas::drawRotated(const Canvas& src, float centerX, float centerY, float rotation) {
        pa2d::drawRotated(buffer_, src.buffer_, centerX, centerY, rotation);
        return *this;
    }

    Canvas& Canvas::drawScaled(const Canvas& src, float centerX, float centerY, float scaleX, float scaleY) {
        pa2d::drawScaled(buffer_, src.buffer_, centerX, centerY, scaleX, scaleY);
        return *this;
    }

    Canvas& Canvas::drawResized(const Canvas& src, float centerX, float centerY, int width, int height) {
        pa2d::drawResized(buffer_, src.buffer_, centerX, centerY, width, height);
        return *this;
    }

    Canvas& Canvas::drawScaled(const Canvas& src, float centerX, float centerY, float scale) {
        pa2d::drawScaled(buffer_, src.buffer_, centerX, centerY, scale);
        return *this;
    }

    Canvas& Canvas::drawTransformed(const Canvas& src, float centerX, float centerY, float scale, float rotation) {
        pa2d::drawTransformed(buffer_, src.buffer_, centerX, centerY, scale, rotation);
        return *this;
    }

    Canvas& Canvas::drawTransformed(const Canvas& src, float centerX, float centerY, float scaleX, float scaleY, float rotation) {
        pa2d::drawTransformed(buffer_, src.buffer_, centerX, centerY, scaleX, scaleY, rotation);
        return *this;
    }

    Canvas Canvas::cropped(int x, int y, int width, int height) const {
        if (!buffer_.isValid()) {
            return Canvas();
        }

        x = std::max(0, std::min(x, buffer_.width - 1));
        y = std::max(0, std::min(y, buffer_.height - 1));
        width = std::max(1, std::min(width, buffer_.width - x));
        height = std::max(1, std::min(height, buffer_.height - y));

        Buffer croppedBuffer = pa2d::crop(buffer_, x, y, width, height);
        return Canvas(std::move(croppedBuffer));
    }

    Canvas Canvas::scaled(float scaleX, float scaleY) const {
        if (!buffer_.isValid() || scaleX <= 0.0f || scaleY <= 0.0f) {
            return Canvas();
        }

        Buffer scaledBuffer = pa2d::scaled(buffer_, scaleX, scaleY);
        return Canvas(std::move(scaledBuffer));
    }

    Canvas Canvas::resized(int width, int height) const {
        if (!buffer_.isValid() || width <= 0 || height <= 0) {
            return Canvas();
        }
        Buffer scaledBuffer = pa2d::resized(buffer_, width, height);
        return Canvas(std::move(scaledBuffer));
    }

    Canvas Canvas::scaled(float scale) const {
        if (!buffer_.isValid() || scale <= 0.0f) {
            return Canvas();
        }
        Buffer scaledBuffer = pa2d::scaled(buffer_, scale);
        return Canvas(std::move(scaledBuffer));
    }

    Canvas Canvas::rotated(float rotation) const {
        if (!buffer_.isValid()) {
            return Canvas();
        }

        Buffer rotated = pa2d::rotated(buffer_, rotation);
        return Canvas(std::move(rotated));
    }

    Canvas Canvas::transformed(float scale, float rotation) const {
        if (!buffer_.isValid() || scale <= 0.0f) {
            return Canvas();
        }

        Buffer resultBuffer = pa2d::transformed(buffer_, scale, rotation);
        return Canvas(std::move(resultBuffer));
    }

    Canvas Canvas::transformed(float scaleX, float scaleY, float rotation) const {
        if (!buffer_.isValid() || scaleX <= 0.0f || scaleY <= 0.0f) {
            return Canvas();
        }

        Buffer resultBuffer = pa2d::transformed(buffer_, scaleX, scaleY, rotation);
        return Canvas(std::move(resultBuffer));
    }

    Canvas& Canvas::rect(float x, float y, float width, float height, const Style& style) {
        pa2d::roundRect(buffer_, x, y, width, height,
            style.fill_, style.stroke_, style.radius_, style.width_);
        return *this;
    }

    Canvas& Canvas::rect(float centerX, float centerY, float width, float height,
        float angle, const Style& style) {
        pa2d::roundRect(buffer_, centerX, centerY, width, height, angle,
            style.fill_, style.stroke_, style.radius_, style.width_);
        return *this;
    }

    Canvas& Canvas::circle(float centerX, float centerY, float radius,
        const Style& style) {
        pa2d::circle(buffer_, centerX, centerY, radius,
            style.fill_, style.stroke_, style.width_);
        return *this;
    }

    Canvas& Canvas::ellipse(float cx, float cy, float width, float height,
        const Style& style) {
        pa2d::ellipse(buffer_, cx, cy, width, height,
            style.fill_, style.stroke_, style.width_);
        return *this;
    }

    Canvas& Canvas::ellipse(float cx, float cy, float width, float height,
        float angle, const Style& style) {
        pa2d::ellipse(buffer_, cx, cy, width, height, angle,
            style.fill_, style.stroke_, style.width_);
        return *this;
    }

    Canvas& Canvas::triangle(float ax, float ay, float bx, float by,
        float cx, float cy, const Style& style) {
        pa2d::triangle(buffer_, ax, ay, bx, by, cx, cy,
            style.fill_, style.stroke_, style.width_);
        return *this;
    }

    Canvas& Canvas::sector(float cx, float cy, float radius,
        float startAngleDeg, float endAngleDeg,
        const Style& style) {
        pa2d::sector(buffer_, cx, cy, radius, startAngleDeg, endAngleDeg,
            style.fill_, style.stroke_, style.width_,
            style.arc_, style.edges_);
        return *this;
    }

    Canvas& Canvas::line(float x0, float y0, float x1, float y1,
        const Style& style) {
        pa2d::line(buffer_, x0, y0, x1, y1,
            style.stroke_, style.width_);
        return *this;
    }

    Canvas& Canvas::polyline(const std::vector<Point>& points,
        const Style& style, bool closed) {
        pa2d::polyline(buffer_, points, style.stroke_,
            style.width_, closed);
        return *this;
    }

    Canvas& Canvas::polygon(const std::vector<Point>& vertices,
        const Style& style) {
        pa2d::polygon(buffer_, vertices,
            style.fill_, style.stroke_, style.width_);
        return *this;
    }

    Canvas& Canvas::draw(const Shape& shape, const Style& style) {
        switch (shape.getType()) {
        case Shape::GeometryType::POINTS: {
            const Points& points = static_cast<const Points&>(shape);
            return drawImpl(points, style);
        }
        case Shape::GeometryType::LINE: {
            const Line& line = static_cast<const Line&>(shape);
            return drawImpl(line, style);
        }
        case Shape::GeometryType::POLYGON: {
            const Polygon& polygon = static_cast<const Polygon&>(shape);
            return drawImpl(polygon, style);
        }
        case Shape::GeometryType::RECT: {
            const Rect& rect = static_cast<const Rect&>(shape);
            return drawImpl(rect, style);
        }
        case Shape::GeometryType::TRIANGLE: {
            const Triangle& triangle = static_cast<const Triangle&>(shape);
            return drawImpl(triangle, style);
        }
        case Shape::GeometryType::CIRCLE: {
            const Circle& circle = static_cast<const Circle&>(shape);
            return drawImpl(circle, style);
        }
        case Shape::GeometryType::ELLIPTIC: {
            const Elliptic& ellipse = static_cast<const Elliptic&>(shape);
            return drawImpl(ellipse, style);
        }
        case Shape::GeometryType::SECTOR: {
            const Sector& sector = static_cast<const Sector&>(shape);
            return drawImpl(sector, style);
        }
        case Shape::GeometryType::RAY: {
            const Ray& ray = static_cast<const Ray&>(shape);
            return drawImpl(ray, style);
        }
        default: {
#ifdef _DEBUG
            assert(false && "Unknown shape type in Canvas::draw");
#endif
            return *this;
        }
        }
    }

    Canvas& Canvas::drawImpl(const Line& line, const Style& style) {
        return this->line(line.start().x, line.start().y,
            line.end().x, line.end().y, style);
    }

    Canvas& Canvas::drawImpl(const Ray& ray, const Style& style) {
        return line(ray.start().x, ray.start().y,
            ray.end().x, ray.end().y, style);
    }

    Canvas& Canvas::drawImpl(const Triangle& triangle, const Style& style) {
        return this->triangle(triangle[0].x, triangle[0].y,
            triangle[1].x, triangle[1].y,
            triangle[2].x, triangle[2].y, style);
    }

    Canvas& Canvas::drawImpl(const Rect& rect, const Style& style) {
        return this->rect(rect.center().x, rect.center().y,
            rect.width(), rect.height(),
            rect.rotation(), style);
    }

    Canvas& Canvas::drawImpl(const Circle& circle, const Style& style) {
        return this->circle(circle.x(), circle.y(), circle.radius(), style);
    }

    Canvas& Canvas::drawImpl(const Elliptic& ellipse, const Style& style) {
        return this->ellipse(ellipse.x(), ellipse.y(),
            ellipse.width(), ellipse.height(),
            ellipse.rotation(), style);
    }

    Canvas& Canvas::drawImpl(const Sector& sector, const Style& style) {
        return this->sector(sector.x(), sector.y(), sector.radius(),
            sector.startAngle(), sector.endAngle(), style);
    }

    Canvas& Canvas::drawImpl(const Points& points, const Style& style) {
        if (points.points.size() >= 3) {
            std::vector<Point> pointVec(points.points.begin(), points.points.end());
            return polyline(pointVec, style, false);
        }
        else if (points.points.size() == 2) {
            return line(points.points[0].x, points.points[0].y,
                points.points[1].x, points.points[1].y, style);
        }
        else if (points.points.size() == 1) {
            Style pointStyle = style;
            if (style.fill_ == None && style.stroke_ != None) {
                pointStyle.fill(style.stroke_);
            }
            return circle(points.points[0].x, points.points[0].y, 1.0f, pointStyle);
        }
        return *this;
    }

    Canvas& Canvas::drawImpl(const Polygon& polygon, const Style& style) {
        return this->polygon(polygon, style);
    }

    Canvas& Canvas::text(int x, int y, const std::wstring& text, int fontSize, const Color& color, FontStyle style, const std::wstring& fontName) {
        pa2d::text(buffer_, (float)x, (float)y, text, fontSize, color, style, fontName);
        return *this;
    }

    Canvas& Canvas::textCentered(int centerX, int centerY, const std::wstring& text, int fontSize, const Color& color, FontStyle style, const std::wstring& fontName) {
        pa2d::textCentered(buffer_, (float)centerX, (float)centerY, text, fontSize, color, style, fontName);
        return *this;
    }

    Canvas& Canvas::textInRect(int rectX, int rectY, int rectWidth, int rectHeight, const std::wstring& text, int fontSize, const Color& color, FontStyle style, const std::wstring& fontName) {
        pa2d::textInRect(buffer_, (float)rectX, (float)rectY, (float)rectWidth, (float)rectHeight, text, fontSize, color, style, fontName);
        return *this;
    }

    Canvas& Canvas::textFitRect(int rectX, int rectY, int rectWidth, int rectHeight, const std::wstring& text, int preferredFontSize, const Color& color, FontStyle style, const std::wstring& fontName) {
        pa2d::textFitRect(buffer_, (float)rectX, (float)rectY, (float)rectWidth, (float)rectHeight, text, preferredFontSize, color, style, fontName);
        return *this;
    }

    Canvas& Canvas::text(int x, int y, const std::string& text, int fontSize, const Color& color, FontStyle style, const std::string& fontName) {
        pa2d::text(buffer_, (float)x, (float)y, text, fontSize, color, style, fontName);
        return *this;
    }

    Canvas& Canvas::textCentered(int centerX, int centerY, const std::string& text, int fontSize, const Color& color, FontStyle style, const std::string& fontName) {
        pa2d::textCentered(buffer_, (float)centerX, (float)centerY, text, fontSize, color, style, fontName);
        return *this;
    }

    Canvas& Canvas::textInRect(int rectX, int rectY, int rectWidth, int rectHeight, const std::string& text, int fontSize, const Color& color, FontStyle style, const std::string& fontName) {
        pa2d::textInRect(buffer_, (float)rectX, (float)rectY, (float)rectWidth, (float)rectHeight, text, fontSize, color, style, fontName);
        return *this;
    }

    Canvas& Canvas::textFitRect(int rectX, int rectY, int rectWidth, int rectHeight, const std::string& text, int preferredFontSize, const Color& color, FontStyle style, const std::string& fontName) {
        pa2d::textFitRect(buffer_, (float)rectX, (float)rectY, (float)rectWidth, (float)rectHeight, text, preferredFontSize, color, style, fontName);
        return *this;
    }
} // namespace pa2d