#include "../include/geometry/ellipse.h"
#include "../include/geometry/rect.h"
#include <cmath>

namespace pa2d {
    constexpr float GEOMETRY_PI = 3.14159265358979323846f;
    constexpr float GEOMETRY_EPSILON = 1e-6f;

    Elliptic::Elliptic() : Shape(GeometryType::ELLIPTIC), center_({}), width_(0), height_(0), rotation_(0) {}

    Elliptic::Elliptic(float centerX, float centerY, float width, float height, float rotation)
        : Shape(GeometryType::ELLIPTIC), center_(centerX, centerY), width_(width), height_(height), rotation_(rotation) {
    }

    float Elliptic::x() const { return center_.x; }
    float Elliptic::y() const { return center_.y; }
    Point Elliptic::center() const { return center_; }
    Point& Elliptic::center() { return center_; }
    float Elliptic::width() const { return width_; }
    float Elliptic::height() const { return height_; }
    float Elliptic::rotation() const { return rotation_; }

    Elliptic& Elliptic::x(float x) {
        center_.x = x;
        return *this;
    }

    Elliptic& Elliptic::y(float y) {
        center_.y = y;
        return *this;
    }

    Elliptic& Elliptic::center(const Point& center) {
        center_ = center;
        return *this;
    }

    Elliptic& Elliptic::center(float x, float y) {
        center_ = Point(x, y);
        return *this;
    }

    Elliptic& Elliptic::width(float width) {
        width_ = width;
        return *this;
    }

    Elliptic& Elliptic::height(float height) {
        height_ = height;
        return *this;
    }

    Elliptic& Elliptic::rotation(float rotation) {
        rotation_ = rotation;
        return *this;
    }

    Elliptic& Elliptic::translate(float dx, float dy) {
        center_.x += dx;
        center_.y += dy;
        return *this;
    }

    Elliptic& Elliptic::translate(Point delta) {
        return translate(delta.x, delta.y);
    }

    Elliptic& Elliptic::scale(float factor) {
        width_ *= factor;
        height_ *= factor;
        center_ *= factor;
        return *this;
    }

    Elliptic& Elliptic::scale(float factorX, float factorY) {
        width_ *= factorX;
        height_ *= factorY;
        center_.x *= factorX;
        center_.y *= factorY;
        return *this;
    }

    Elliptic& Elliptic::scaleOnSelf(float factor) {
        width_ *= factor;
        height_ *= factor;
        return *this;
    }

    Elliptic& Elliptic::scaleOnSelf(float factorX, float factorY) {
        width_ *= factorX;
        height_ *= factorY;
        return *this;
    }

    Elliptic& Elliptic::rotate(float angleDegrees) {
        return rotate(angleDegrees, 0, 0);
    }

    Elliptic& Elliptic::rotate(float angleDegrees, float centerX, float centerY) {
        float angleRad = angleDegrees * (GEOMETRY_PI / 180.0f);
        float cosTheta = std::cos(angleRad);
        float sinTheta = std::sin(angleRad);

        float dx = center_.x - centerX;
        float dy = center_.y - centerY;

        center_.x = centerX + dx * cosTheta - dy * sinTheta;
        center_.y = centerY + dx * sinTheta + dy * cosTheta;

        rotation_ += angleDegrees;
        return *this;
    }

    Elliptic& Elliptic::rotate(float angleDegrees, Point center) {
        return rotate(angleDegrees, center.x, center.y);
    }

    Elliptic& Elliptic::rotateOnSelf(float angleDegrees) {
        rotation_ += angleDegrees;
        return *this;
    }

    bool Elliptic::contains(Point point) const {
        if (width_ == 0 || height_ == 0) return false;
        float angleRad = -rotation_ * (GEOMETRY_PI / 180.0f);
        float cosTheta = std::cos(angleRad);
        float sinTheta = std::sin(angleRad);
        float dx = point.x - center_.x;
        float dy = point.y - center_.y;
        float localX = dx * cosTheta - dy * sinTheta;
        float localY = dx * sinTheta + dy * cosTheta;
        float a = width_ / 2.0f;
        float b = height_ / 2.0f;
        float value = (localX * localX) / (a * a) + (localY * localY) / (b * b);
        return value <= 1.0f + GEOMETRY_EPSILON;
    }

    Rect Elliptic::getBoundingBox() const {
        if (width_ == 0 || height_ == 0) return Rect();

        float a = width_ / 2.0f;
        float b = height_ / 2.0f;
        float angleRad = rotation_ * (GEOMETRY_PI / 180.0f);
        float cosTheta = std::cos(angleRad);
        float sinTheta = std::sin(angleRad);

        float halfWidth = std::sqrt((a * cosTheta) * (a * cosTheta) + (b * sinTheta) * (b * sinTheta));
        float halfHeight = std::sqrt((a * sinTheta) * (a * sinTheta) + (b * cosTheta) * (b * cosTheta));

        return Rect(center_.x, center_.y, halfWidth * 2.0f, halfHeight * 2.0f);
    }

    Point Elliptic::getCenter() const {
        return center_;
    }

    Elliptic::operator Point() const {
        return center_;
    }
}