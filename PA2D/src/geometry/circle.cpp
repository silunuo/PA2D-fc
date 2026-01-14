#include "../include/geometry/circle.h"
#include "../include/geometry/rect.h"
#include <cmath>

namespace pa2d {
    constexpr float GEOMETRY_PI = 3.14159265358979323846f;
    constexpr float GEOMETRY_EPSILON = 1e-6f;

    Circle::Circle() : Shape(GeometryType::CIRCLE), center_({}), radius_(0) {}

    Circle::Circle(float centerX, float centerY, float radius)
        : Shape(GeometryType::CIRCLE), center_(centerX, centerY), radius_(radius) {
    }

    Circle::Circle(const Point& center, float radius)
        : Shape(GeometryType::CIRCLE), center_(center), radius_(radius) {
    }

    float Circle::x() const { return center_.x; }
    float Circle::y() const { return center_.y; }
    Point Circle::center() const { return center_; }
    Point& Circle::center() { return center_; }
    float Circle::radius() const { return radius_; }

    Circle& Circle::x(float x) {
        center_.x = x;
        return *this;
    }

    Circle& Circle::y(float y) {
        center_.y = y;
        return *this;
    }

    Circle& Circle::center(const Point& center) {
        center_ = center;
        return *this;
    }

    Circle& Circle::center(float x, float y) {
        center_ = Point(x, y);
        return *this;
    }

    Circle& Circle::radius(float radius) {
        radius_ = radius;
        return *this;
    }

    Circle& Circle::translate(float dx, float dy) {
        center_.x += dx;
        center_.y += dy;
        return *this;
    }

    Circle& Circle::translate(Point delta) {
        return translate(delta.x, delta.y);
    }

    Circle& Circle::scale(float factor) {
        radius_ *= factor;
        center_ *= factor;
        return *this;
    }

    Circle& Circle::scale(float factorX, float factorY) {
        float avgFactor = (factorX + factorY) / 2.0f;
        radius_ *= avgFactor;
        center_ *= avgFactor;
        return *this;
    }

    Circle& Circle::scaleOnSelf(float factor) {
        radius_ *= factor;
        return *this;
    }

    Circle& Circle::scaleOnSelf(float factorX, float factorY) {
        radius_ *= (factorX + factorY) / 2.0f;
        return *this;
    }

    Circle& Circle::rotate(float angleDegrees) {
        return rotate(angleDegrees, 0, 0);
    }

    Circle& Circle::rotate(float angleDegrees, float centerX, float centerY) {
        float angleRad = angleDegrees * (GEOMETRY_PI / 180.0f);
        float cosTheta = std::cos(angleRad);
        float sinTheta = std::sin(angleRad);

        float dx = center_.x - centerX;
        float dy = center_.y - centerY;

        center_.x = centerX + dx * cosTheta - dy * sinTheta;
        center_.y = centerY + dx * sinTheta + dy * cosTheta;
        return *this;
    }

    Circle& Circle::rotate(float angleDegrees, Point center) {
        return rotate(angleDegrees, center.x, center.y);
    }

    Circle& Circle::rotateOnSelf(float angleDegrees) {
        return *this;
    }

    bool Circle::contains(Point point) const {
        float dx = point.x - center_.x;
        float dy = point.y - center_.y;
        float distanceSquared = dx * dx + dy * dy;
        return distanceSquared <= (radius_ * radius_) + GEOMETRY_EPSILON;
    }

    Rect Circle::getBoundingBox() const {
        float diameter = radius_ * 2.0f;
        return Rect(center_.x, center_.y, diameter, diameter);
    }

    Point Circle::getCenter() const {
        return center_;
    }

    Circle::operator Point() const {
        return center_;
    }
}