#include "../include/geometry/ray.h"
#include "../include/geometry/rect.h"
#include <cmath>

namespace pa2d {
    constexpr float GEOMETRY_PI = 3.14159265358979323846f;
    constexpr float GEOMETRY_EPSILON = 1e-6f;

    Ray::Ray() : Shape(GeometryType::RAY), start_(), end_() {}

    Ray::Ray(const Point& start, float length, float angle)
        : Shape(GeometryType::RAY), start_(start) {
        float rad = angle * (GEOMETRY_PI / 180.0f);
        end_.x = start.x + length * std::cos(rad);
        end_.y = start.y - length * std::sin(rad);
    }

    Ray::Ray(const Point& start, const Point& end)
        : Shape(GeometryType::RAY), start_(start), end_(end) {
    }

    Ray::Ray(float x0, float y0, float x1, float y1)
        : Shape(GeometryType::RAY), start_(x0, y0), end_(x1, y1) {
    }

    Ray& Ray::start(const Point& start) {
        start_ = start;
        return *this;
    }

    Ray& Ray::start(float x, float y) {
        return start({ x, y });
    }

    Ray& Ray::end(const Point& end) {
        end_ = end;
        return *this;
    }

    Ray& Ray::end(float x, float y) {
        return end({ x, y });
    }

    Point Ray::start() const {
        return start_;
    }

    Point Ray::end() const {
        return end_;
    }

    float Ray::length() const {
        float dx = end_.x - start_.x;
        float dy = end_.y - start_.y;
        return std::sqrt(dx * dx + dy * dy);
    }

    float Ray::angle() const {
        float dx = end_.x - start_.x;
        float dy = end_.y - start_.y;
        return std::atan2(-dy, dx) * (180.0f / GEOMETRY_PI);
    }

    Ray& Ray::angle(float newAngle) {
        float currentLength = length();
        float rad = newAngle * (GEOMETRY_PI / 180.0f);
        end_.x = start_.x + currentLength * std::cos(rad);
        end_.y = start_.y - currentLength * std::sin(rad);
        return *this;
    }

    Ray& Ray::length(float newLength) {
        float currentAngle = angle();
        float rad = currentAngle * (GEOMETRY_PI / 180.0f);
        end_.x = start_.x + newLength * std::cos(rad);
        end_.y = start_.y - newLength * std::sin(rad);
        return *this;
    }

    Ray& Ray::toEnd() {
        start_ = end_;
        return *this;
    }

    Ray& Ray::translate(float dx, float dy) {
        start_.translate(dx, dy);
        end_.translate(dx, dy);
        return *this;
    }

    Ray& Ray::translate(Point delta) {
        return translate(delta.x, delta.y);
    }

    Ray& Ray::scale(float factor) {
        start_.scale(factor);
        end_.scale(factor);
        return *this;
    }

    Ray& Ray::scale(float factorX, float factorY) {
        start_.scale(factorX, factorY);
        end_.scale(factorX, factorY);
        return *this;
    }

    Ray& Ray::scaleOnSelf(float factor) {
        Point center = (start_ + end_) / 2;
        start_ -= center;
        end_ -= center;
        start_ *= factor;
        end_ *= factor;
        start_ += center;
        end_ += center;
        return *this;
    }

    Ray& Ray::scaleOnSelf(float factorX, float factorY) {
        Point center = (start_ + end_) / 2;
        start_ -= center;
        end_ -= center;
        start_.x *= factorX;
        end_.x *= factorX;
        start_.y *= factorY;
        end_.y *= factorY;
        start_ += center;
        end_ += center;
        return *this;
    }

    Ray& Ray::rotate(float angleDegrees) {
        start_.rotate(angleDegrees);
        end_.rotate(angleDegrees);
        return *this;
    }

    Ray& Ray::rotate(float angleDegrees, float centerX, float centerY) {
        start_.rotate(angleDegrees, centerX, centerY);
        end_.rotate(angleDegrees, centerX, centerY);
        return *this;
    }

    Ray& Ray::rotate(float angleDegrees, Point center) {
        return rotate(angleDegrees, center.x, center.y);
    }

    Ray& Ray::rotateOnSelf(float angleDegrees) {
        return rotate(angleDegrees, (start_ + end_) / 2);
    }

    bool Ray::contains(Point point) const {
        float cross = (point.x - start_.x) * (end_.y - start_.y)
            - (point.y - start_.y) * (end_.x - start_.x);
        if (std::abs(cross) > 1e-6f) return false;

        float dot = (point.x - start_.x) * (end_.x - start_.x)
            + (point.y - start_.y) * (end_.y - start_.y);
        float len2 = (end_.x - start_.x) * (end_.x - start_.x)
            + (end_.y - start_.y) * (end_.y - start_.y);

        return dot >= 0 && dot <= len2;
    }

    Rect Ray::getBoundingBox() const {
        float minX = std::min(start_.x, end_.x);
        float maxX = std::max(start_.x, end_.x);
        float minY = std::min(start_.y, end_.y);
        float maxY = std::max(start_.y, end_.y);

        return Rect(
            (minX + maxX) / 2.0f,
            (minY + maxY) / 2.0f,
            maxX - minX,
            maxY - minY
        );
    }

    Point Ray::getCenter() const {
        return (start_ + end_) / 2;
    }

    Ray& Ray::stretch(float factor) {
        Point dir = end_ - start_;
        end_ = start_ + dir * factor;
        return *this;
    }

    Ray& Ray::spin(float degrees) {
        end_.rotate(degrees, start_.x, start_.y);
        return *this;
    }

    Ray::operator std::vector<Point>() const {
        return { start_, end_ };
    }
}