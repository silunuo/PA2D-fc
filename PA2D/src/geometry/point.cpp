#include "../include/geometry/point.h"
#include <cmath>

namespace pa2d {
    constexpr float GEOMETRY_PI = 3.14159265358979323846f;

    Point::Point(float x, float y) : x(x), y(y) {}

    Point Point::operator+(const Point& other) const {
        return Point(x + other.x, y + other.y);
    }

    Point Point::operator-(const Point& other) const {
        return Point(x - other.x, y - other.y);
    }

    Point& Point::operator+=(const Point& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    Point& Point::operator-=(const Point& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    Point Point::operator-() const {
        return Point(-x, -y);
    }

    bool Point::operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }

    bool Point::operator!=(const Point& other) const {
        return x != other.x || y != other.y;
    }

    Point& Point::operator+=(float scalar) {
        x += scalar;
        y += scalar;
        return *this;
    }

    Point& Point::operator-=(float scalar) {
        x -= scalar;
        y -= scalar;
        return *this;
    }

    Point& Point::operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        return *this;
    }

    Point& Point::operator/=(float scalar) {
        x /= scalar;
        y /= scalar;
        return *this;
    }

    Point Point::operator+(float scalar) const {
        return Point(x + scalar, y + scalar);
    }

    Point Point::operator-(float scalar) const {
        return Point(x - scalar, y - scalar);
    }

    Point Point::operator/(float scalar) const {
        return Point(x / scalar, y / scalar);
    }

    Point Point::operator*(float scalar) const {
        return Point(x * scalar, y * scalar);
    }

    Point operator*(float scalar, const Point& point) {
        return Point(scalar * point.x, scalar * point.y);
    }

    Point& Point::translate(float dx, float dy) {
        x += dx;
        y += dy;
        return *this;
    }

    Point& Point::scale(float factor) {
        x *= factor;
        y *= factor;
        return *this;
    }

    Point& Point::scale(float factorX, float factorY) {
        x *= factorX;
        y *= factorY;
        return *this;
    }

    Point& Point::rotate(float angleDegrees, float centerX, float centerY) {
        float angleRad = angleDegrees * (GEOMETRY_PI / 180.0f);
        float cosTheta = std::cos(angleRad);
        float sinTheta = std::sin(angleRad);
        float dx = x - centerX;
        float dy = y - centerY;
        x = centerX + dx * cosTheta - dy * sinTheta;
        y = centerY + dx * sinTheta + dy * cosTheta;
        return *this;
    }
}