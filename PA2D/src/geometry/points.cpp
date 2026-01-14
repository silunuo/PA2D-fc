#include "../include/geometry/points.h"
#include "../include/geometry/rect.h"
#include <algorithm>
#include <cmath>

namespace pa2d {
    Points::Points() : Shape(GeometryType::POINTS) {}

    Points::Points(int size) : Shape(GeometryType::POINTS), points(size) {}

    Points::Points(const std::vector<Point>& points)
        : Shape(GeometryType::POINTS), points(points) {
    }

    Points& Points::operator=(const std::vector<Point>& other) {
        points = other;
        return *this;
    }

    Points::operator const std::vector<Point>& () const {
        return points;
    }

    Points::operator std::vector<Point>() const {
        return points;
    }

    Points& Points::translate(float dx, float dy) {
        for (auto& p : points) p.translate(dx, dy);
        return *this;
    }

    Points& Points::translate(Point delta) {
        return translate(delta.x, delta.y);
    }

    Points& Points::scale(float factorX, float factorY) {
        for (auto& p : points) p.scale(factorX, factorY);
        return *this;
    }

    Points& Points::scale(float factor) {
        for (auto& p : points) p.scale(factor);
        return *this;
    }

    Points& Points::scaleOnSelf(float factorX, float factorY) {
        Point center = getCenter();
        for (auto& point : points) {
            point -= center;
            point.x *= factorX;
            point.y *= factorY;
            point += center;
        }
        return *this;
    }

    Points& Points::scaleOnSelf(float factor) {
        Point center = getCenter();
        for (auto& point : points) {
            point -= center;
            point *= factor;
            point += center;
        }
        return *this;
    }

    Points& Points::rotate(float angleDegrees, float centerX, float centerY) {
        for (auto& p : points) p.rotate(angleDegrees, centerX, centerY);
        return *this;
    }

    Points& Points::rotate(float angleDegrees) {
        for (auto& p : points) p.rotate(angleDegrees);
        return *this;
    }

    Points& Points::rotate(float angleDegrees, Point center) {
        return rotate(angleDegrees, center.x, center.y);
    }

    Points& Points::rotateOnSelf(float angleDegrees) {
        if (points.empty()) return *this;
        Point center(0, 0);
        for (const auto& p : points) center += p;
        center /= points.size();
        return rotate(angleDegrees, center.x, center.y);
    }

    bool Points::contains(Point point) const {
        return std::find(points.begin(), points.end(), point) != points.end();
    }

    Rect Points::getBoundingBox() const {
        if (points.empty()) return Rect();
        Point min = points[0], max = points[0];
        for (const auto& p : points) {
            min.x = std::min(min.x, p.x);
            min.y = std::min(min.y, p.y);
            max.x = std::max(max.x, p.x);
            max.y = std::max(max.y, p.y);
        }
        return Rect(
            (min.x + max.x) / 2.0f,
            (min.y + max.y) / 2.0f,
            max.x - min.x,
            max.y - min.y
        );
    }

    Point Points::getCenter() const {
        if (points.empty()) return Point();
        Point center(0, 0);
        for (const auto& p : points) center += p;
        return center / points.size();
    }
}