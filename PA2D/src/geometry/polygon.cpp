#include "../include/geometry/polygon.h"
#include "../include/geometry/rect.h"
#include <algorithm>
#include <cmath>

namespace pa2d {
    Polygon::Polygon() : Shape(GeometryType::POLYGON) {}

    Polygon::Polygon(const std::vector<Point>& points)
        : Shape(GeometryType::POLYGON), points_(points) {
    }

    Polygon& Polygon::operator=(const std::vector<Point>& points) {
        points_ = points;
        return *this;
    }

    std::vector<Point>& Polygon::getPoints() {
        return points_.points;
    }

    const std::vector<Point>& Polygon::getPoints() const {
        return points_.points;
    }

    Polygon& Polygon::translate(float dx, float dy) {
        points_.translate(dx, dy);
        return *this;
    }

    Polygon& Polygon::translate(Point delta) {
        return translate(delta.x, delta.y);
    }

    Polygon& Polygon::scale(float factor) {
        points_.scale(factor);
        return *this;
    }

    Polygon& Polygon::scale(float factorX, float factorY) {
        points_.scale(factorX, factorY);
        return *this;
    }

    Polygon& Polygon::scaleOnSelf(float factor) {
        points_.scaleOnSelf(factor);
        return *this;
    }

    Polygon& Polygon::scaleOnSelf(float factorX, float factorY) {
        points_.scaleOnSelf(factorX, factorY);
        return *this;
    }

    Polygon& Polygon::rotate(float angleDegrees) {
        points_.rotate(angleDegrees);
        return *this;
    }

    Polygon& Polygon::rotate(float angleDegrees, float centerX, float centerY) {
        points_.rotate(angleDegrees, centerX, centerY);
        return *this;
    }

    Polygon& Polygon::rotate(float angleDegrees, Point center) {
        return rotate(angleDegrees, center.x, center.y);
    }

    Polygon& Polygon::rotateOnSelf(float angleDegrees) {
        points_.rotateOnSelf(angleDegrees);
        return *this;
    }

    bool Polygon::contains(Point point) const {
        if (points_.points.size() < 3) return points_.contains(point);
        return is_point_in_polygon_odd_even(point, points_);
    }

    Rect Polygon::getBoundingBox() const {
        return points_.getBoundingBox();
    }

    Point Polygon::getCenter() const {
        return points_.getCenter();
    }

    Polygon::operator std::vector<Point>() const {
        return points_.points;
    }

    Polygon::operator const std::vector<Point>& () const {
        return points_.points;
    }

    bool Polygon::is_point_in_polygon_odd_even(const Point& P, const Points& vertices) {
        int crossings = 0;
        size_t n = vertices.points.size();

        for (size_t i = 0; i < n; ++i) {
            const Point& v1 = vertices.points[i];
            const Point& v2 = vertices.points[(i + 1) % n];

            if ((v1.y <= P.y && v2.y > P.y) || (v1.y > P.y && v2.y <= P.y)) {
                float t = (P.y - v1.y) / (v2.y - v1.y);
                float intersectX = v1.x + t * (v2.x - v1.x);
                if (intersectX > P.x) {
                    crossings++;
                }
            }
        }
        return (crossings % 2) == 1;
    }
}