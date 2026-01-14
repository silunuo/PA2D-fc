#include "../include/geometry/triangle.h"
#include "../include/geometry/rect.h"
#include <algorithm>
#include <cmath>

namespace pa2d {
    constexpr float GEOMETRY_EPSILON = 1e-6f;

    Triangle::Triangle() : Shape(GeometryType::TRIANGLE), vertices_() {}

    Triangle::Triangle(const Point& p0, const Point& p1, const Point& p2)
        : Shape(GeometryType::TRIANGLE), vertices_({ p0, p1, p2 }) {
    }

    Triangle::Triangle(float px0, float py0, float px1, float py1, float px2, float py2)
        : Shape(GeometryType::TRIANGLE), vertices_({ Point{px0,py0}, Point{px1,py1}, Point{px2,py2} }) {
    }

    Triangle::Triangle(const std::vector<Point>& points)
        : Shape(GeometryType::TRIANGLE) {
        if (points.size() >= 1) vertices_[0] = points[0];
        if (points.size() >= 2) vertices_[1] = points[1];
        if (points.size() >= 3) vertices_[2] = points[2];
    }

    Triangle& Triangle::operator=(const std::vector<Point>& points) {
        if (points.size() >= 1) vertices_[0] = points[0];
        if (points.size() >= 2) vertices_[1] = points[1];
        if (points.size() >= 3) vertices_[2] = points[2];
        return *this;
    }

    auto Triangle::begin() { return vertices_.begin(); }
    auto Triangle::end() { return vertices_.end(); }
    auto Triangle::begin() const { return vertices_.begin(); }
    auto Triangle::end() const { return vertices_.end(); }

    Point& Triangle::operator[](size_t index) { return vertices_[index]; }
    const Point& Triangle::operator[](size_t index) const { return vertices_[index]; }

    Triangle& Triangle::translate(float dx, float dy) {
        for (auto& p : vertices_) p.translate(dx, dy);
        return *this;
    }

    Triangle& Triangle::translate(Point delta) {
        return translate(delta.x, delta.y);
    }

    Triangle& Triangle::scale(float factor) {
        for (auto& p : vertices_) p.scale(factor);
        return *this;
    }

    Triangle& Triangle::scale(float factorX, float factorY) {
        for (auto& p : vertices_) p.scale(factorX, factorY);
        return *this;
    }

    Triangle& Triangle::scaleOnSelf(float factor) {
        Point center = (vertices_[0] + vertices_[1] + vertices_[2]) / 3.0f;
        for (auto& p : vertices_) {
            p -= center;
            p *= factor;
            p += center;
        }
        return *this;
    }

    Triangle& Triangle::scaleOnSelf(float factorX, float factorY) {
        Point center = (vertices_[0] + vertices_[1] + vertices_[2]) / 3.0f;
        for (auto& p : vertices_) {
            p -= center;
            p.x *= factorX;
            p.y *= factorY;
            p += center;
        }
        return *this;
    }

    Triangle& Triangle::rotate(float angleDegrees) {
        for (auto& p : vertices_) p.rotate(angleDegrees);
        return *this;
    }

    Triangle& Triangle::rotate(float angleDegrees, float centerX, float centerY) {
        for (auto& p : vertices_) p.rotate(angleDegrees, centerX, centerY);
        return *this;
    }

    Triangle& Triangle::rotate(float angleDegrees, Point center) {
        return rotate(angleDegrees, center.x, center.y);
    }

    Triangle& Triangle::rotateOnSelf(float angleDegrees) {
        Point center = (vertices_[0] + vertices_[1] + vertices_[2]) / 3.0f;
        rotate(angleDegrees, center.x, center.y);
        return *this;
    }

    bool Triangle::contains(Point point) const {
        const Point& a = vertices_[0];
        const Point& b = vertices_[1];
        const Point& c = vertices_[2];
        float area = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
        if (std::abs(area) < GEOMETRY_EPSILON) return false;
        float invArea = 1.0f / area;
        float alpha = ((b.y - c.y) * (point.x - c.x) + (c.x - b.x) * (point.y - c.y)) * invArea;
        float beta = ((c.y - a.y) * (point.x - c.x) + (a.x - c.x) * (point.y - c.y)) * invArea;
        float gamma = 1.0f - alpha - beta;
        const float eps = GEOMETRY_EPSILON;
        return (alpha >= -eps) && (beta >= -eps) && (gamma >= -eps);
    }

    Rect Triangle::getBoundingBox() const {
        float minX = std::min({ vertices_[0].x, vertices_[1].x, vertices_[2].x });
        float maxX = std::max({ vertices_[0].x, vertices_[1].x, vertices_[2].x });
        float minY = std::min({ vertices_[0].y, vertices_[1].y, vertices_[2].y });
        float maxY = std::max({ vertices_[0].y, vertices_[1].y, vertices_[2].y });

        return Rect(
            (minX + maxX) * 0.5f,
            (minY + maxY) * 0.5f,
            maxX - minX,
            maxY - minY
        );
    }

    Point Triangle::getCenter() const {
        return (vertices_[0] + vertices_[1] + vertices_[2]) / 3.0f;
    }

    Triangle::operator std::vector<Point>() const {
        return { vertices_[0], vertices_[1], vertices_[2] };
    }
}