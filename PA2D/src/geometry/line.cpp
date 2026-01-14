#include "../include/geometry/line.h"
#include "../include/geometry/rect.h"
#include <algorithm>
#include <cmath>

namespace pa2d {
    Line::Line() : Shape(GeometryType::LINE), start_(), end_() {}

    Line::Line(float x0, float y0, float x1, float y1)
        : Shape(GeometryType::LINE), start_(x0, y0), end_(x1, y1) {
    }

    Line::Line(const Point& start, const Point& end)
        : Shape(GeometryType::LINE), start_(start), end_(end) {
    }

    Line& Line::start(float x, float y) {
        start_ = { x, y };
        return *this;
    }

    Line& Line::end(float x, float y) {
        end_ = { x, y };
        return *this;
    }

    Line& Line::start(const Point& start) {
        start_ = start;
        return *this;
    }

    Line& Line::end(const Point& end) {
        end_ = end;
        return *this;
    }

    Point& Line::start() {
        return start_;
    }

    Point& Line::end() {
        return end_;
    }

    Point Line::start() const {
        return start_;
    }

    Point Line::end() const {
        return end_;
    }

    Line& Line::translate(float dx, float dy) {
        start_.translate(dx, dy);
        end_.translate(dx, dy);
        return *this;
    }

    Line& Line::translate(Point delta) {
        return translate(delta.x, delta.y);
    }

    Line& Line::scale(float factor) {
        start_.scale(factor);
        end_.scale(factor);
        return *this;
    }

    Line& Line::scale(float factorX, float factorY) {
        start_.scale(factorX, factorY);
        end_.scale(factorX, factorY);
        return *this;
    }

    Line& Line::scaleOnSelf(float factor) {
        Point center = (start_ + end_) / 2;
        start_ -= center;
        end_ -= center;
        start_ *= factor;
        end_ *= factor;
        start_ += center;
        end_ += center;
        return *this;
    }

    Line& Line::scaleOnSelf(float factorX, float factorY) {
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

    Line& Line::rotate(float angleDegrees) {
        start_.rotate(angleDegrees);
        end_.rotate(angleDegrees);
        return *this;
    }

    Line& Line::rotate(float angleDegrees, float centerX, float centerY) {
        start_.rotate(angleDegrees, centerX, centerY);
        end_.rotate(angleDegrees, centerX, centerY);
        return *this;
    }

    Line& Line::rotate(float angleDegrees, Point center) {
        return rotate(angleDegrees, center.x, center.y);
    }

    Line& Line::rotateOnSelf(float angleDegrees) {
        return rotate(angleDegrees, (start_ + end_) / 2);
    }

    bool Line::contains(Point point) const {
        return point == start_ || point == end_ ? true : false;
    }

    Rect Line::getBoundingBox() const {
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

    Point Line::getCenter() const {
        return (start_ + end_) / 2;
    }

    Line::operator std::vector<Point>() const {
        return { start_, end_ };
    }
}