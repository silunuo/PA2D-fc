#include "../include/geometry/sector.h"
#include "../include/geometry/rect.h"
#include <cmath>
#include <vector>

namespace pa2d {
    constexpr float GEOMETRY_PI = 3.14159265358979323846f;
    constexpr float GEOMETRY_EPSILON = 1e-6f;

    Sector::Sector() : Shape(GeometryType::SECTOR), center_({}), radius_(0.0f), startAngle_(0.0f), endAngle_(360.0f) {}

    Sector::Sector(Point center, float radius, float startAngle, float endAngle)
        : Shape(GeometryType::SECTOR), center_(center.x, center.y),
        radius_(radius), startAngle_(startAngle), endAngle_(endAngle) {
    }

    Sector::Sector(float centerX, float centerY, float radius, float startAngle, float endAngle)
        : Shape(GeometryType::SECTOR), center_(centerX, centerY),
        radius_(radius), startAngle_(startAngle), endAngle_(endAngle) {
    }

    float Sector::x() const { return center_.x; }
    float Sector::y() const { return center_.y; }
    Point Sector::center() const { return center_; }
    Point& Sector::center() { return center_; }
    float Sector::radius() const { return radius_; }
    float Sector::startAngle() const { return startAngle_; }
    float Sector::endAngle() const { return endAngle_; }

    Sector& Sector::x(float x) {
        center_.x = x;
        return *this;
    }

    Sector& Sector::y(float y) {
        center_.y = y;
        return *this;
    }

    Sector& Sector::center(const Point& center) {
        center_ = center;
        return *this;
    }

    Sector& Sector::center(float x, float y) {
        center_ = Point(x, y);
        return *this;
    }

    Sector& Sector::radius(float radius) {
        radius_ = radius;
        return *this;
    }

    Sector& Sector::startAngle(float startAngle) {
        startAngle_ = startAngle;
        return *this;
    }

    Sector& Sector::endAngle(float endAngle) {
        endAngle_ = endAngle;
        return *this;
    }

    Sector& Sector::translate(float dx, float dy) {
        center_.x += dx;
        center_.y += dy;
        return *this;
    }

    Sector& Sector::translate(Point delta) {
        return translate(delta.x, delta.y);
    }

    Sector& Sector::scale(float factor) {
        radius_ *= factor;
        center_ *= factor;
        return *this;
    }

    Sector& Sector::scale(float factorX, float factorY) {
        float avgFactor = (factorX + factorY) / 2.0f;
        radius_ *= avgFactor;
        center_ *= avgFactor;
        return *this;
    }

    Sector& Sector::scaleOnSelf(float factor) {
        radius_ *= factor;
        return *this;
    }

    Sector& Sector::scaleOnSelf(float factorX, float factorY) {
        radius_ *= (factorX + factorY) / 2.0f;
        return *this;
    }

    Sector& Sector::rotate(float angleDegrees) {
        return rotate(angleDegrees, 0, 0);
    }

    Sector& Sector::rotate(float angleDegrees, float centerX, float centerY) {
        float angleRad = angleDegrees * (GEOMETRY_PI / 180.0f);
        float cosTheta = std::cos(angleRad);
        float sinTheta = std::sin(angleRad);
        float dx = center_.x - centerX;
        float dy = center_.y - centerY;
        center_.x = centerX + dx * cosTheta - dy * sinTheta;
        center_.y = centerY + dx * sinTheta + dy * cosTheta;
        startAngle_ = normalizeAngle(startAngle_ + angleDegrees);
        endAngle_ = normalizeAngle(endAngle_ + angleDegrees);
        return *this;
    }

    Sector& Sector::rotate(float angleDegrees, Point center) {
        return rotate(angleDegrees, center.x, center.y);
    }

    Sector& Sector::rotateOnSelf(float angleDegrees) {
        startAngle_ = normalizeAngle(startAngle_ + angleDegrees);
        endAngle_ = normalizeAngle(endAngle_ + angleDegrees);
        return *this;
    }

    bool Sector::contains(Point point) const {
        if (radius_ <= 0) return false;
        float dx = point.x - center_.x;
        float dy = point.y - center_.y;
        float distanceSquared = dx * dx + dy * dy;
        if (distanceSquared > (radius_ * radius_) + GEOMETRY_EPSILON) {
            return false;
        }
        float pointAngle = std::atan2(dy, dx) * (180.0f / GEOMETRY_PI);
        pointAngle = normalizeAngle(pointAngle);
        return isAngleInRange(pointAngle);
    }

    Rect Sector::getBoundingBox() const {
        if (radius_ <= 0) return Rect();

        float minX = center_.x, maxX = center_.x;
        float minY = center_.y, maxY = center_.y;
        std::vector<float> keyAngles = { startAngle_, endAngle_ };

        float angleRange = std::abs(endAngle_ - startAngle_);
        if (angleRange > 360.0f) angleRange = 360.0f;

        for (int i = 0; i <= 4; ++i) {
            float angle = startAngle_ + (angleRange * i) / 4;
            if (isAngleInRange(angle)) {
                keyAngles.push_back(angle);
            }
        }

        for (float angle : keyAngles) {
            float angleRad = angle * (GEOMETRY_PI / 180.0f);
            Point edgePoint(
                center_.x + radius_ * std::cos(angleRad),
                center_.y + radius_ * std::sin(angleRad)
            );
            minX = std::min(minX, edgePoint.x);
            maxX = std::max(maxX, edgePoint.x);
            minY = std::min(minY, edgePoint.y);
            maxY = std::max(maxY, edgePoint.y);
        }

        float width = maxX - minX;
        float height = maxY - minY;
        float centerX = (minX + maxX) * 0.5f;
        float centerY = (minY + maxY) * 0.5f;

        return Rect(centerX, centerY, width, height);
    }

    Point Sector::getCenter() const {
        if (startAngle_ == 0.0f && endAngle_ == 360.0f) {
            return center_;
        }
        float midAngle = (startAngle_ + endAngle_) * 0.5f;
        float angleRad = midAngle * (GEOMETRY_PI / 180.0f);
        float distance = radius_ * 0.67f;
        return Point(
            center_.x + distance * std::cos(angleRad),
            center_.y + distance * std::sin(angleRad)
        );
    }

    Sector::operator Point() const {
        return center_;
    }

    bool Sector::isAngleInRange(float angle) const {
        if (startAngle_ <= endAngle_) {
            return angle >= startAngle_ - GEOMETRY_EPSILON &&
                angle <= endAngle_ + GEOMETRY_EPSILON;
        }
        else {
            return angle >= startAngle_ - GEOMETRY_EPSILON ||
                angle <= endAngle_ + GEOMETRY_EPSILON;
        }
    }

    float Sector::normalizeAngle(float angle) const {
        angle = std::fmod(angle, 360.0f);
        if (angle < 0) {
            angle += 360.0f;
        }
        return angle;
    }
}