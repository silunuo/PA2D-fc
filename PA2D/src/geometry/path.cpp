#include "../include/geometry/path.h"
#include <utility>

namespace pa2d {
    Path::Builder::Builder(float startX, float startY) {
        points_.emplace_back(startX, startY);
    }

    Path::Builder Path::Builder::move(float dx, float dy)&& {
        auto& last = points_.back();
        points_.emplace_back(last.x + dx, last.y + dy);
        return std::move(*this);
    }

    Path::Builder Path::Builder::moveTo(float x, float y)&& {
        points_.emplace_back(x, y);
        return std::move(*this);
    }

    Path::Builder Path::Builder::translate(float dx, float dy)&& {
        for (auto& p : points_) p.translate(dx, dy);
        return std::move(*this);
    }

    Path::Builder Path::Builder::scale(float factor)&& {
        for (auto& p : points_) p.scale(factor);
        return std::move(*this);
    }

    Path::Builder Path::Builder::scale(float factorX, float factorY)&& {
        for (auto& p : points_) p.scale(factorX, factorY);
        return std::move(*this);
    }

    Path::Builder Path::Builder::scaleOnSelf(float factor)&& {
        if (points_.empty()) return std::move(*this);
        Point center(0, 0);
        for (const auto& p : points_) center += p;
        center /= points_.size();
        for (auto& p : points_) {
            p = center + (p - center) * factor;
        }
        return std::move(*this);
    }

    Path::Builder Path::Builder::scaleOnSelf(float factorX, float factorY)&& {
        if (points_.empty()) return std::move(*this);
        Point center(0, 0);
        for (const auto& p : points_) center += p;
        center /= points_.size();
        for (auto& p : points_) {
            Point relative = p - center;
            p.x = center.x + relative.x * factorX;
            p.y = center.y + relative.y * factorY;
        }
        return std::move(*this);
    }

    Path::Builder Path::Builder::rotate(float angleDegrees)&& {
        for (auto& p : points_) p.rotate(angleDegrees);
        return std::move(*this);
    }

    Path::Builder Path::Builder::rotate(float angleDegrees, float centerX, float centerY)&& {
        for (auto& p : points_) p.rotate(angleDegrees, centerX, centerY);
        return std::move(*this);
    }

    Path::Builder Path::Builder::rotateOnSelf(float angleDegrees)&& {
        if (points_.empty()) return std::move(*this);
        Point center(0, 0);
        for (const auto& p : points_) center += p;
        center /= points_.size();
        for (auto& p : points_) p.rotate(angleDegrees, center.x, center.y);
        return std::move(*this);
    }

    Path::Builder Path::from(float x, float y) {
        return Builder(x, y);
    }

    Path::Builder Path::from(const Point& start) {
        return Builder(start.x, start.y);
    }
}