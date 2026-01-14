#pragma once
#include "point.h"
#include <vector>
#ifndef PATH_H
#define PATH_H
namespace pa2d {
    class Path {
        class Builder {
            std::vector<Point> points_;

        public:
            Builder(float startX, float startY);
            Builder(const Builder&) = delete;
            Builder& operator=(const Builder&) = delete;
            Builder(Builder&&) = default;
            Builder& operator=(Builder&&) = default;

            // 构建方法
            Builder move(float dx, float dy) &&;
            Builder moveTo(float x, float y) &&;
            Builder translate(float dx, float dy) &&;
            Builder scale(float factor) &&;
            Builder scale(float factorX, float factorY) &&;
            Builder scaleOnSelf(float factor) &&;
            Builder scaleOnSelf(float factorX, float factorY) &&;
            Builder rotate(float angleDegrees) &&;
            Builder rotate(float angleDegrees, float centerX, float centerY) &&;
            Builder rotateOnSelf(float angleDegrees) &&;

            // 转换为具体形状
            template<typename T>
            T to() && {
                return T(std::move(points_));
            }

            template<typename T>
            operator T() && {
                return T(std::move(points_));
            }

            template<typename T>
            operator T() & {
                return T(points_);
            }

            template<typename T>
            operator T() const& = delete;
        };

        Path() = delete;

    public:
        static Builder from(float x, float y);
        static Builder from(const Point& start);
    };
}
#endif // !PATH_H