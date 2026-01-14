#pragma once
#include "shape.h"
#include "point.h"
#include <vector>
#ifndef POINTS_H
#define POINTS_H
namespace pa2d {
    struct Points : public Shape {
        std::vector<Point> points;

        Points();
        Points(int size);
        Points(const std::vector<Point>& points);

        Points& operator=(const std::vector<Point>& points);

        // 变换方法
        virtual Points& translate(float dx, float dy) override;
        virtual Points& translate(Point delta) override;
        virtual Points& scale(float factor) override;
        virtual Points& scale(float factorX, float factorY) override;
        virtual Points& scaleOnSelf(float factorX, float factorY) override;
        virtual Points& scaleOnSelf(float factor) override;
        virtual Points& rotate(float angleDegrees, float centerX, float centerY) override;
        virtual Points& rotate(float angleDegrees, Point center) override;
        virtual Points& rotate(float angleDegrees) override;
        virtual Points& rotateOnSelf(float angleDegrees) override;

        // 几何计算
        virtual bool contains(Point point) const override;
        virtual Rect getBoundingBox() const override;
        virtual Point getCenter() const override;

        // 转换运算符
        operator std::vector<Point>() const;
        operator const std::vector<Point>&() const;
    };
}
#endif // !POINTS_H