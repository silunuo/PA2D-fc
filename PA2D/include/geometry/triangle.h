#pragma once
#include "shape.h"
#include "point.h"
#include <array>
#include <vector>
#ifndef TRIANGLE_H
#define TRIANGLE_H
namespace pa2d {
    class Triangle : public Shape {
    private:
        std::array<Point, 3> vertices_;

    public:
        Triangle();
        Triangle(const Point& p0, const Point& p1, const Point& p2);
        Triangle(float px0, float py0, float px1, float py1, float px2, float py2);
        Triangle(const std::vector<Point>& points);
        Triangle& operator=(const std::vector<Point>& points);

        // 迭代器支持
        auto begin();
        auto end();
        auto begin() const;
        auto end() const;
        Point& operator[](size_t index);
        const Point& operator[](size_t index) const;

        // 变换方法
        virtual Triangle& translate(float dx, float dy) override;
        virtual Triangle& translate(Point delta) override;
        virtual Triangle& scale(float factor) override;
        virtual Triangle& scale(float factorX, float factorY) override;
        virtual Triangle& scaleOnSelf(float factorX, float factorY) override;
        virtual Triangle& scaleOnSelf(float factor) override;
        virtual Triangle& rotate(float angleDegrees, float centerX, float centerY) override;
        virtual Triangle& rotate(float angleDegrees, Point center) override;
        virtual Triangle& rotate(float angleDegrees) override;
        virtual Triangle& rotateOnSelf(float angleDegrees) override;

        // 几何计算
        virtual bool contains(Point point) const override;
        virtual Rect getBoundingBox() const override;
        virtual Point getCenter() const override;

        // 转换运算符
        operator std::vector<Point>() const;
    };
}
#endif // !TRIANGLE_H