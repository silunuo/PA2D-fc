#pragma once
#include "shape.h"
#include "point.h"
#include "points.h"
#include <vector>
#ifndef POLYGON_H
#define POLYGON_H
namespace pa2d {
    class Polygon : public Shape {
    private:
        Points points_;
        static bool is_point_in_polygon_odd_even(const Point& P, const Points& vertices);

    public:
        Polygon();
        Polygon(const std::vector<Point>& points);

        Polygon& operator=(const std::vector<Point>& vec);

        std::vector<Point>& getPoints();
        const std::vector<Point>& getPoints() const;

        // 变换方法
        virtual Polygon& translate(float dx, float dy) override;
        virtual Polygon& translate(Point delta) override;
        virtual Polygon& scale(float factor) override;
        virtual Polygon& scale(float factorX, float factorY) override;
        virtual Polygon& scaleOnSelf(float factorX, float factorY) override;
        virtual Polygon& scaleOnSelf(float factor) override;
        virtual Polygon& rotate(float angleDegrees, float centerX, float centerY) override;
        virtual Polygon& rotate(float angleDegrees, Point center) override;
        virtual Polygon& rotate(float angleDegrees) override;
        virtual Polygon& rotateOnSelf(float angleDegrees) override;

        // 几何计算
        virtual bool contains(Point point) const override;
        virtual Rect getBoundingBox() const override;
        virtual Point getCenter() const override;

        // 转换运算符
        operator std::vector<Point>() const;
        operator const std::vector<Point>&() const;
    };
}
#endif // !POLYGON_H