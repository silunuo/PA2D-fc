#pragma once
#include "shape.h"
#include "point.h"
#include <vector>
#ifndef LINE_H
#define LINE_H
namespace pa2d {
    class Line : public Shape {
    private:
        Point start_, end_;
    public:
        Line();
        Line(float x0, float y0, float x1, float y1);
        Line(const Point& start, const Point& end);

        // 端点操作
        Line& start(float x, float y);
        Line& start(const Point& start);
        Line& end(float x, float y);
        Line& end(const Point& end);
        Point& start();
        Point& end();
        Point start() const;
        Point end() const;

        // 变换方法
        virtual Line& translate(float dx, float dy) override;
        virtual Line& translate(Point delta) override;
        virtual Line& scale(float factor) override;
        virtual Line& scale(float factorX, float factorY) override;
        virtual Line& scaleOnSelf(float factorX, float factorY) override;
        virtual Line& scaleOnSelf(float factor) override;
        virtual Line& rotate(float angleDegrees, float centerX, float centerY) override;
        virtual Line& rotate(float angleDegrees, Point center) override;
        virtual Line& rotate(float angleDegrees) override;
        virtual Line& rotateOnSelf(float angleDegrees) override;

        // 几何计算
        virtual bool contains(Point point) const override;
        virtual Rect getBoundingBox() const override;
        virtual Point getCenter() const override;

        // 转换运算符
        operator std::vector<Point>() const;
    };
}
#endif // !LINE_H