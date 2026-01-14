#pragma once
#include "shape.h"
#include "point.h"
#include <vector>
#ifndef RAY_H
#define RAY_H
namespace pa2d {
    class Ray : public Shape {
    private:
        Point start_, end_;

    public:
        Ray();
        Ray(float x1, float y1, float x2, float y2);
        Ray(const Point& start, const Point& end);
        Ray(const Point& start, float length = 0.0f, float angle = 0.0f);

        // 端点操作
        Ray& start(float x, float y);
        Ray& start(const Point& start);
        Ray& end(float x, float y);
        Ray& end(const Point& end);
        Point start() const;
        Point end() const;

        // 射线特有操作
        Ray& angle(float angle);
        float angle() const;
        Ray& length(float length);
        float length() const;
        Ray& toEnd();
        Ray& stretch(float factor);
        Ray& spin(float angle);

        // 变换方法
        virtual Ray& translate(float dx, float dy) override;
        virtual Ray& translate(Point delta) override;
        virtual Ray& scale(float factor) override;
        virtual Ray& scale(float factorX, float factorY) override;
        virtual Ray& scaleOnSelf(float factorX, float factorY) override;
        virtual Ray& scaleOnSelf(float factor) override;
        virtual Ray& rotate(float angleDegrees, float centerX, float centerY) override;
        virtual Ray& rotate(float angleDegrees, Point center) override;
        virtual Ray& rotate(float angleDegrees) override;
        virtual Ray& rotateOnSelf(float angleDegrees) override;

        // 几何计算
        virtual bool contains(Point point) const override;
        virtual Rect getBoundingBox() const override;
        virtual Point getCenter() const override;

        // 转换运算符
        operator std::vector<Point>() const;
    };
}
#endif // !RAY_H