#pragma once
#include "shape.h"
#include "point.h"
#ifndef CIRCLE_H
#define CIRCLE_H
namespace pa2d {
    class Circle : public Shape {
    private:
        Point center_;
        float radius_;

    public:
        Circle();
        Circle(float centerX, float centerY, float radius);
        Circle(const Point& center, float radius);

        // 属性设置
        Circle& x(float x);
        Circle& y(float y);
        Circle& center(const Point& center);
        Circle& center(float x, float y);
        Circle& radius(float radius);

        // 属性获取
        float x() const;
        float y() const;
        Point center() const;
        Point& center();
        float radius() const;

        // 变换方法
        virtual Circle& translate(float dx, float dy) override;
        virtual Circle& translate(Point delta) override;
        virtual Circle& scale(float factor) override;
        virtual Circle& scale(float factorX, float factorY) override;
        virtual Circle& scaleOnSelf(float factorX, float factorY) override;
        virtual Circle& scaleOnSelf(float factor) override;
        virtual Circle& rotate(float angleDegrees, float centerX, float centerY) override;
        virtual Circle& rotate(float angleDegrees, Point center) override;
        virtual Circle& rotate(float angleDegrees) override;
        virtual Circle& rotateOnSelf(float angleDegrees) override;

        // 几何计算
        virtual bool contains(Point point) const override;
        virtual Rect getBoundingBox() const override;
        virtual Point getCenter() const override;

        // 转换运算符
        operator Point() const;
    };
}
#endif // !CIRCLE_H