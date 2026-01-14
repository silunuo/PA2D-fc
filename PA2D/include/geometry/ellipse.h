#pragma once
#include "point.h"
#include "shape.h"
#ifndef ELLIPTIC_H
#define ELLIPTIC_H
namespace pa2d {
    class Elliptic : public Shape {
    private:
        Point center_;
        float width_;
        float height_;
        float rotation_;

    public:
        Elliptic();
        Elliptic(float centerX, float centerY, float width, float height, float rotation = 0.0f);

        // 属性设置
        Elliptic& x(float x);
        Elliptic& y(float y);
        Elliptic& center(const Point& center);
        Elliptic& center(float x, float y);
        Elliptic& width(float width);
        Elliptic& height(float height);
        Elliptic& rotation(float rotation);

        // 属性获取
        float x() const;
        float y() const;
        Point center() const;
        Point& center();
        float width() const;
        float height() const;
        float rotation() const;

        // 变换方法
        virtual Elliptic& translate(float dx, float dy) override;
        virtual Elliptic& translate(Point delta) override;
        virtual Elliptic& scale(float factor) override;
        virtual Elliptic& scale(float factorX, float factorY) override;
        virtual Elliptic& scaleOnSelf(float factorX, float factorY) override;
        virtual Elliptic& scaleOnSelf(float factor) override;
        virtual Elliptic& rotate(float angleDegrees, float centerX, float centerY) override;
        virtual Elliptic& rotate(float angleDegrees, Point center) override;
        virtual Elliptic& rotate(float angleDegrees) override;
        virtual Elliptic& rotateOnSelf(float angleDegrees) override;

        // 几何计算
        virtual bool contains(Point point) const override;
        virtual Rect getBoundingBox() const override;
        virtual Point getCenter() const override;

        // 转换运算符
        operator Point() const;
    };
}
#endif // !ELLIPTIC_H