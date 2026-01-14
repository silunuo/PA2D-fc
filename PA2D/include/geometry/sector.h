#pragma once
#include "shape.h"
#include "point.h"
#ifndef SECTOR_H
#define SECTOR_H
namespace pa2d {
    class Sector : public Shape {
    private:
        Point center_;
        float radius_;
        float startAngle_;
        float endAngle_;

        bool isAngleInRange(float angle) const;
        float normalizeAngle(float angle) const;

    public:
        Sector();
        Sector(Point center, float radius, float startAngle = 0.0f, float endAngle = 360.0f);
        Sector(float centerX, float centerY, float radius, float startAngle = 0.0f, float endAngle = 360.0f);

        // 属性设置
        Sector& x(float x);
        Sector& y(float y);
        Sector& center(const Point& center);
        Sector& center(float x, float y);
        Sector& radius(float radius);
        Sector& startAngle(float startAngle);
        Sector& endAngle(float endAngle);

        // 属性获取
        float x() const;
        float y() const;
        Point center() const;
        Point& center();
        float radius() const;
        float startAngle() const;
        float endAngle() const;

        // 变换方法
        virtual Sector& translate(float dx, float dy) override;
        virtual Sector& translate(Point delta) override;
        virtual Sector& scale(float factor) override;
        virtual Sector& scale(float factorX, float factorY) override;
        virtual Sector& scaleOnSelf(float factorX, float factorY) override;
        virtual Sector& scaleOnSelf(float factor) override;
        virtual Sector& rotate(float angleDegrees, float centerX, float centerY) override;
        virtual Sector& rotate(float angleDegrees, Point center) override;
        virtual Sector& rotate(float angleDegrees) override;
        virtual Sector& rotateOnSelf(float angleDegrees) override;

        // 几何计算
        virtual bool contains(Point point) const override;
        virtual Rect getBoundingBox() const override;
        virtual Point getCenter() const override;

        // 转换运算符
        operator Point() const;
    };
}
#endif // !SECTOR_H