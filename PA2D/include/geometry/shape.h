#pragma once
#include <vector>
#include "point.h"
#ifndef SHAPE_H
#define SHAPE_H
namespace pa2d {
    class Rect;  // 前向声明
    class Shape {
    public:
        enum class GeometryType {
            POINTS,
            LINE,
            POLYGON,
            RECT,
            TRIANGLE,
            CIRCLE,
            ELLIPTIC,
            SECTOR,
            RAY
        };
        GeometryType getType() const { return type_; }

        // 平移
        virtual Shape& translate(float dx, float dy) = 0;
        virtual Shape& translate(Point delta) = 0;

        // 缩放
        virtual Shape& scale(float factor) = 0;
        virtual Shape& scale(float factorX, float factorY) = 0;
        virtual Shape& scaleOnSelf(float factorX, float factorY) = 0;
        virtual Shape& scaleOnSelf(float factor) = 0;

        // 旋转
        virtual Shape& rotate(float angleDegrees, float centerX, float centerY) = 0;
        virtual Shape& rotate(float angleDegrees, Point center) = 0;
        virtual Shape& rotate(float angleDegrees) = 0;
        virtual Shape& rotateOnSelf(float angleDegrees) = 0;

        // 几何计算
        virtual bool contains(Point point) const = 0;
        virtual Rect getBoundingBox() const = 0;
        virtual Point getCenter() const = 0;

    protected:
        Shape(const GeometryType& type);

    private:
        GeometryType type_ = GeometryType::POINTS;
    };
}
#endif // !SHAPE_H