#pragma once
#include "shape.h"
#include "point.h"
#include <array>
#include <vector>
#ifndef RECT_H
#define RECT_H
namespace pa2d {
    class Rect : public Shape {
    private:
        Point center_;
        float width_, height_, rotation_;
        mutable std::array<Point, 4> cachedVertices_;
        mutable bool verticesDirty_;

        void ensureVerticesUpdated() const;
        void updateVertices() const;
        void calculateParametersFromVertices();

    public:
        Rect();
        Rect(float centerX, float centerY, float width, float height, float rotation = 0.0f);
        Rect(const std::vector<Point>& points);
        Rect& operator=(const std::vector<Point>& points);

        // 属性设置
        Rect& center(const Point& center);
        Rect& center(float x, float y);
        Rect& width(float width);
        Rect& height(float height);
        Rect& rotation(float rotation);

        // 属性获取
        Point center() const;
        float width() const;
        float height() const;
        float rotation() const;

        // 迭代器支持
        auto begin();
        auto end();
        auto begin() const;
        auto end() const;
        Point& operator[](size_t index);
        const Point& operator[](size_t index) const;

        // 变换方法
        virtual Rect& translate(float dx, float dy) override;
        virtual Rect& translate(Point delta) override;
        virtual Rect& scale(float factor) override;
        virtual Rect& scale(float factorX, float factorY) override;
        virtual Rect& scaleOnSelf(float factorX, float factorY) override;
        virtual Rect& scaleOnSelf(float factor) override;
        virtual Rect& rotate(float angleDegrees, float centerX, float centerY) override;
        virtual Rect& rotate(float angleDegrees, Point center) override;
        virtual Rect& rotate(float angleDegrees) override;
        virtual Rect& rotateOnSelf(float angleDegrees) override;

        // 几何计算
        virtual bool contains(Point point) const override;
        virtual Rect getBoundingBox() const override;
        virtual Point getCenter() const override;

        // 转换运算符
        operator std::vector<Point>() const;
    };
}
#endif // !RECT_H