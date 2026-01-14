#pragma once
#include <cmath>
#ifndef POINT_H
#define POINT_H
namespace pa2d {
    struct Point {
        float x, y;

        Point() = default;
        Point(float x, float y);

        // 运算符重载
        Point operator+(const Point&) const;
        Point& operator+=(const Point&);
        Point operator-(const Point&) const;
        Point& operator-=(const Point&);
        Point operator-() const;
        bool operator==(const Point&) const;
        bool operator!=(const Point&) const;

        Point& operator+=(float);
        Point operator+(float) const;
        Point& operator-=(float);
        Point operator-(float) const;
        Point& operator*=(float);
        Point operator*(float) const;
        Point& operator/=(float);
        Point operator/(float) const;

        friend Point operator*(float, const Point&);

        // 变换方法
        Point& translate(float dx, float dy);
        Point& scale(float factorX, float factorY);
        Point& scale(float factor);
        Point& rotate(float angleDegrees, float centerX = 0, float centerY = 0);
    };

    struct PointInt {
        int x, y;
    };

    struct Size {
        int width, height;
    };
}
#endif // !POINT_H