// draw_flower.h - 花朵渲染函数
#pragma once
#include "pa2d.h"

/**
 * @brief 使用样式对象渲染花朵
 * @param src 目标画布
 * @param cx 花朵中心x坐标
 * @param cy 花朵中心y坐标
 * @param radius 花朵半径
 * @param angle 旋转角度(弧度)
 * @param style 花朵样式
 */
void drawFlower(
    pa2d::Canvas& src,
    float cx, float cy,
    float radius,
    float angle,
    const pa2d::Style style
);

/**
 * @brief 渲染花朵到画布上
 * @param src 目标画布
 * @param cx 花朵中心x坐标
 * @param cy 花朵中心y坐标
 * @param radius 花朵半径
 * @param angle 旋转角度(弧度)
 * @param fillColor 填充颜色
 * @param strokeColor 描边颜色
 * @param strokeWidth 描边宽度
 */
void drawFlower(
    pa2d::Canvas& src,
    float cx, float cy,
    float radius,
    float angle,
    const pa2d::Color& fillColor,
    const pa2d::Color& strokeColor,
    float strokeWidth
);