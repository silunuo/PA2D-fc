// draw_heart.h - 心形渲染函数
#pragma once
#include "pa2d.h"

/**
 * @brief SIMD优化的高质量心形渲染
 * @param canvas 目标画布
 * @param cx 心形中心x坐标
 * @param cy 心形中心y坐标
 * @param size 心形尺寸
 * @param color 心形颜色
 */
void drawHeart(
    pa2d::Canvas& canvas,
    float cx, float cy,
    float size,
    const pa2d::Color& color
);