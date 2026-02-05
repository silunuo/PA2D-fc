// draw_particle.h - 粒子绘制函数
#pragma once
#include "pa2d.h"

/**
 * @brief 绘制渐变圆形粒子
 * @param src 目标画布
 * @param x 粒子中心x坐标
 * @param y 粒子中心y坐标
 * @param radius 粒子半径
 * @param centerColor 中心颜色
 * @param edgeColor 边缘颜色
 */
void drawParticle(
    pa2d::Canvas& src,
    float x, float y,
    float radius,
    const pa2d::Color& centerColor,
    const pa2d::Color& edgeColor
);