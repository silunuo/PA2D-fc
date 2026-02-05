/*
// extensions.cpp - 拓展渲染函数演示
// 
// 函数源码公开，但并包含在当前发布版本中，可自行复制源码进行使用
*/
#include"../extensions/graphics/include/advance_draw.h"

int main() {
	// 创建窗口
	pa2d::Window window(960, 640, "拓展函数绘制");
	// 创建画布
	pa2d::Canvas canvas(960, 640);
	// 绘制花朵
	drawFlower(canvas, 150.0f, 100.0f, 50.0f, 0.0f, 0xFF00FFFF_fill + 0xFFFF0000_stroke + 5_w);
	canvas.textCentered(150.0f, 170.0f, "花朵");
	// 绘制爱心
	drawHeart(canvas, 300.0f, 100.0f, 50.0f, pa2d::Red);
	canvas.textCentered(300.0f, 170.0f, "爱心");
	// 绘制粒子
	drawParticle(canvas, 450.0f, 100.0f, 50.0f, 0x80FF0000, 0x000000FF);
	canvas.textCentered(450.0f, 170.0f, "粒子");
	// 进行展示
	window.show()
		.render(canvas)
		.waitForClose();
}