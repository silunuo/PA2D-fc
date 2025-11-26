#include "pa2d.h"
pa2d::Window window(600, 400, "Draw a circle"); // Create a window - 创建一个窗口对象
pa2d::Canvas canvas(600, 400);                  // Create a canvas - 创建一个画布对象
int main() {
    window.show(); // Make the window visible - 需手动显示窗口
    canvas.circle(200, 200, 50, None,Red,3); // a circle on the canvas - 绘制圆在画布上
    while (window.isOpen()) {
        window.render(canvas); // Read the canvas and draw it onto the window - 绘制到窗口上
    }
    return 0;
}
