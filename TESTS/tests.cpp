#include"pa2d.h"
using namespace pa2d;

int main() {
    // 创建一个画布
    Canvas canvas(640, 480);
    // 构造样式： 半透明红色填充 + 半透明绿色线框 + 3像素线宽 + 25弧度圆角
    Style style = 0x80FF0000_fill + 0x800000FF_stroke + 3_w + 25_r;
    // 绘制圆角矩形
    canvas.rect(100, 100, 200, 100, style);
    // 创建窗口
    Window window(640, 480, "My window");
    // 显式窗口 -> 渲染画布 -> 堵塞线程
    window.show()
        .render(canvas)
        .waitForClose();
}