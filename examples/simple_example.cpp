#include "pa2d.h"
pa2d::Window window(600, 400, "Draw a circle");
pa2d::Canvas canvas(600, 400);

int main() {
    window.show();
    canvas.circle(200, 200, 50, None, Red, 3);
    
    while (window.isOpen()) {
        window.render(canvas);
    }
    return 0;
}
