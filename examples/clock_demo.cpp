#include "pa2d.h"
using namespace pa2d;
const float ClockX(300), ClockY(200), Radius(120);
Window window(600, 400, "PA2D Clock");
Canvas canvas(600, 400), background(600, 400, 0xffffffff);
int main() {
    for (int i = 5; i < 65; ++i) {
        Ray mark(ClockX, ClockY - Radius, ClockX, ClockY - Radius + 10);
        background.draw(mark.rotate(i * 6.0f, ClockX, ClockY), i % 5 ? "#ff787878"_stroke + 1_w : "#ff464646"_stroke + 3_w);
        if (i % 5 == 0) {
            auto pos = mark.stretch(3).end();
            background.textCentered(std::to_string(i / 5), pos.x - 2, pos.y, 0xff3c3c64);
        }
    }
    background.circle(ClockX, ClockY, Radius, None, 0xff5a5a5a, 4);
    background.textCentered("PA2D Clock", ClockX, 50, 0xff505078, 20, "Microsoft YaHei", FontStyle::Bold);
    window.show();
    while (window.isOpen()) {
        SYSTEMTIME st;
        GetLocalTime(&st);
        static int lastSecond = -1;
        if (st.wSecond != std::exchange(lastSecond, st.wSecond)) {
            char timeStr[9];
            sprintf_s(timeStr, "%02d:%02d:%02d", st.wHour, st.wMinute, st.wSecond);
            Ray hand(ClockX, ClockY, ClockX, ClockY-Radius * 0.5f);
            float h_angle = st.wHour % 12 * 30.0f + st.wMinute * 0.5f;
            float m_angle = st.wMinute * 6.0f + st.wSecond * 0.1f;
            float s_angle = st.wSecond * 6.0f;
            canvas.copyBlend(background)
                .draw(Ray(hand).spin(h_angle), "#ff1e1e1e"_stroke + 8_w)
                .draw(Ray(hand).stretch(1.4).spin(m_angle), "#ff505050"_stroke + 5_w)
                .draw(Ray(hand).stretch(1.7).spin(s_angle), "#ffdc0000"_stroke + 2_w)
                .circle(ClockX, ClockY, 4, 0xffffffff, 0xff1e1e1e, 4.5)
                .textCentered(timeStr, ClockX, ClockY + Radius + 30, 0xff505078, 20);
        }
        window.renderCentered(canvas);
        Sleep(16);
    }
    return 0;
}
