#pragma once
#include <cstdint>
namespace pa2d {
    struct Color {
        union {
            uint32_t data, argb;
            struct { uint8_t b, g, r, a; };
            struct { uint32_t rgb : 24, _ : 8; };
        };
        Color() :data(0) {}
        Color(uint32_t argb) :data(argb) {}
        Color(uint8_t a, uint8_t r, uint8_t g, uint8_t b) :a(a), r(r), g(g), b(b) {}
        Color(uint8_t r, uint8_t g, uint8_t b) : a(255), r(r), g(g), b(b) {}
        Color(uint8_t alpha, const Color& base) : data(((uint32_t)alpha << 24) | (base.data & 0x00FFFFFF)) {}
        operator uint32_t() const { return data; }
    };
    extern const Color White, Black,     None,
                       Red,   Green,     Blue,
                       Cyan,  Magenta,   Yellow,
                       Gray,  LightGray, DarkGray;
}
