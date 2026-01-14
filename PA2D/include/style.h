#pragma once
#include "color.h"
namespace pa2d {
    struct Style {
        Color fill_;    Style& fill(Color);
        Color stroke_;  Style& stroke(Color);
        float width_;   Style& width(float);
        float radius_; Style& radius(float);
        bool arc_;      Style& arc(bool);
        bool edges_;    Style& edges(bool);
        Style(Color fill_ = 0, Color stroke_ = 0, float width_ = 1.0f, float radius_ = 0.0f, bool arc_ = true, bool edges_ = true);
        Style operator+(const Style& other) const;
        Style& operator+=(const Style& other);
    };
    uint32_t parseColorString(const char* str, size_t len);
    extern const pa2d::Style
        arc, no_arc,    // Sector arc control
        edges, no_edges,  // Sector edges control
        White_fill, White_stroke,
        Black_fill, Black_stroke,
        None_fill, None_stroke,
        Red_fill, Red_stroke,
        Green_fill, Green_stroke,
        Blue_fill, Blue_stroke,
        Cyan_fill, Cyan_stroke,
        Magenta_fill, Magenta_stroke,
        Yellow_fill, Yellow_stroke,
        Gray_fill, Gray_stroke,
        LightGray_fill, LightGray_stroke,
        DarkGray_fill, DarkGray_stroke;
} // namespace pa2d
#ifndef PA2D_DISABLE_LITERALS
inline pa2d::Style operator"" _fill(unsigned long long hex) { return pa2d::Style().fill(static_cast<uint32_t>(hex)); }
inline pa2d::Style operator"" _stroke(unsigned long long hex) { return pa2d::Style().stroke(static_cast<uint32_t>(hex)); }
inline pa2d::Style operator"" _w(long double w) { return pa2d::Style().width(static_cast<float>(w)); }
inline pa2d::Style operator"" _w(unsigned long long w) { return pa2d::Style().width(static_cast<float>(w)); }
inline pa2d::Style operator"" _r(long double r) { return pa2d::Style().radius(static_cast<float>(r)); }
inline pa2d::Style operator"" _r(unsigned long long r) { return pa2d::Style().radius(static_cast<float>(r)); }
inline pa2d::Style operator"" _fill(const char* str, size_t len) { return pa2d::Style().fill(pa2d::parseColorString(str, len)); }
inline pa2d::Style operator"" _stroke(const char* str, size_t len) { return pa2d::Style().stroke(pa2d::parseColorString(str, len)); }
#endif