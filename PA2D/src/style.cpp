// style.cpp
#include "../include/style.h"
#include <algorithm>
#include <string>
namespace pa2d {
    // 常量定义
    const Style arc = [] {return Style().arc(true);}();
    const Style no_arc = [] {return Style().arc(false);}();
    const Style edges = [] {return Style().edges(true);}();
    const Style no_edges = [] {return Style().edges(false);}();
    const Style
        White_fill = 0xFFFFFFFF_fill, White_stroke = 0xFFFFFFFF_stroke,
        Black_fill = 0xFF000000_fill, Black_stroke = 0xFF000000_stroke,
        None_fill = 0x00000000_fill, None_stroke = 0x00000000_stroke,
        Red_fill = 0xFFFF0000_fill, Red_stroke = 0xFFFF0000_stroke,
        Green_fill = 0xFF00FF00_fill, Green_stroke = 0xFF00FF00_stroke,
        Blue_fill = 0xFF0000FF_fill, Blue_stroke = 0xFF0000FF_stroke,
        Cyan_fill = 0xFF00FFFF_fill, Cyan_stroke = 0xFF00FFFF_stroke,
        Magenta_fill = 0xFFFF00FF_fill, Magenta_stroke = 0xFFFF00FF_stroke,
        Yellow_fill = 0xFFFFFF00_fill, Yellow_stroke = 0xFFFFFF00_stroke,
        Gray_fill = 0xFF808080_fill, Gray_stroke = 0xFF808080_stroke,
        LightGray_fill = 0xFFC0C0C0_fill, LightGray_stroke = 0xFFC0C0C0_stroke,
        DarkGray_fill = 0xFF404040_fill, DarkGray_stroke = 0xFF404040_stroke;
    // 构造函数
    Style& Style::fill(Color v) { fill_ = v; return *this; }
    Style& Style::stroke(Color v) { stroke_ = v; return *this; }
    Style& Style::width(float v) { width_ = v; return *this; }
    Style& Style::radius(float v) { radius_ = v; return *this; }
    Style& Style::arc(bool v) { arc_ = v; return *this; }
    Style& Style::edges(bool v) { edges_ = v; return *this; }
    Style::Style(Color fill, Color stroke, float width,float radius, bool arc, bool edges)
        : fill_(fill), stroke_(stroke), width_(width),radius_(radius), arc_(arc), edges_(edges) {}

    // 加法运算符
    Style Style::operator+(const Style& other) const {
        Style result = *this;
        if (other.fill_ != 0) result.fill_ = other.fill_;
        if (other.stroke_ != 0) result.stroke_ = other.stroke_;
        if (other.width_ != 1.0f) result.width_ = other.width_;
        if (other.radius_ != 0.0f) result.radius_ = other.radius_;
        if (other.arc_ != true) result.arc_ = other.arc_;
        if (other.edges_ != true) result.edges_ = other.edges_;
        return result;
    }

    Style& Style::operator+=(const Style& other) {
        *this = *this + other;
        return *this;
    }

    // 颜色字符串解析辅助函数
    namespace {
        inline uint8_t hexToDigit(char c) {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return 0;
        }
    }

    uint32_t parseColorString(const char* str, size_t len) {
        std::string s(str, len);

        // 处理 #RRGGBB 或 #AARRGGBB 格式
        if (!s.empty() && s[0] == '#') {
            uint32_t color = 0;
            for (size_t i = 1; i < s.length(); ++i) {
                color = (color << 4) | hexToDigit(s[i]);
            }

            // 如果是6位（# + 6个字符），添加Alpha
            if (s.length() == 7) {
                color |= 0xFF000000;
            }
            return color;
        }

        // 处理 R,G,B 或 A,R,G,B 格式
        size_t pos1 = s.find(',');
        if (pos1 == std::string::npos) {
            return 0x00000000;  // 无效格式
        }

        size_t pos2 = s.find(',', pos1 + 1);
        if (pos2 == std::string::npos) {
            return 0x00000000;  // 至少需要两个逗号
        }

        size_t pos3 = s.find(',', pos2 + 1);

        if (pos3 == std::string::npos) {
            // R,G,B 格式 - 两个逗号
            int r = std::min(std::max(std::stoi(s.substr(0, pos1)), 0), 255);
            int g = std::min(std::max(std::stoi(s.substr(pos1 + 1, pos2 - pos1 - 1)), 0), 255);
            int b = std::min(std::max(std::stoi(s.substr(pos2 + 1)), 0), 255);
            return (255u << 24) | (r << 16) | (g << 8) | b;
        }
        else {
            // A,R,G,B 格式 - 三个逗号
            int a = std::min(std::max(std::stoi(s.substr(0, pos1)), 0), 255);
            int r = std::min(std::max(std::stoi(s.substr(pos1 + 1, pos2 - pos1 - 1)), 0), 255);
            int g = std::min(std::max(std::stoi(s.substr(pos2 + 1, pos3 - pos2 - 1)), 0), 255);
            int b = std::min(std::max(std::stoi(s.substr(pos3 + 1)), 0), 255);
            return (a << 24) | (r << 16) | (g << 8) | b;
        }
    }
} // namespace pa2d