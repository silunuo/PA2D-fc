#pragma once
#include "buffer.h"
#include "color.h"
#include<string>
#pragma comment(lib, "gdiplus.lib")
namespace pa2d {
    enum class TextEncoding {ANSI,UTF8,CURRENT};
    TextEncoding textEncoding(TextEncoding newEncoding = TextEncoding::CURRENT);
    void setTextAntialias(bool enable);
    TextEncoding getTextEncoding();
    void setTextEncoding(TextEncoding encoding);

    class FontStyle {
    private:
        int styleBits_ = 0;
        float italicAngle_ = 0.0f;
        float rotationAngle_ = 0.0f;

    public:
        static const FontStyle Regular;
        static const FontStyle Bold;
        static const FontStyle Italic;
        static const FontStyle Underline;
        static const FontStyle Strikeout;

        FontStyle() = default;
        FontStyle(int styleBits);

        FontStyle italicAngle(float angle) const;
        FontStyle rotation(float angle) const;

        int getStyleBits() const;
        float getItalicAngle() const;
        float getRotationAngle() const;

        bool isBold() const;
        bool isItalic() const;
        bool isUnderline() const;
        bool isStrikeout() const;

        FontStyle operator|(const FontStyle& other) const;
        bool operator&(const FontStyle& other) const;
        bool operator==(const FontStyle& other) const;
    };

    bool measureText(const std::wstring& text, int fontSize, const std::wstring& fontName, const FontStyle& style, int& width, int& height);
    bool measureText(const std::string& text, int fontSize, const std::string& fontName, const FontStyle& style, int& width, int& height);

    int calculateFontSize(const std::wstring& text, float maxWidth, float maxHeight, int preferredFontSize = 12, const std::wstring& fontName = L"Microsoft YaHei", const FontStyle& style = FontStyle::Regular);
    int calculateFontSize(const std::string& text, float maxWidth, float maxHeight, int preferredFontSize = 12, const std::string& fontName = "Microsoft YaHei", const FontStyle& style = FontStyle::Regular);

    bool text(Buffer& buffer, float x, float y, const std::wstring& text, int fontSize, const Color& color, const FontStyle& style, const std::wstring& fontName);
    bool text(Buffer& buffer, float x, float y, const std::string& text, int fontSize, const Color& color, const FontStyle& style, const std::string& fontName);
    bool textCentered(Buffer& buffer, float centerX, float centerY, const std::wstring& text, int fontSize, const Color& color, const FontStyle& style, const std::wstring& fontName);
    bool textCentered(Buffer& buffer, float centerX, float centerY, const std::string& text, int fontSize, const Color& color, const FontStyle& style, const std::string& fontName);
    bool textInRect(Buffer& buffer, float rectX, float rectY, float rectWidth, float rectHeight, const std::wstring& text, int fontSize, const Color& color, const FontStyle& style, const std::wstring& fontName);
    bool textInRect(Buffer& buffer, float rectX, float rectY, float rectWidth, float rectHeight, const std::string& text, int fontSize, const Color& color, const FontStyle& style, const std::string& fontName);
    bool textFitRect(Buffer& buffer, float rectX, float rectY, float rectWidth, float rectHeight, const std::wstring& text, int preferredFontSize, const Color& color, const FontStyle& style, const std::wstring& fontName);
    bool textFitRect(Buffer& buffer, float rectX, float rectY, float rectWidth, float rectHeight, const std::string& text, int preferredFontSize, const Color& color, const FontStyle& style, const std::string& fontName);
}