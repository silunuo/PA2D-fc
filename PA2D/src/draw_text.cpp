#include "../include/draw_text.h"
#include <string>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
#include <gdiplus.h>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <functional>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace pa2d {
    TextEncoding textEncoding(TextEncoding newEncoding) {
        static TextEncoding encoding = TextEncoding::ANSI;
        if (newEncoding != TextEncoding::CURRENT) {
            encoding = newEncoding;
        }
        return encoding;
    }

    TextEncoding getTextEncoding() {
        return textEncoding(TextEncoding::CURRENT);
    }

    void setTextEncoding(TextEncoding encoding) {
        textEncoding(encoding);
    }

    class GdiplusManager {
    private:
        ULONG_PTR m_gdiplusToken;
        bool m_initialized;
        HDC m_measureHDC;
        bool m_antiAliasingEnabled;
        std::unique_ptr<Gdiplus::Graphics> m_measureGraphics;
        HBITMAP m_measureBitmap;
        HBITMAP m_oldBitmap;

        void CleanupGDIPlus();
        GdiplusManager();
        ~GdiplusManager();

    public:
        bool InitializeGDIPlus();
        static GdiplusManager& GetInstance();
        void SetAntiAliasing(bool enable);
        bool IsAntiAliasingEnabled() const;
        Gdiplus::Graphics* GetMeasureGraphics();
        HDC GetMeasureHDC() { return m_measureHDC; }

        GdiplusManager(const GdiplusManager&) = delete;
        GdiplusManager& operator=(const GdiplusManager&) = delete;
    };

    void setTextAntialias(bool enable) {
        GdiplusManager::GetInstance().SetAntiAliasing(enable);
    }

    const FontStyle FontStyle::Regular = FontStyle(0);
    const FontStyle FontStyle::Bold = FontStyle(1);
    const FontStyle FontStyle::Italic = FontStyle(2).italicAngle(15.0f);
    const FontStyle FontStyle::Underline = FontStyle(4);
    const FontStyle FontStyle::Strikeout = FontStyle(8);

    FontStyle::FontStyle(int styleBits) : styleBits_(styleBits) {}

    FontStyle FontStyle::italicAngle(float angle) const {
        FontStyle newStyle = *this;
        newStyle.italicAngle_ = angle;
        if (std::abs(angle) > 0.1f) newStyle.styleBits_ |= 2;
        return newStyle;
    }

    FontStyle FontStyle::rotation(float angle) const {
        FontStyle newStyle = *this;
        newStyle.rotationAngle_ = angle;
        return newStyle;
    }

    int FontStyle::getStyleBits() const { return styleBits_; }
    float FontStyle::getItalicAngle() const { return italicAngle_; }
    float FontStyle::getRotationAngle() const { return rotationAngle_; }

    bool FontStyle::isBold() const { return (styleBits_ & 1) != 0; }
    bool FontStyle::isItalic() const { return (styleBits_ & 2) != 0; }
    bool FontStyle::isUnderline() const { return (styleBits_ & 4) != 0; }
    bool FontStyle::isStrikeout() const { return (styleBits_ & 8) != 0; }

    FontStyle FontStyle::operator|(const FontStyle& other) const {
        FontStyle result;
        result.styleBits_ = styleBits_ | other.styleBits_;
        result.italicAngle_ = (other.italicAngle_ != 0.0f) ? other.italicAngle_ : italicAngle_;
        result.rotationAngle_ = (other.rotationAngle_ != 0.0f) ? other.rotationAngle_ : rotationAngle_;
        return result;
    }

    bool FontStyle::operator&(const FontStyle& other) const {
        return (styleBits_ & other.styleBits_) != 0;
    }

    bool FontStyle::operator==(const FontStyle& other) const {
        return styleBits_ == other.styleBits_ &&
            italicAngle_ == other.italicAngle_ &&
            rotationAngle_ == other.rotationAngle_;
    }

    namespace internal {
        std::wstring to_wstring_ansi(const std::string& str);
        std::wstring to_wstring_utf8(const std::string& str);
        std::wstring to_wstring_auto(const std::string& str);

        bool DrawAdvancedText(Gdiplus::Graphics& graphics, const std::wstring& text,
            float x, float y, float rotCenterX, float rotCenterY,
            Gdiplus::Font* font, const pa2d::Color& textColor,
            const FontStyle& style);

        bool Text(Buffer& buffer, const std::wstring& text, float x, float y, float rotCenterX, float rotCenterY,
            const Color& textColor, int fontSize,
            const std::wstring& fontName, const FontStyle& style);

        class FontCache {
        private:
            struct FontKey {
                std::wstring fontName;
                int fontSize;
                bool bold;
                bool italic;
                bool underline;
                bool strikeout;

                bool operator==(const FontKey& other) const {
                    return fontName == other.fontName &&
                        fontSize == other.fontSize &&
                        bold == other.bold &&
                        italic == other.italic &&
                        underline == other.underline &&
                        strikeout == other.strikeout;
                }
            };

            struct FontKeyHash {
                size_t operator()(const FontKey& key) const {
                    size_t h1 = std::hash<std::wstring>{}(key.fontName);
                    size_t h2 = std::hash<int>{}(key.fontSize);
                    return h1 ^ (h2 << 1) ^
                        (key.bold ? 0x10 : 0) ^
                        (key.italic ? 0x20 : 0) ^
                        (key.underline ? 0x40 : 0) ^
                        (key.strikeout ? 0x80 : 0);
                }
            };

            std::unordered_map<FontKey, std::unique_ptr<Gdiplus::Font>, FontKeyHash> fontCache_;

        public:
            Gdiplus::Font* GetFont(const std::wstring& fontName, int fontSize, const FontStyle& style);
            void Clear();
        };

        class TextCacheManager {
        private:
            FontCache fontCache_;
            TextCacheManager() = default;

        public:
            static TextCacheManager& GetInstance();
            static Gdiplus::Font* GetFont(const std::wstring& name, int size, const FontStyle& style);
            static void ClearAll();

            TextCacheManager(const TextCacheManager&) = delete;
            TextCacheManager& operator=(const TextCacheManager&) = delete;
        };
    }

    namespace internal {
        std::wstring to_wstring_ansi(const std::string& str) {
            if (str.empty()) return L"";
            int len = MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, NULL, 0);
            if (len == 0) return L"";
            std::wstring res(len - 1, 0);
            MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, &res[0], len);
            return res;
        }

        std::wstring to_wstring_utf8(const std::string& str) {
            if (str.empty()) return L"";
            int len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, NULL, 0);
            if (len == 0) return L"";
            std::wstring res(len - 1, 0);
            MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &res[0], len);
            return res;
        }

        std::wstring to_wstring_auto(const std::string& str) {
            return (textEncoding() == TextEncoding::UTF8) ? to_wstring_utf8(str) : to_wstring_ansi(str);
        }

        bool DrawAdvancedText(Gdiplus::Graphics& graphics, const std::wstring& text,
            float x, float y, float rotCenterX, float rotCenterY,
            Gdiplus::Font* font, const pa2d::Color& textColor,
            const FontStyle& style) {

            if (!font || text.empty()) return false;

            Gdiplus::Color gdiColor(textColor.argb);
            Gdiplus::SolidBrush brush(gdiColor);

            graphics.SetTextRenderingHint(Gdiplus::TextRenderingHintAntiAlias);

            Gdiplus::Matrix originalMatrix;
            graphics.GetTransform(&originalMatrix);

            Gdiplus::REAL rcX = (Gdiplus::REAL)rotCenterX;
            Gdiplus::REAL rcY = (Gdiplus::REAL)rotCenterY;
            Gdiplus::REAL drawX = (Gdiplus::REAL)x;
            Gdiplus::REAL drawY = (Gdiplus::REAL)y;

            Gdiplus::Matrix M_new;

            M_new.Translate(-rcX, -rcY, Gdiplus::MatrixOrderAppend);

            if (style.getRotationAngle() != 0.0f) {
                M_new.Rotate(style.getRotationAngle(), Gdiplus::MatrixOrderAppend);
            }

            M_new.Translate(rcX, rcY, Gdiplus::MatrixOrderAppend);

            float italicAngle = style.getItalicAngle();
            if (italicAngle != 0.0f) {
                Gdiplus::Matrix shearMatrix;
                float shearFactor = -std::tan(italicAngle * (float)(M_PI / 180.0f));
                shearMatrix.Shear(shearFactor, 0.0f);
                M_new.Multiply(&shearMatrix, Gdiplus::MatrixOrderAppend);
            }

            graphics.MultiplyTransform(&M_new, Gdiplus::MatrixOrderAppend);

            Gdiplus::StringFormat format(Gdiplus::StringFormat::GenericTypographic());

            format.SetAlignment(Gdiplus::StringAlignmentNear);
            format.SetLineAlignment(Gdiplus::StringAlignmentNear);

            format.SetFormatFlags(format.GetFormatFlags() | Gdiplus::StringFormatFlagsMeasureTrailingSpaces);
            format.SetTrimming(Gdiplus::StringTrimmingNone);

            Gdiplus::PointF origin(drawX, drawY);

            Gdiplus::Status status = graphics.DrawString(text.c_str(), -1, font, origin, &format, &brush);

            graphics.SetTransform(&originalMatrix);

            return status == Gdiplus::Ok;
        }

        bool Text(Buffer& buffer, const std::wstring& text, float x, float y, float rotCenterX, float rotCenterY,
            const Color& textColor, int fontSize,
            const std::wstring& fontName, const FontStyle& style) {

            if (!buffer.isValid() || text.empty() || textColor.a == 0)
                return false;

            if (!GdiplusManager::GetInstance().GetMeasureGraphics()) {
                return false;
            }

            int stride = buffer.width * sizeof(Color);
            Gdiplus::Bitmap bmp(buffer.width, buffer.height, stride,
                PixelFormat32bppARGB,
                reinterpret_cast<BYTE*>(buffer.color));

            if (bmp.GetLastStatus() != Gdiplus::Ok)
                return false;

            Gdiplus::Graphics graphics(&bmp);

            if (GdiplusManager::GetInstance().IsAntiAliasingEnabled()) {
                graphics.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
                graphics.SetTextRenderingHint(Gdiplus::TextRenderingHintAntiAlias);
            }
            else {
                graphics.SetSmoothingMode(Gdiplus::SmoothingModeNone);
                graphics.SetTextRenderingHint(Gdiplus::TextRenderingHintSystemDefault);
            }

            Gdiplus::Font* font = TextCacheManager::GetFont(fontName, fontSize, style);
            if (!font)
                return false;

            return DrawAdvancedText(graphics, text, x, y, rotCenterX, rotCenterY, font, textColor, style);
        }

        Gdiplus::Font* FontCache::GetFont(const std::wstring& fontName, int fontSize, const FontStyle& style) {
            FontKey key{
                fontName,
                fontSize,
                style.isBold(),
                style.isItalic(),
                style.isUnderline(),
                style.isStrikeout()
            };

            auto it = fontCache_.find(key);
            if (it != fontCache_.end()) return it->second.get();

            LOGFONTW lf = { 0 };
            lf.lfHeight = -fontSize;
            lf.lfWeight = style.isBold() ? FW_BOLD : FW_NORMAL;
            lf.lfItalic = style.isItalic() ? TRUE : FALSE;
            lf.lfUnderline = style.isUnderline() ? TRUE : FALSE;
            lf.lfStrikeOut = style.isStrikeout() ? TRUE : FALSE;
            lf.lfCharSet = DEFAULT_CHARSET;
            lf.lfOutPrecision = OUT_TT_PRECIS;
            lf.lfQuality = CLEARTYPE_QUALITY;
            lf.lfPitchAndFamily = DEFAULT_PITCH | FF_DONTCARE;
            wcsncpy_s(lf.lfFaceName, fontName.c_str(), LF_FACESIZE - 1);

            HDC hdc = GdiplusManager::GetInstance().GetMeasureHDC();
            auto font = std::make_unique<Gdiplus::Font>(hdc, &lf);

            if (!font || font->GetLastStatus() != Gdiplus::Ok) {
                font = std::make_unique<Gdiplus::Font>(L"Microsoft YaHei", (Gdiplus::REAL)fontSize, Gdiplus::FontStyleRegular, Gdiplus::UnitPixel);
            }

            if (!font || font->GetLastStatus() != Gdiplus::Ok) {
                return nullptr;
            }

            Gdiplus::Font* result = font.get();
            fontCache_.emplace(key, std::move(font));
            return result;
        }

        void FontCache::Clear() { fontCache_.clear(); }

        TextCacheManager& TextCacheManager::GetInstance() {
            static TextCacheManager instance;
            return instance;
        }

        Gdiplus::Font* TextCacheManager::GetFont(const std::wstring& name, int size, const FontStyle& style) {
            return GetInstance().fontCache_.GetFont(name, size, style);
        }

        void TextCacheManager::ClearAll() {
            GetInstance().fontCache_.Clear();
        }
    }

    void GdiplusManager::CleanupGDIPlus() {
        if (!m_initialized) return;

        m_measureGraphics.reset();

        if (m_measureHDC) {
            if (m_measureBitmap) {
                SelectObject(m_measureHDC, m_oldBitmap);
                DeleteObject(m_measureBitmap);
                m_measureBitmap = NULL;
            }
            DeleteDC(m_measureHDC);
            m_measureHDC = NULL;
        }

        if (m_gdiplusToken) {
            Gdiplus::GdiplusShutdown(m_gdiplusToken);
            m_gdiplusToken = 0;
        }

        m_initialized = false;
    }

    GdiplusManager::GdiplusManager()
        : m_gdiplusToken(0)
        , m_initialized(false)
        , m_measureHDC(NULL)
        , m_antiAliasingEnabled(true)
        , m_measureBitmap(NULL)
        , m_oldBitmap(NULL) {
    }

    GdiplusManager::~GdiplusManager() {
        CleanupGDIPlus();
    }

    bool GdiplusManager::InitializeGDIPlus() {
        if (m_initialized) return true;

        Gdiplus::GdiplusStartupInput input;
        input.GdiplusVersion = 1;
        input.DebugEventCallback = nullptr;
        input.SuppressBackgroundThread = FALSE;

        Gdiplus::Status status = Gdiplus::GdiplusStartup(&m_gdiplusToken, &input, NULL);
        if (status != Gdiplus::Ok) {
            return false;
        }

        m_measureHDC = CreateCompatibleDC(NULL);
        if (!m_measureHDC) {
            Gdiplus::GdiplusShutdown(m_gdiplusToken);
            return false;
        }

        m_measureBitmap = CreateCompatibleBitmap(GetDC(NULL), 1, 1);
        if (!m_measureBitmap) {
            DeleteDC(m_measureHDC);
            Gdiplus::GdiplusShutdown(m_gdiplusToken);
            return false;
        }

        m_oldBitmap = (HBITMAP)SelectObject(m_measureHDC, m_measureBitmap);

        m_measureGraphics = std::make_unique<Gdiplus::Graphics>(m_measureHDC);
        if (!m_measureGraphics || m_measureGraphics->GetLastStatus() != Gdiplus::Ok) {
            if (m_measureBitmap) {
                SelectObject(m_measureHDC, m_oldBitmap);
                DeleteObject(m_measureBitmap);
            }
            DeleteDC(m_measureHDC);
            Gdiplus::GdiplusShutdown(m_gdiplusToken);
            return false;
        }

        m_measureGraphics->SetTextRenderingHint(Gdiplus::TextRenderingHintAntiAlias);
        m_initialized = true;

        return true;
    }

    GdiplusManager& GdiplusManager::GetInstance() {
        static GdiplusManager instance;
        return instance;
    }

    void GdiplusManager::SetAntiAliasing(bool enable) {
        m_antiAliasingEnabled = enable;
    }

    bool GdiplusManager::IsAntiAliasingEnabled() const {
        return m_antiAliasingEnabled;
    }

    Gdiplus::Graphics* GdiplusManager::GetMeasureGraphics() {
        if (!m_initialized && !InitializeGDIPlus()) {
            return nullptr;
        }
        return m_measureGraphics.get();
    }

    bool initializeTextRenderer() {
        return GdiplusManager::GetInstance().InitializeGDIPlus();
    }

    void shutdownTextRenderer() {
        internal::TextCacheManager::ClearAll();
    }

    bool measureText(const std::wstring& text, int fontSize, const std::wstring& fontName,
        const FontStyle& style, int& width, int& height) {

        if (!GdiplusManager::GetInstance().GetMeasureGraphics() || text.empty() || fontSize <= 0) {
            width = 0; height = 0;
            return false;
        }

        Gdiplus::Font* font = internal::TextCacheManager::GetFont(fontName, fontSize, style);
        if (!font) {
            width = 0; height = 0;
            return false;
        }

        Gdiplus::Graphics* g = GdiplusManager::GetInstance().GetMeasureGraphics();
        Gdiplus::RectF rect;
        Gdiplus::RectF layoutRect(0, 0, 10000.0f, 10000.0f);

        Gdiplus::StringFormat format(Gdiplus::StringFormat::GenericTypographic());
        format.SetAlignment(Gdiplus::StringAlignmentNear);
        format.SetFormatFlags(format.GetFormatFlags() | Gdiplus::StringFormatFlagsMeasureTrailingSpaces);

        Gdiplus::Status status = g->MeasureString(text.c_str(), -1, font, layoutRect, &format, &rect);

        if (status != Gdiplus::Ok) {
            width = 0; height = 0;
            return false;
        }

        width = (int)std::ceil(rect.Width);
        height = (int)std::ceil(rect.Height);

        return true;
    }

    bool measureText(const std::string& text, int fontSize, const std::string& fontName,
        const FontStyle& style, int& width, int& height) {
        return measureText(internal::to_wstring_auto(text), fontSize, internal::to_wstring_auto(fontName), style, width, height);
    }

    int calculateFontSize(const std::wstring& text, float maxWidth, float maxHeight,
        int preferredFontSize, const std::wstring& fontName, const FontStyle& style) {

        if (text.empty() || maxWidth <= 0 || maxHeight <= 0)
            return 1;

        bool wantMaxFill = (preferredFontSize <= 0);

        if (!wantMaxFill) {
            int w, h;
            if (measureText(text, preferredFontSize, fontName, style, w, h)
                && w <= maxWidth && h <= maxHeight) {
                return preferredFontSize;
            }
        }

        int minSize = 1;
        int maxSize = std::min(static_cast<int>(maxWidth), static_cast<int>(maxHeight));
        int optimalSize = minSize;

        while (minSize <= maxSize) {
            int midSize = (minSize + maxSize) / 2;
            int w = 0, h = 0;
            if (measureText(text, midSize, fontName, style, w, h)) {
                if (w <= maxWidth && h <= maxHeight) {
                    optimalSize = midSize;
                    minSize = midSize + 1;
                }
                else {
                    maxSize = midSize - 1;
                }
            }
            else {
                maxSize = midSize - 1;
            }
        }

        return optimalSize;
    }

    int calculateFontSize(const std::string& text, float maxWidth, float maxHeight,
        int preferredFontSize, const std::string& fontName, const FontStyle& style) {
        return calculateFontSize(internal::to_wstring_auto(text), maxWidth, maxHeight,
            preferredFontSize, internal::to_wstring_auto(fontName), style);
    }

    bool text(Buffer& buffer, float x, float y, const std::wstring& text,
        int fontSize, const Color& textColor, const FontStyle& style, const std::wstring& fontName) {
        return internal::Text(buffer, text, x, y, x, y, textColor, fontSize, fontName, style);
    }

    bool text(Buffer& buffer, float x, float y, const std::string& text,
        int fontSize, const Color& textColor, const FontStyle& style, const std::string& fontName) {
        return pa2d::text(buffer, x, y, internal::to_wstring_auto(text),
            fontSize, textColor, style, internal::to_wstring_auto(fontName));
    }

    bool textInRect(Buffer& buffer, float rectX, float rectY, float rectWidth, float rectHeight, const std::wstring& text,
        int fontSize, const Color& color, const FontStyle& style, const std::wstring& fontName) {
        if (!buffer.isValid() || text.empty()) return false;

        std::vector<std::wstring> lines;
        size_t start = 0;
        size_t end = text.find(L'\n');

        while (end != std::wstring::npos) {
            std::wstring line = text.substr(start, end - start);
            if (!line.empty() && line.back() == L'\r') {
                line.pop_back();
            }
            lines.push_back(line);
            start = end + 1;
            end = text.find(L'\n', start);
        }
        std::wstring lastLine = text.substr(start);
        if (!lastLine.empty() && lastLine.back() == L'\r') {
            lastLine.pop_back();
        }
        lines.push_back(lastLine);

        int singleLineHeight = 0;
        int maxLineWidth = 0;
        std::vector<int> lineWidths;

        for (const auto& line : lines) {
            int w = 0, h = 0;
            if (measureText(line, fontSize, fontName, style, w, h)) {
                lineWidths.push_back(w);
                maxLineWidth = std::max(maxLineWidth, w);
                if (singleLineHeight == 0) singleLineHeight = h;
            }
            else {
                lineWidths.push_back(0);
            }
        }

        int totalHeight = singleLineHeight * (int)lines.size();
        if (totalHeight == 0) return false;

        float centerX = rectX + rectWidth / 2.0f;
        float centerY = rectY + rectHeight / 2.0f;

        float currentY = centerY - (float)totalHeight / 2.0f;

        for (size_t i = 0; i < lines.size(); ++i) {
            if (lines[i].empty()) {
                currentY += singleLineHeight;
                continue;
            }
            float x = centerX - (float)lineWidths[i] / 2.0f;
            internal::Text(buffer, lines[i], x, currentY, centerX, centerY, color, fontSize, fontName, style);
            currentY += singleLineHeight;
        }
        return true;
    }

    bool textInRect(Buffer& buffer, float rectX, float rectY, float rectWidth, float rectHeight, const std::string& text,
        int fontSize, const Color& color, const FontStyle& style, const std::string& fontName) {
        return textInRect(buffer, rectX, rectY, rectWidth, rectHeight, internal::to_wstring_auto(text),
            fontSize, color, style, internal::to_wstring_auto(fontName));
    }

    bool textFitRect(Buffer& buffer, float rectX, float rectY, float rectWidth, float rectHeight, const std::wstring& text,
        int preferredFontSize, const Color& color, const FontStyle& style, const std::wstring& fontName) {
        if (!buffer.isValid() || text.empty()) return false;

        int optimalSize = calculateFontSize(text, rectWidth, rectHeight,
            preferredFontSize, fontName, style);

        return textInRect(buffer, rectX, rectY, rectWidth, rectHeight, text,
            optimalSize, color, style, fontName);
    }

    bool textFitRect(Buffer& buffer, float rectX, float rectY, float rectWidth, float rectHeight, const std::string& text,
        int preferredFontSize, const Color& color, const FontStyle& style, const std::string& fontName) {
        return textFitRect(buffer, rectX, rectY, rectWidth, rectHeight, internal::to_wstring_auto(text),
            preferredFontSize, color, style, internal::to_wstring_auto(fontName));
    }

    bool textCentered(Buffer& buffer, float centerX, float centerY, const std::wstring& text,
        int fontSize, const Color& color, const FontStyle& style, const std::wstring& fontName) {

        if (!buffer.isValid() || text.empty()) return false;

        std::vector<std::wstring> lines;
        size_t start = 0;
        size_t end = text.find(L'\n');

        while (end != std::wstring::npos) {
            std::wstring line = text.substr(start, end - start);
            if (!line.empty() && line.back() == L'\r') { line.pop_back(); }
            lines.push_back(line);
            start = end + 1;
            end = text.find(L'\n', start);
        }
        std::wstring lastLine = text.substr(start);
        if (!lastLine.empty() && lastLine.back() == L'\r') { lastLine.pop_back(); }
        lines.push_back(lastLine);

        int singleLineHeight = 0;
        std::vector<int> lineWidths;
        for (const auto& line : lines) {
            int w = 0, h = 0;
            if (measureText(line, fontSize, fontName, style, w, h)) {
                lineWidths.push_back(w);
                if (singleLineHeight == 0) singleLineHeight = h;
            }
            else { lineWidths.push_back(0); }
        }

        int totalHeight = singleLineHeight * (int)lines.size();
        if (totalHeight == 0) return false;

        float currentY = centerY - (float)totalHeight / 2.0f;

        for (size_t i = 0; i < lines.size(); ++i) {
            if (lines[i].empty()) {
                currentY += singleLineHeight;
                continue;
            }
            float x = centerX - (float)lineWidths[i] / 2.0f;
            internal::Text(buffer, lines[i], x, currentY, centerX, centerY, color, fontSize, fontName, style);
            currentY += singleLineHeight;
        }
        return true;
    }

    bool textCentered(Buffer& buffer, float centerX, float centerY, const std::string& text,
        int fontSize, const Color& color, const FontStyle& style, const std::string& fontName) {
        return textCentered(buffer, centerX, centerY, internal::to_wstring_auto(text),
            fontSize, color, style, internal::to_wstring_auto(fontName));
    }

} // namespace pa2d