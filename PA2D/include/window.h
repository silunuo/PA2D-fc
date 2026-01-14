#pragma once
#define WIN32_LEAN_AND_MEAN
#ifdef _WIN32
#define NOMINMAX  // 避免max,min宏冲突
#include <windows.h>
#endif
#include <future>
#include"geometry/point.h"
namespace pa2d {

    struct Buffer;
    class Canvas;
    struct KeyEvent {
        int key;
        bool pressed;
    };

    struct MouseEvent {
        int x, y;
        int button;       // 0=左键, 1=右键, 2=中键, -1=滚轮, -2=移动, -3=离开
        bool pressed;     // 对于按键按下/释放
        int wheelDelta;  // 滚轮增量
    };
    // 事件回调
    using KeyCallback = std::function<void(const KeyEvent&)>;
    using MouseCallback = std::function<void(const MouseEvent&)>;
    using ResizeCallback = std::function<void(int, int)>;
    using CharCallback = std::function<void(wchar_t)>;
    using FocusCallback = std::function<void(bool)>;
    using CloseCallback = std::function<bool()>;
    using MenuCallback = std::function<void(int)>;
    using FileListCallback = std::function<void(const std::vector<std::string>&)>;

    class Window {
    public:
        using HWND = void*;
        using HMENU = void*;
        using HICON = void*;
        using HCURSOR = void*;
        using COLORREF = unsigned long;
        // ==================== 常用功能 ====================
        // 构造/析构
        Window(int width = -1, int height = -1, const char* title ="PA2D Window");
        Window(const Window& other);
        Window(Window&& other) noexcept;
        Window& operator=(const Window& other);
        Window& operator=(Window&&)noexcept;
        ~Window();


        HWND getHandle() const;
        // 窗口状态查询
        int width() const;
        int height() const;
        bool isOpen() const;
        bool isClosed() const;
        bool isVisible() const;
        bool isMaximized() const;
        bool isMinimized() const;
        bool isFullscreen() const;

        // 窗口操作
        Window& show();
        Window& hide();
        Window& close();
        Window& setVisible(bool visible);
        Window& focus();

        // 堵塞等待
        void waitForClose();

        // 窗口大小和位置
        Window& setPosition(int x, int y);
        Window& setClientSize(int width, int height);
        Window& setWindowSize(int width, int height);
        PointInt getPosition() const;
        Size getClientSize() const;
        Size getWindowSize() const;

        // 渲染功能
        Window& render(const Canvas& canvas, int destX = 0, int destY = 0, int srcX = 0, int srcY = 0, int width = -1, int height = -1, bool clearBackground = true, COLORREF bgColor = 0);
        Window& renderCentered(const Canvas& canvas, bool clearBackground = true, COLORREF bgColor = 0);
        Window& render(const Buffer& buffer, int destX = 0, int destY = 0, int srcX = 0, int srcY = 0, int width = -1, int height = -1, bool clearBackground = true, COLORREF bgColor = 0);
        Window& renderCentered(const Buffer& buffer, bool clearBackground = true, COLORREF bgColor = 0);

        // ==================== 事件回调 ====================
        Window& onKey(KeyCallback cb);
        Window& onMouse(MouseCallback cb);
        Window& onResize(ResizeCallback cb);
        Window& onClose(CloseCallback cb);
        Window& onChar(CharCallback cb);
        Window& onFocus(FocusCallback cb);
        Window& onMenu(MenuCallback cb);
        Window& onFileDrop(FileListCallback cb);
        Window& onClipboardFiles(FileListCallback cb);
        Window& disableClipboardFiles();

        // ==================== 输入状态 ====================
        Point getMousePosition() const;
        bool isMouseInWindow() const;
        bool isMouseButtonPressed(int button) const;
        bool isKeyPressed(int vkCode) const;
        bool isShiftPressed() const;
        bool isCtrlPressed() const;
        bool isAltPressed() const;

        // ==================== 高级窗口控制 ====================
        Window& maximize();
        Window& minimize();
        Window& restore();
        Window& flash(bool flashTitleBar = true);
        Window& setResizable(bool resizable);
        Window& setAlwaysOnTop(bool onTop);
        Window& setBorderless(bool borderless);
        Window& setTitlebarless(bool titlebarless);
        Window& setFullscreen(bool fullscreen);
        Window& setMinSize(int minWidth, int minHeight);
        Window& setMaxSize(int maxWidth, int maxHeight);
        Window& setMinimizeButton(bool show);
        Window& setMaximizeButton(bool show);
        Window& setCloseButton(bool show);

        // ==================== 外观设置 ====================
        Window& setTitle(const char* title);
        Window& setTitle(const wchar_t* title);
        std::string getTitle() const;
        Window& setCursor(HCURSOR cursor);
        Window& setCursorDefault();
        Window& setCursorWait();
        Window& setCursorCross();
        Window& setCursorHand();
        Window& setCursorText();
        Window& setCursorVisibility(bool visible);
        Window& setCursorPosition(int x, int y);
        Window& setIcon(HICON icon);
        Window& setIconFromResource(int resourceId);

        // ==================== 输入控制 ====================
        Window& setMouseCapture(bool capture);

        // ==================== 剪贴板和文件 ====================
        bool setClipboardText(const std::string& text);
        std::string getClipboardText();
        bool setClipboardText(const std::wstring& text);
        std::wstring getClipboardTextW();
        bool hasClipboardText() const;
        std::vector<std::string> getClipboardFiles();
        bool hasClipboardFiles() const;
        Window& enableFileDrop(bool enable = true);

        // ==================== 菜单功能 ====================
        Window& setMenu(HMENU menu);
        HMENU createMenu();
        HMENU createPopupMenu();
        Window& appendMenuItem(HMENU menu, const char* text, int id, bool enabled = true);
        Window& appendMenuSeparator(HMENU menu);
        Window& appendMenuPopup(HMENU menu, const char* text, HMENU popupMenu);
        Window& destroyMenu(HMENU menu);
        // 静态Window Proc
        static LRESULT CALLBACK windowProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam);
    private:
        struct Impl;
        Impl* pImpl_;
        // 内部方法
        void startMessageThread(int width, int height, const std::string& title);
        void stopMessageThread();
        void handleMouse(UINT msg, WPARAM wparam, LPARAM lparam);
        void initializeRender();
        void updateWindowStyles();

        // 静态注册函数
        static void registerWindowClassOnce();
};
// ==================== 系统信息 ====================
Point getGlobalMousePosition();
Point getScreenSize();
Point getWorkAreaSize();
double getDpiScale();
} // namespace pa2d