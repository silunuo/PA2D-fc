#include "../include/window.h"
#include "../include/canvas.h"
#include <windowsx.h>
#include <shellapi.h> 

namespace pa2d {
#define REAL_HWND(h) (reinterpret_cast<::HWND>(h))
#define REAL_HMENU(m) (reinterpret_cast<::HMENU>(m))
#define REAL_HDC(dc) (reinterpret_cast<::HDC>(dc))
#define REAL_HICON(i) (reinterpret_cast<::HICON>(i))
#define REAL_HCURSOR(c) (reinterpret_cast<::HCURSOR>(c))
#define REAL_HDROP(d) (reinterpret_cast<::HDROP>(d))

    struct Window::Impl {
        HWND hwnd_ = nullptr;
        std::thread messageThread_;
        std::atomic<bool> running_{ false };
        DWORD threadId_ = 0;
        std::promise<bool> initPromise_;
        std::condition_variable closeCv_;
        std::mutex closeMutex_;
        bool windowClosed_ = false;

        LONG borderlessOriginalStyle_ = 0;
        LONG borderlessOriginalExStyle_ = 0;
        int borderlessOriginalClientWidth_ = 0;
        int borderlessOriginalClientHeight_ = 0;
        bool borderlessStateSaved_ = false;
        bool isBorderless_ = false;

        bool isTitlebarless_ = false;
        bool titlebarlessStateSaved_ = false;
        LONG titlebarlessOriginalStyle_ = 0;
        LONG titlebarlessOriginalExStyle_ = 0;

        LONG fullscreenOriginalStyle_ = 0;
        LONG fullscreenOriginalExStyle_ = 0;
        RECT fullscreenOriginalRect_ = { 0 };
        bool fullscreenStateSaved_ = false;
        bool isFullscreen_ = false;

        int minWidth_ = 0;
        int minHeight_ = 0;
        int maxWidth_ = 0;
        int maxHeight_ = 0;
        std::atomic<bool> trackingMouse_{ false };

        struct DirectBuffer {
            BITMAPINFO bmi = {};
        } directBuffer_;

        int lastClientWidth_ = 0;
        int lastClientHeight_ = 0;
        int lastBufferWidth_ = 0;
        int lastBufferHeight_ = 0;
        bool sizeChanged_ = true;

        KeyCallback keyCb_;
        MouseCallback mouseCb_;
        ResizeCallback resizeCb_;
        CharCallback charCb_;
        FocusCallback focusCb_;
        CloseCallback closeCb_;
        MenuCallback menuCb_;
        FileListCallback dropCb_;
        FileListCallback clipboardFilesCb_;

        HWND nextClipboardViewer_ = nullptr;
        HDC persistentDC_ = nullptr;
    };

    inline static LRESULT CALLBACK RealWindowProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
        return Window::windowProc(reinterpret_cast<::HWND>(hwnd), msg, wparam, lparam);
    }

    void Window::registerWindowClassOnce() {
        static std::once_flag onceFlag;
        std::call_once(onceFlag, []() {
            WNDCLASSEX wc = { sizeof(WNDCLASSEX) };
            wc.style = CS_HREDRAW | CS_VREDRAW;
            wc.lpfnWndProc = RealWindowProc;
            wc.hInstance = GetModuleHandle(nullptr);
            wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
            wc.hbrBackground = nullptr;
            wc.lpszClassName = "PA2D";
            RegisterClassEx(&wc);
            });
    }

    Window::Window(int width, int height, const char* title)
        : pImpl_(new Impl) {
        std::future<bool> initFuture = pImpl_->initPromise_.get_future();
        startMessageThread(width, height, std::string(title ? title : ""));
        bool success = initFuture.get();
        if (!success) {
            pImpl_->running_ = false;
        }
    }

    Window::Window(const Window& other)
        : pImpl_(new Impl()) {
        pImpl_->minWidth_ = other.pImpl_->minWidth_;
        pImpl_->minHeight_ = other.pImpl_->minHeight_;
        pImpl_->maxWidth_ = other.pImpl_->maxWidth_;
        pImpl_->maxHeight_ = other.pImpl_->maxHeight_;
        pImpl_->isBorderless_ = other.pImpl_->isBorderless_;
        pImpl_->isFullscreen_ = other.pImpl_->isFullscreen_;
        pImpl_->keyCb_ = other.pImpl_->keyCb_;
        pImpl_->mouseCb_ = other.pImpl_->mouseCb_;
        pImpl_->resizeCb_ = other.pImpl_->resizeCb_;
        pImpl_->charCb_ = other.pImpl_->charCb_;
        pImpl_->focusCb_ = other.pImpl_->focusCb_;
        pImpl_->closeCb_ = other.pImpl_->closeCb_;
        pImpl_->menuCb_ = other.pImpl_->menuCb_;
        pImpl_->dropCb_ = other.pImpl_->dropCb_;
        pImpl_->clipboardFilesCb_ = other.pImpl_->clipboardFilesCb_;

        PointInt source = other.getPosition();
        std::future<bool> initFuture = pImpl_->initPromise_.get_future();
        startMessageThread(other.width(), other.height(), other.getTitle());
        bool success = initFuture.get();

        if (success) {
            setPosition(source.x, source.y);
            if (pImpl_->isBorderless_) {
                setBorderless(true);
            }
            if (pImpl_->isFullscreen_) {
                setFullscreen(true);
            }

            LONG sourceStyle = GetWindowLong(REAL_HWND(other.pImpl_->hwnd_), GWL_STYLE);
            if (!(sourceStyle & WS_MINIMIZEBOX)) {
                setMinimizeButton(false);
            }
            if (!(sourceStyle & WS_MAXIMIZEBOX)) {
                setMaximizeButton(false);
            }
            if (!(sourceStyle & WS_SYSMENU)) {
                setCloseButton(false);
            }

            bool sourceResizable = (sourceStyle & WS_THICKFRAME) != 0;
            setResizable(sourceResizable);

            if (GetWindowLong(REAL_HWND(other.pImpl_->hwnd_), GWL_EXSTYLE) & WS_EX_TOPMOST) {
                setAlwaysOnTop(true);
            }
            if (other.isVisible()) show();
        }
        else {
            pImpl_->running_ = false;
        }
    }

    Window& Window::operator=(const Window& other) {
        if (this != &other) {
            if (pImpl_) {
                pImpl_->running_ = false;
                if (pImpl_->messageThread_.joinable()) {
                    if (pImpl_->threadId_ != 0) {
                        PostThreadMessage(pImpl_->threadId_, WM_QUIT, 0, 0);
                    }
                    pImpl_->messageThread_.join();
                }

                if (pImpl_->hwnd_) {
                    if (pImpl_->nextClipboardViewer_) {
                        ChangeClipboardChain(REAL_HWND(pImpl_->hwnd_), REAL_HWND(pImpl_->nextClipboardViewer_));
                        pImpl_->nextClipboardViewer_ = nullptr;
                    }

                    if (pImpl_->persistentDC_) {
                        ReleaseDC(REAL_HWND(pImpl_->hwnd_), REAL_HDC(pImpl_->persistentDC_));
                        pImpl_->persistentDC_ = nullptr;
                    }

                    DestroyWindow(REAL_HWND(pImpl_->hwnd_));
                    pImpl_->hwnd_ = nullptr;
                }

                delete pImpl_;
            }

            pImpl_ = new Impl();
            pImpl_->minWidth_ = other.pImpl_->minWidth_;
            pImpl_->minHeight_ = other.pImpl_->minHeight_;
            pImpl_->maxWidth_ = other.pImpl_->maxWidth_;
            pImpl_->maxHeight_ = other.pImpl_->maxHeight_;
            pImpl_->isBorderless_ = other.pImpl_->isBorderless_;
            pImpl_->isFullscreen_ = other.pImpl_->isFullscreen_;
            pImpl_->keyCb_ = other.pImpl_->keyCb_;
            pImpl_->mouseCb_ = other.pImpl_->mouseCb_;
            pImpl_->resizeCb_ = other.pImpl_->resizeCb_;
            pImpl_->charCb_ = other.pImpl_->charCb_;
            pImpl_->focusCb_ = other.pImpl_->focusCb_;
            pImpl_->closeCb_ = other.pImpl_->closeCb_;
            pImpl_->menuCb_ = other.pImpl_->menuCb_;
            pImpl_->dropCb_ = other.pImpl_->dropCb_;
            pImpl_->clipboardFilesCb_ = other.pImpl_->clipboardFilesCb_;

            PointInt source = other.getPosition();
            std::future<bool> initFuture = pImpl_->initPromise_.get_future();
            startMessageThread(other.width(), other.height(), other.getTitle());
            bool success = initFuture.get();

            if (success) {
                setPosition(source.x, source.y);
                if (pImpl_->isBorderless_) {
                    setBorderless(true);
                }
                if (pImpl_->isFullscreen_) {
                    setFullscreen(true);
                }

                LONG sourceStyle = GetWindowLong(REAL_HWND(other.pImpl_->hwnd_), GWL_STYLE);
                if (!(sourceStyle & WS_MINIMIZEBOX)) {
                    setMinimizeButton(false);
                }
                if (!(sourceStyle & WS_MAXIMIZEBOX)) {
                    setMaximizeButton(false);
                }
                if (!(sourceStyle & WS_SYSMENU)) {
                    setCloseButton(false);
                }

                bool sourceResizable = (sourceStyle & WS_THICKFRAME) != 0;
                setResizable(sourceResizable);

                if (GetWindowLong(REAL_HWND(other.pImpl_->hwnd_), GWL_EXSTYLE) & WS_EX_TOPMOST) {
                    setAlwaysOnTop(true);
                }
                if (other.isVisible()) show();
            }
            else {
                pImpl_->running_ = false;
            }
        }
        return *this;
    }

    Window::Window(Window&& other) noexcept : pImpl_(std::exchange(other.pImpl_, nullptr)) {}

    Window& Window::operator=(Window&& other) noexcept {
        if (this != &other) {
            if (pImpl_) {
                if (pImpl_->hwnd_ && pImpl_->nextClipboardViewer_) {
                    ChangeClipboardChain(REAL_HWND(pImpl_->hwnd_), REAL_HWND(pImpl_->nextClipboardViewer_));
                    pImpl_->nextClipboardViewer_ = nullptr;
                }
                if (pImpl_->persistentDC_) {
                    ReleaseDC(REAL_HWND(pImpl_->hwnd_), REAL_HDC(pImpl_->persistentDC_));
                    pImpl_->persistentDC_ = nullptr;
                }
                stopMessageThread();
                delete pImpl_;
            }
            pImpl_ = std::exchange(other.pImpl_, nullptr);
        }
        return *this;
    }

    Window::~Window() {
        if (!pImpl_) return;
        if (pImpl_->hwnd_ && pImpl_->nextClipboardViewer_) {
            ChangeClipboardChain(REAL_HWND(pImpl_->hwnd_), REAL_HWND(pImpl_->nextClipboardViewer_));
            pImpl_->nextClipboardViewer_ = nullptr;
        }

        if (pImpl_->persistentDC_) {
            ReleaseDC(REAL_HWND(pImpl_->hwnd_), REAL_HDC(pImpl_->persistentDC_));
            pImpl_->persistentDC_ = nullptr;
        }

        stopMessageThread();
        delete pImpl_;
    }

    pa2d::Window::HWND Window::getHandle() const {
        return REAL_HWND(pImpl_->hwnd_);
    }

    int Window::width() const {
        if (!pImpl_->hwnd_) return 0;
        RECT rect;
        GetClientRect(REAL_HWND(pImpl_->hwnd_), &rect);
        return rect.right - rect.left;
    }

    int Window::height() const {
        if (!pImpl_->hwnd_) return 0;
        RECT rect;
        GetClientRect(REAL_HWND(pImpl_->hwnd_), &rect);
        return rect.bottom - rect.top;
    }

    Window& Window::onKey(KeyCallback cb) {
        pImpl_->keyCb_ = cb;
        return *this;
    }

    Window& Window::onMouse(MouseCallback cb) {
        pImpl_->mouseCb_ = cb;
        return *this;
    }

    Window& Window::onResize(ResizeCallback cb) {
        pImpl_->resizeCb_ = cb;
        return *this;
    }

    Window& Window::onChar(CharCallback cb) {
        pImpl_->charCb_ = cb;
        return *this;
    }

    Window& Window::onFocus(FocusCallback cb) {
        pImpl_->focusCb_ = cb;
        return *this;
    }

    Window& Window::onClose(CloseCallback cb) {
        pImpl_->closeCb_ = cb;
        return *this;
    }

    Window& Window::onMenu(MenuCallback cb) {
        pImpl_->menuCb_ = cb;
        return *this;
    }

    void Window::waitForClose() {
        if (!pImpl_->hwnd_ || !pImpl_->running_) return;
        std::unique_lock<std::mutex> lock(pImpl_->closeMutex_);
        pImpl_->closeCv_.wait(lock, [this]() {
            return pImpl_->windowClosed_ || !pImpl_->running_;
            });
    }

    Window& Window::close() {
        if (pImpl_->hwnd_) {
            PostMessage(REAL_HWND(pImpl_->hwnd_), WM_CLOSE, 0, 0);
        }
        return *this;
    }

    bool Window::isOpen() const {
        if (!pImpl_ || !pImpl_->hwnd_) {
            return false;
        }

        if (!IsWindow(REAL_HWND(pImpl_->hwnd_))) {
            return false;
        }

        if (!pImpl_->running_) {
            return false;
        }

        std::lock_guard<std::mutex> lock(pImpl_->closeMutex_);
        return !pImpl_->windowClosed_;
    }

    bool Window::isClosed() const {
        return !isOpen();
    }

    bool Window::isVisible() const { return pImpl_->hwnd_ && IsWindowVisible(REAL_HWND(pImpl_->hwnd_)); }

    bool Window::isMaximized() const { return pImpl_->hwnd_ && IsZoomed(REAL_HWND(pImpl_->hwnd_)); }

    bool Window::isMinimized() const { return pImpl_->hwnd_ && IsIconic(REAL_HWND(pImpl_->hwnd_)); }

    bool Window::isFullscreen() const { return pImpl_->isFullscreen_; }

    Window& Window::setMinSize(int minWidth, int minHeight) {
        if (!pImpl_->hwnd_) return *this;
        pImpl_->minWidth_ = minWidth;
        pImpl_->minHeight_ = minHeight;
        return *this;
    }

    Window& Window::setMaxSize(int maxWidth, int maxHeight) {
        if (!pImpl_->hwnd_) return *this;
        pImpl_->maxWidth_ = maxWidth;
        pImpl_->maxHeight_ = maxHeight;
        return *this;
    }

    Window& Window::setResizable(bool resizable) {
        if (!pImpl_->hwnd_) return *this;

        LONG style = GetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_STYLE);
        if (resizable)
            style |= (WS_THICKFRAME | WS_MAXIMIZEBOX);
        else
            style &= ~(WS_THICKFRAME | WS_MAXIMIZEBOX);

        SetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_STYLE, style);
        return *this;
    }

    Window& Window::setAlwaysOnTop(bool onTop) {
        if (!pImpl_->hwnd_) return *this;

        SetWindowPos((::HWND)pImpl_->hwnd_,
            onTop ? (::HWND)HWND_TOPMOST : (::HWND)HWND_NOTOPMOST,
            0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
        return *this;
    }

    Window& Window::setBorderless(bool borderless) {
        if (!pImpl_->hwnd_ || pImpl_->isBorderless_ == borderless) return *this;

        if (borderless) {
            if (!pImpl_->borderlessStateSaved_) {
                pImpl_->borderlessOriginalStyle_ = GetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_STYLE);
                pImpl_->borderlessOriginalExStyle_ = GetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_EXSTYLE);

                RECT clientRect;
                GetClientRect(REAL_HWND(pImpl_->hwnd_), &clientRect);
                pImpl_->borderlessOriginalClientWidth_ = clientRect.right;
                pImpl_->borderlessOriginalClientHeight_ = clientRect.bottom;

                pImpl_->borderlessStateSaved_ = true;
            }

            SetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_STYLE, WS_POPUP | WS_VISIBLE);
            SetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_EXSTYLE, WS_EX_APPWINDOW);

            RECT windowRect;
            GetWindowRect(REAL_HWND(pImpl_->hwnd_), &windowRect);
            SetWindowPos(REAL_HWND(pImpl_->hwnd_), nullptr,
                windowRect.left, windowRect.top,
                pImpl_->borderlessOriginalClientWidth_, pImpl_->borderlessOriginalClientHeight_,
                SWP_FRAMECHANGED | SWP_NOZORDER);
        }
        else {
            SetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_STYLE, pImpl_->borderlessOriginalStyle_);
            SetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_EXSTYLE, pImpl_->borderlessOriginalExStyle_);

            RECT desiredClientRect = { 0, 0, pImpl_->borderlessOriginalClientWidth_, pImpl_->borderlessOriginalClientHeight_ };
            AdjustWindowRect(&desiredClientRect, pImpl_->borderlessOriginalStyle_, FALSE);

            RECT windowRect;
            GetWindowRect(REAL_HWND(pImpl_->hwnd_), &windowRect);
            SetWindowPos(REAL_HWND(pImpl_->hwnd_), nullptr,
                windowRect.left, windowRect.top,
                desiredClientRect.right - desiredClientRect.left,
                desiredClientRect.bottom - desiredClientRect.top,
                SWP_FRAMECHANGED | SWP_NOZORDER);

            SetForegroundWindow(REAL_HWND(pImpl_->hwnd_));
            SetFocus(REAL_HWND(pImpl_->hwnd_));
        }

        pImpl_->isBorderless_ = borderless;
        return *this;
    }

    Window& Window::setTitlebarless(bool titlebarless) {
        if (!pImpl_->hwnd_ || pImpl_->isTitlebarless_ == titlebarless) return *this;

        if (titlebarless) {
            if (!pImpl_->titlebarlessStateSaved_) {
                pImpl_->titlebarlessOriginalStyle_ = GetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_STYLE);
                pImpl_->titlebarlessOriginalExStyle_ = GetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_EXSTYLE);
                pImpl_->titlebarlessStateSaved_ = true;
            }

            LONG style = pImpl_->titlebarlessOriginalStyle_;
            style &= ~WS_CAPTION;
            style |= WS_THICKFRAME;
            style |= WS_SYSMENU;

            SetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_STYLE, style);

            RECT windowRect;
            GetWindowRect(REAL_HWND(pImpl_->hwnd_), &windowRect);
            SetWindowPos(REAL_HWND(pImpl_->hwnd_), nullptr, 0, 0, 0, 0,
                SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER |
                SWP_NOACTIVATE | SWP_NOOWNERZORDER);
        }
        else {
            SetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_STYLE, pImpl_->titlebarlessOriginalStyle_);
            SetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_EXSTYLE, pImpl_->titlebarlessOriginalExStyle_);

            SetWindowPos(REAL_HWND(pImpl_->hwnd_), nullptr, 0, 0, 0, 0,
                SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER |
                SWP_NOACTIVATE | SWP_NOOWNERZORDER);
        }

        pImpl_->isTitlebarless_ = titlebarless;
        return *this;
    }

    Window& Window::setFullscreen(bool fullscreen) {
        if (!pImpl_->hwnd_ || pImpl_->isFullscreen_ == fullscreen) return *this;

        if (fullscreen) {
            if (!pImpl_->fullscreenStateSaved_) {
                pImpl_->fullscreenOriginalStyle_ = GetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_STYLE);
                pImpl_->fullscreenOriginalExStyle_ = GetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_EXSTYLE);
                GetWindowRect(REAL_HWND(pImpl_->hwnd_), &pImpl_->fullscreenOriginalRect_);
                pImpl_->fullscreenStateSaved_ = true;
            }

            HMONITOR hMonitor = MonitorFromWindow(REAL_HWND(pImpl_->hwnd_), MONITOR_DEFAULTTONEAREST);
            MONITORINFO monitorInfo = { sizeof(monitorInfo) };
            GetMonitorInfo(hMonitor, &monitorInfo);

            SetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_STYLE, WS_POPUP | WS_VISIBLE);
            SetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_EXSTYLE, WS_EX_APPWINDOW);

            SetWindowPos(REAL_HWND(pImpl_->hwnd_),
                reinterpret_cast<::HWND>(HWND_TOP),
                monitorInfo.rcMonitor.left, monitorInfo.rcMonitor.top,
                monitorInfo.rcMonitor.right - monitorInfo.rcMonitor.left,
                monitorInfo.rcMonitor.bottom - monitorInfo.rcMonitor.top,
                SWP_FRAMECHANGED | SWP_NOACTIVATE);
        }
        else {
            SetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_STYLE, pImpl_->fullscreenOriginalStyle_);
            SetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_EXSTYLE, pImpl_->fullscreenOriginalExStyle_);

            SetWindowPos(REAL_HWND(pImpl_->hwnd_), nullptr,
                pImpl_->fullscreenOriginalRect_.left, pImpl_->fullscreenOriginalRect_.top,
                pImpl_->fullscreenOriginalRect_.right - pImpl_->fullscreenOriginalRect_.left,
                pImpl_->fullscreenOriginalRect_.bottom - pImpl_->fullscreenOriginalRect_.top,
                SWP_FRAMECHANGED | SWP_NOACTIVATE | SWP_NOZORDER);
        }

        pImpl_->isFullscreen_ = fullscreen;
        return *this;
    }

    Window& Window::setMinimizeButton(bool show) {
        if (!pImpl_->hwnd_) return *this;

        LONG style = GetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_STYLE);
        if (show)
            style |= WS_MINIMIZEBOX;
        else
            style &= ~WS_MINIMIZEBOX;

        SetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_STYLE, style);
        updateWindowStyles();
        return *this;
    }

    Window& Window::setMaximizeButton(bool show) {
        if (!pImpl_->hwnd_) return *this;

        LONG style = GetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_STYLE);
        if (show)
            style |= WS_MAXIMIZEBOX;
        else
            style &= ~WS_MAXIMIZEBOX;

        SetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_STYLE, style);
        updateWindowStyles();
        return *this;
    }

    Window& Window::setCloseButton(bool show) {
        if (!pImpl_->hwnd_) return *this;

        LONG style = GetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_STYLE);

        if (show) {
            style |= WS_SYSMENU;
        }
        else {
            style &= ~WS_SYSMENU;
        }

        SetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_STYLE, style);
        updateWindowStyles();
        return *this;
    }

    Point Window::getMousePosition() const {
        POINT pt;
        if (GetCursorPos(&pt) && ScreenToClient(REAL_HWND(pImpl_->hwnd_), &pt)) {
            return { (float)pt.x, (float)pt.y };
        }
        return { -1, -1 };
    }

    bool Window::isMouseInWindow() const {
        POINT pt;
        if (!GetCursorPos(&pt)) return false;

        RECT rect;
        GetWindowRect(REAL_HWND(pImpl_->hwnd_), &rect);
        return PtInRect(&rect, pt);
    }

    Point getGlobalMousePosition() {
        POINT pt;
        GetCursorPos(&pt);
        return { (float)pt.x, (float)pt.y };
    }

    bool Window::isMouseButtonPressed(int button) const {
        static const int vkButtons[] = { VK_LBUTTON, VK_RBUTTON, VK_MBUTTON };
        return (button >= 0 && button < 3) ?
            (GetAsyncKeyState(vkButtons[button]) & 0x8000) : false;
    }

    Window& Window::setMouseCapture(bool capture) {
        if (pImpl_->hwnd_) {
            if (capture) SetCapture(REAL_HWND(pImpl_->hwnd_));
            else ReleaseCapture();
        }
        return *this;
    }

    Window& Window::setCursorVisibility(bool visible) {
        ShowCursor(visible);
        return *this;
    }

    Window& Window::setCursorPosition(int x, int y) {
        POINT pt = { x, y };
        ClientToScreen(REAL_HWND(pImpl_->hwnd_), &pt);
        SetCursorPos(pt.x, pt.y);
        return *this;
    }

    bool Window::isKeyPressed(int vkCode) const { return GetAsyncKeyState(vkCode) & 0x8000; }

    bool Window::isShiftPressed() const { return isKeyPressed(VK_SHIFT); }

    bool Window::isCtrlPressed() const { return isKeyPressed(VK_CONTROL); }

    bool Window::isAltPressed() const { return isKeyPressed(VK_MENU); }

    bool Window::setClipboardText(const std::string& text) {
        if (!pImpl_->hwnd_ || text.empty()) return false;
        if (!OpenClipboard(REAL_HWND(pImpl_->hwnd_))) return false;

        bool success = false;
        EmptyClipboard();

        TextEncoding encoding = textEncoding();

        if (encoding == TextEncoding::ANSI) {
            size_t bufferSize = text.size() + 1;
            HGLOBAL hMem = GlobalAlloc(GMEM_MOVEABLE, bufferSize);
            if (hMem) {
                char* ptr = static_cast<char*>(GlobalLock(hMem));
                if (ptr) {
                    memcpy(ptr, text.c_str(), bufferSize);
                    GlobalUnlock(hMem);
                }
                success = (SetClipboardData(CF_TEXT, hMem) != nullptr);
            }
        }
        else if (encoding == TextEncoding::UTF8) {
            const char* srcText = text.c_str();
            int wideLen = 0;

            if (encoding == TextEncoding::UTF8) {
                wideLen = MultiByteToWideChar(CP_UTF8, 0, srcText, -1, nullptr, 0);
            }
            else {
                wideLen = MultiByteToWideChar(CP_ACP, 0, srcText, -1, nullptr, 0);
            }

            if (wideLen > 0) {
                size_t bufferSize = wideLen * sizeof(wchar_t);
                HGLOBAL hMem = GlobalAlloc(GMEM_MOVEABLE, bufferSize);
                if (hMem) {
                    wchar_t* ptr = static_cast<wchar_t*>(GlobalLock(hMem));
                    if (ptr) {
                        if (encoding == TextEncoding::UTF8) {
                            MultiByteToWideChar(CP_UTF8, 0, srcText, -1, ptr, wideLen);
                        }
                        else {
                            MultiByteToWideChar(CP_ACP, 0, srcText, -1, ptr, wideLen);
                        }
                        GlobalUnlock(hMem);
                    }
                    success = (SetClipboardData(CF_UNICODETEXT, hMem) != nullptr);
                }
            }
        }

        CloseClipboard();
        return success;
    }

    std::string Window::getClipboardText() {
        if (!pImpl_->hwnd_) return "";
        if (!OpenClipboard(REAL_HWND(pImpl_->hwnd_))) return "";

        std::string result;
        TextEncoding encoding = textEncoding();

        if (IsClipboardFormatAvailable(CF_UNICODETEXT)) {
            HANDLE hData = GetClipboardData(CF_UNICODETEXT);
            if (hData) {
                wchar_t* wideText = static_cast<wchar_t*>(GlobalLock(hData));
                if (wideText) {
                    int ansiLen = 0;
                    UINT codePage = CP_ACP;
                    if (encoding == TextEncoding::UTF8) {
                        codePage = CP_UTF8;
                    }

                    ansiLen = WideCharToMultiByte(codePage, 0, wideText, -1, nullptr, 0, nullptr, nullptr);
                    if (ansiLen > 0) {
                        result.resize(ansiLen);
                        WideCharToMultiByte(codePage, 0, wideText, -1, &result[0], ansiLen, nullptr, nullptr);
                        if (!result.empty() && result.back() == '\0') {
                            result.pop_back();
                        }
                    }
                    GlobalUnlock(hData);
                }
            }
        }
        else if (IsClipboardFormatAvailable(CF_TEXT)) {
            HANDLE hData = GetClipboardData(CF_TEXT);
            if (hData) {
                char* ansiText = static_cast<char*>(GlobalLock(hData));
                if (ansiText) {
                    result = ansiText;
                    GlobalUnlock(hData);
                }
            }
        }

        CloseClipboard();
        return result;
    }

    bool Window::setClipboardText(const std::wstring& text) {
        if (!pImpl_->hwnd_ || text.empty()) return false;
        if (!OpenClipboard(REAL_HWND(pImpl_->hwnd_))) return false;

        bool success = false;
        EmptyClipboard();

        size_t bufferSize = (text.size() + 1) * sizeof(wchar_t);
        HGLOBAL hMem = GlobalAlloc(GMEM_MOVEABLE, bufferSize);
        if (hMem) {
            wchar_t* ptr = static_cast<wchar_t*>(GlobalLock(hMem));
            if (ptr) {
                memcpy(ptr, text.c_str(), bufferSize);
                GlobalUnlock(hMem);
            }
            success = (SetClipboardData(CF_UNICODETEXT, hMem) != nullptr);
        }

        CloseClipboard();
        return success;
    }

    std::wstring Window::getClipboardTextW() {
        if (!pImpl_->hwnd_) return L"";
        if (!OpenClipboard(REAL_HWND(pImpl_->hwnd_))) return L"";

        std::wstring result;

        if (IsClipboardFormatAvailable(CF_UNICODETEXT)) {
            HANDLE hData = GetClipboardData(CF_UNICODETEXT);
            if (hData) {
                wchar_t* wideText = static_cast<wchar_t*>(GlobalLock(hData));
                if (wideText) {
                    result = wideText;
                    GlobalUnlock(hData);
                }
            }
        }
        else if (IsClipboardFormatAvailable(CF_TEXT)) {
            HANDLE hData = GetClipboardData(CF_TEXT);
            if (hData) {
                char* ansiText = static_cast<char*>(GlobalLock(hData));
                if (ansiText) {
                    int wideLen = MultiByteToWideChar(CP_ACP, 0, ansiText, -1, nullptr, 0);
                    if (wideLen > 0) {
                        result.resize(wideLen);
                        MultiByteToWideChar(CP_ACP, 0, ansiText, -1, &result[0], wideLen);
                        if (!result.empty() && result.back() == L'\0') {
                            result.pop_back();
                        }
                    }
                    GlobalUnlock(hData);
                }
            }
        }

        CloseClipboard();
        return result;
    }

    bool Window::hasClipboardText() const {
        if (!pImpl_->hwnd_) return false;
        return IsClipboardFormatAvailable(CF_TEXT) ||
            IsClipboardFormatAvailable(CF_UNICODETEXT);
    }

    Window& Window::enableFileDrop(bool enable) {
        if (pImpl_->hwnd_) {
            DragAcceptFiles(REAL_HWND(pImpl_->hwnd_), enable);
        }
        return *this;
    }

    bool Window::hasClipboardFiles() const {
        if (!pImpl_->hwnd_) return false;
        return IsClipboardFormatAvailable(CF_HDROP);
    }

    std::vector<std::string> Window::getClipboardFiles() {
        std::vector<std::string> files;

        if (!pImpl_->hwnd_ || !OpenClipboard(REAL_HWND(pImpl_->hwnd_))) return files;

        if (IsClipboardFormatAvailable(CF_HDROP)) {
            HDROP hDrop = (HDROP)GetClipboardData(CF_HDROP);
            if (hDrop) {
                UINT fileCount = DragQueryFile(hDrop, 0xFFFFFFFF, nullptr, 0);
                for (UINT i = 0; i < fileCount; i++) {
                    char filePath[MAX_PATH];
                    DragQueryFileA(hDrop, i, filePath, MAX_PATH);
                    files.push_back(filePath);
                }
            }
        }

        CloseClipboard();
        return files;
    }

    Window& Window::onFileDrop(FileListCallback cb) {
        pImpl_->dropCb_ = cb;
        enableFileDrop(true);
        return *this;
    }

    Window& Window::onClipboardFiles(FileListCallback cb) {
        pImpl_->clipboardFilesCb_ = cb;

        if (pImpl_->hwnd_ && cb) {
            pImpl_->nextClipboardViewer_ = SetClipboardViewer(REAL_HWND(pImpl_->hwnd_));
        }

        return *this;
    }

    Window& Window::disableClipboardFiles() {
        pImpl_->clipboardFilesCb_ = nullptr;

        if (pImpl_->hwnd_ && pImpl_->nextClipboardViewer_) {
            ChangeClipboardChain(REAL_HWND(pImpl_->hwnd_), REAL_HWND(pImpl_->nextClipboardViewer_));
            pImpl_->nextClipboardViewer_ = nullptr;
        }

        return *this;
    }

    Window& Window::show() {
        if (pImpl_->hwnd_) ShowWindow(REAL_HWND(pImpl_->hwnd_), SW_SHOW);
        return *this;
    }

    Window& Window::hide() {
        if (pImpl_->hwnd_) ShowWindow(REAL_HWND(pImpl_->hwnd_), SW_HIDE);
        return *this;
    }

    Window& Window::maximize() {
        if (pImpl_->hwnd_) ShowWindow(REAL_HWND(pImpl_->hwnd_), SW_MAXIMIZE);
        return *this;
    }

    Window& Window::minimize() {
        if (pImpl_->hwnd_) ShowWindow(REAL_HWND(pImpl_->hwnd_), SW_MINIMIZE);
        return *this;
    }

    Window& Window::restore() {
        if (pImpl_->hwnd_) ShowWindow(REAL_HWND(pImpl_->hwnd_), SW_RESTORE);
        return *this;
    }

    Window& Window::setVisible(bool visible) {
        if (pImpl_->hwnd_) ShowWindow(REAL_HWND(pImpl_->hwnd_), visible ? SW_SHOW : SW_HIDE);
        return *this;
    }

    Window& Window::focus() {
        if (pImpl_->hwnd_) {
            if (IsIconic(REAL_HWND(pImpl_->hwnd_))) {
                ShowWindow(REAL_HWND(pImpl_->hwnd_), SW_RESTORE);
            }
            SetForegroundWindow(REAL_HWND(pImpl_->hwnd_));
            SetFocus(REAL_HWND(pImpl_->hwnd_));
            SetActiveWindow(REAL_HWND(pImpl_->hwnd_));
        }
        return *this;
    }

    Window& Window::flash(bool flashTitleBar) {
        if (pImpl_->hwnd_) {
            FLASHWINFO flashInfo = { sizeof(FLASHWINFO) };
            flashInfo.hwnd = REAL_HWND(pImpl_->hwnd_);
            flashInfo.dwFlags = FLASHW_ALL | (flashTitleBar ? FLASHW_CAPTION : 0);
            flashInfo.uCount = 3;
            flashInfo.dwTimeout = 0;
            FlashWindowEx(&flashInfo);
        }
        return *this;
    }

    Window& Window::setPosition(int x, int y) {
        if (pImpl_->hwnd_) {
            SetWindowPos(REAL_HWND(pImpl_->hwnd_), nullptr, x, y, 0, 0, SWP_NOSIZE | SWP_NOZORDER);
        }
        return *this;
    }

    Window& Window::setClientSize(int width, int height) {
        if (pImpl_->hwnd_) {
            RECT rect = { 0, 0, width, height };
            BOOL hasMenu = (GetMenu(REAL_HWND(pImpl_->hwnd_)) != nullptr);
            AdjustWindowRect(&rect, GetWindowLong(REAL_HWND(pImpl_->hwnd_), GWL_STYLE), hasMenu);
            SetWindowPos(REAL_HWND(pImpl_->hwnd_), nullptr, 0, 0,
                rect.right - rect.left, rect.bottom - rect.top,
                SWP_NOMOVE | SWP_NOZORDER);
        }
        return *this;
    }

    Window& Window::setWindowSize(int width, int height) {
        if (pImpl_->hwnd_) {
            SetWindowPos(REAL_HWND(pImpl_->hwnd_), nullptr, 0, 0, width, height, SWP_NOMOVE | SWP_NOZORDER);
        }
        return *this;
    }

    PointInt Window::getPosition() const {
        RECT rect = { 0 };
        if (pImpl_->hwnd_ && GetWindowRect(REAL_HWND(pImpl_->hwnd_), &rect)) {
            return { rect.left, rect.top };
        }
        return { 0, 0 };
    }

    Size Window::getClientSize() const {
        RECT rect = { 0 };
        if (pImpl_->hwnd_ && GetClientRect(REAL_HWND(pImpl_->hwnd_), &rect)) {
            return { rect.right - rect.left, rect.bottom - rect.top };
        }
        return { 0, 0 };
    }

    Size Window::getWindowSize() const {
        RECT rect = { 0 };
        if (pImpl_->hwnd_ && GetWindowRect(REAL_HWND(pImpl_->hwnd_), &rect)) {
            return { rect.right - rect.left, rect.bottom - rect.top };
        }
        return { 0, 0 };
    }

    Window& Window::setTitle(const char* title) {
        if (pImpl_->hwnd_) SetWindowTextA(REAL_HWND(pImpl_->hwnd_), title);
        return *this;
    }

    Window& Window::setTitle(const wchar_t* title) {
        if (pImpl_->hwnd_) SetWindowTextW(REAL_HWND(pImpl_->hwnd_), title);
        return *this;
    }

    std::string Window::getTitle() const {
        if (pImpl_->hwnd_) {
            char buffer[256];
            GetWindowTextA(REAL_HWND(pImpl_->hwnd_), buffer, sizeof(buffer));
            return std::string(buffer);
        }
        return "";
    }

    Window& Window::setCursor(HCURSOR cursor) {
        if (pImpl_->hwnd_) {
            SetClassLongPtr(REAL_HWND(pImpl_->hwnd_), GCLP_HCURSOR, (LONG_PTR)cursor);
            SetCursor(REAL_HCURSOR(cursor));
        }
        return *this;
    }

    Window& Window::setCursorDefault() {
        return setCursor(LoadCursor(nullptr, IDC_ARROW));
    }

    Window& Window::setCursorWait() {
        return setCursor(LoadCursor(nullptr, IDC_WAIT));
    }

    Window& Window::setCursorCross() {
        return setCursor(LoadCursor(nullptr, IDC_CROSS));
    }

    Window& Window::setCursorHand() {
        return setCursor(LoadCursor(nullptr, IDC_HAND));
    }

    Window& Window::setCursorText() {
        return setCursor(LoadCursor(nullptr, IDC_IBEAM));
    }

    Window& Window::setIcon(HICON icon) {
        if (pImpl_->hwnd_) {
            SendMessage(REAL_HWND(pImpl_->hwnd_), WM_SETICON, ICON_SMALL, (LPARAM)icon);
            SendMessage(REAL_HWND(pImpl_->hwnd_), WM_SETICON, ICON_BIG, (LPARAM)icon);
        }
        return *this;
    }

    Window& Window::setIconFromResource(int resourceId) {
        HICON icon = LoadIcon(GetModuleHandle(nullptr), MAKEINTRESOURCE(resourceId));
        if (icon) setIcon(icon);
        return *this;
    }

    Window& Window::setMenu(HMENU menu) {
        if (pImpl_->hwnd_) {

            Size size = getClientSize();

            ::SetMenu(REAL_HWND(pImpl_->hwnd_), REAL_HMENU(menu));
            DrawMenuBar(REAL_HWND(pImpl_->hwnd_));

            setClientSize(size.width, size.height);
        }
        return *this;
    }

    pa2d::Window::HMENU Window::createMenu() {
        return reinterpret_cast<HMENU>(CreateMenu());
    }

    pa2d::Window::HMENU Window::createPopupMenu() {
        return reinterpret_cast<HMENU>(CreatePopupMenu());
    }

    Window& Window::appendMenuItem(HMENU menu, const char* text, int id, bool enabled) {
        AppendMenuA(REAL_HMENU(menu), MF_STRING | (enabled ? MF_ENABLED : MF_GRAYED), id, text);
        return *this;
    }

    Window& Window::appendMenuSeparator(HMENU menu) {
        AppendMenu(REAL_HMENU(menu), MF_SEPARATOR, 0, nullptr);
        return *this;
    }

    Window& Window::appendMenuPopup(HMENU menu, const char* text, HMENU popupMenu) {
        AppendMenuA(REAL_HMENU(menu), MF_POPUP | MF_STRING, (UINT_PTR)popupMenu, text);
        return *this;
    }

    Window& Window::destroyMenu(HMENU menu) {
        DestroyMenu(REAL_HMENU(menu));
        return *this;
    }

    void Window::initializeRender() {
        pImpl_->directBuffer_.bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        pImpl_->directBuffer_.bmi.bmiHeader.biPlanes = 1;
        pImpl_->directBuffer_.bmi.bmiHeader.biBitCount = 32;
        pImpl_->directBuffer_.bmi.bmiHeader.biCompression = BI_RGB;
        if (pImpl_->hwnd_ && !pImpl_->persistentDC_) {
            pImpl_->persistentDC_ = GetDC(REAL_HWND(pImpl_->hwnd_));
        }
    }

    void Window::updateWindowStyles() {
        if (pImpl_->hwnd_) {
            SetWindowPos(REAL_HWND(pImpl_->hwnd_), nullptr, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
        }
    }

    Window& Window::render(const Buffer& buffer, int destX, int destY,
        int srcX, int srcY, int width, int height,
        bool clearBackground, COLORREF bgColor) {

        if (!pImpl_->hwnd_ || !buffer.color || !pImpl_->persistentDC_) return *this;

        RECT clientRect;
        GetClientRect(REAL_HWND(pImpl_->hwnd_), &clientRect);
        const int clientWidth = clientRect.right;
        const int clientHeight = clientRect.bottom;

        const int srcFullWidth = (width == -1) ? buffer.width : width;
        const int srcFullHeight = (height == -1) ? buffer.height : height;

        int actualSrcX = srcX;
        int actualSrcY = srcY;
        int actualSrcWidth = srcFullWidth;
        int actualSrcHeight = srcFullHeight;

        if (actualSrcX < 0) {
            actualSrcWidth += actualSrcX;
            actualSrcX = 0;
        }
        if (actualSrcY < 0) {
            actualSrcHeight += actualSrcY;
            actualSrcY = 0;
        }

        if (actualSrcX + actualSrcWidth > buffer.width) {
            actualSrcWidth = buffer.width - actualSrcX;
        }
        if (actualSrcY + actualSrcHeight > buffer.height) {
            actualSrcHeight = buffer.height - actualSrcY;
        }

        int actualDestX = destX;
        int actualDestY = destY;
        int renderWidth = actualSrcWidth;
        int renderHeight = actualSrcHeight;

        if (actualDestX < 0) {
            int clipLeft = -actualDestX;
            actualSrcX += clipLeft;
            renderWidth -= clipLeft;
            actualDestX = 0;
        }
        if (actualDestY < 0) {
            int clipTop = -actualDestY;
            actualSrcY += clipTop;
            renderHeight -= clipTop;
            actualDestY = 0;
        }

        if (actualDestX + renderWidth > clientWidth) {
            renderWidth = clientWidth - actualDestX;
        }
        if (actualDestY + renderHeight > clientHeight) {
            renderHeight = clientHeight - actualDestY;
        }

        if (renderWidth <= 0 || renderHeight <= 0 ||
            actualSrcX >= buffer.width || actualSrcY >= buffer.height ||
            actualSrcX + renderWidth <= 0 || actualSrcY + renderHeight <= 0) {
            return *this;
        }

        bool currentSizeChanged = (clientWidth != pImpl_->lastClientWidth_ ||
            clientHeight != pImpl_->lastClientHeight_ ||
            buffer.width != pImpl_->lastBufferWidth_ ||
            buffer.height != pImpl_->lastBufferHeight_);

        if (clearBackground && currentSizeChanged) {
            HBRUSH hBgBrush = CreateSolidBrush(bgColor);

            if (destY > 0) {
                RECT topRect = { 0, 0, clientWidth, destY };
                FillRect(REAL_HDC(pImpl_->persistentDC_), &topRect, hBgBrush);
            }

            if (destY + srcFullHeight < clientHeight) {
                RECT bottomRect = { 0, destY + srcFullHeight, clientWidth, clientHeight };
                FillRect(REAL_HDC(pImpl_->persistentDC_), &bottomRect, hBgBrush);
            }

            if (destX > 0) {
                RECT leftRect = { 0, destY, destX, destY + srcFullHeight };
                FillRect(REAL_HDC(pImpl_->persistentDC_), &leftRect, hBgBrush);
            }

            if (destX + srcFullWidth < clientWidth) {
                RECT rightRect = { destX + srcFullWidth, destY, clientWidth, destY + srcFullHeight };
                FillRect(REAL_HDC(pImpl_->persistentDC_), &rightRect, hBgBrush);
            }

            DeleteObject(hBgBrush);

            pImpl_->lastClientWidth_ = clientWidth;
            pImpl_->lastClientHeight_ = clientHeight;
            pImpl_->lastBufferWidth_ = buffer.width;
            pImpl_->lastBufferHeight_ = buffer.height;
            pImpl_->sizeChanged_ = true;
        }
        else {
            pImpl_->sizeChanged_ = false;
        }

        if (pImpl_->directBuffer_.bmi.bmiHeader.biWidth != buffer.width ||
            pImpl_->directBuffer_.bmi.bmiHeader.biHeight != -buffer.height) {
            pImpl_->directBuffer_.bmi.bmiHeader.biWidth = buffer.width;
            pImpl_->directBuffer_.bmi.bmiHeader.biHeight = -buffer.height;
        }

        int bufferSrcY = buffer.height - actualSrcY - renderHeight;

        SetDIBitsToDevice(
            REAL_HDC(pImpl_->persistentDC_),
            actualDestX, actualDestY,
            renderWidth, renderHeight,
            actualSrcX,
            bufferSrcY,
            0,
            buffer.height,
            buffer.color,
            &pImpl_->directBuffer_.bmi,
            DIB_RGB_COLORS
        );

        return *this;
    }

    Window& Window::renderCentered(const Buffer& buffer, bool clearBackground, COLORREF bgColor) {
        if (!pImpl_->hwnd_ || !buffer.color || !pImpl_->persistentDC_) return *this;

        RECT clientRect;
        GetClientRect(REAL_HWND(pImpl_->hwnd_), &clientRect);
        const int clientWidth = clientRect.right;
        const int clientHeight = clientRect.bottom;
        const int destX = (clientWidth - buffer.width) / 2;
        const int destY = (clientHeight - buffer.height) / 2;

        bool currentSizeChanged = (clientWidth != pImpl_->lastClientWidth_ ||
            clientHeight != pImpl_->lastClientHeight_ ||
            buffer.width != pImpl_->lastBufferWidth_ ||
            buffer.height != pImpl_->lastBufferHeight_);

        if (clearBackground && currentSizeChanged &&
            (clientWidth > buffer.width || clientHeight > buffer.height)) {

            HBRUSH hBgBrush = CreateSolidBrush(bgColor);

            if (destY > 0) {
                RECT topRect = { 0, 0, clientWidth, destY };
                FillRect(REAL_HDC(pImpl_->persistentDC_), &topRect, hBgBrush);
            }
            if (destY + buffer.height < clientHeight) {
                RECT bottomRect = { 0, destY + buffer.height, clientWidth, clientHeight };
                FillRect(REAL_HDC(pImpl_->persistentDC_), &bottomRect, hBgBrush);
            }
            if (destX > 0) {
                RECT leftRect = { 0, destY, destX, destY + buffer.height };
                FillRect(REAL_HDC(pImpl_->persistentDC_), &leftRect, hBgBrush);
            }
            if (destX + buffer.width < clientWidth) {
                RECT rightRect = { destX + buffer.width, destY, clientWidth, destY + buffer.height };
                FillRect(REAL_HDC(pImpl_->persistentDC_), &rightRect, hBgBrush);
            }

            DeleteObject(hBgBrush);

            pImpl_->lastClientWidth_ = clientWidth;
            pImpl_->lastClientHeight_ = clientHeight;
            pImpl_->lastBufferWidth_ = buffer.width;
            pImpl_->lastBufferHeight_ = buffer.height;
            pImpl_->sizeChanged_ = true;
        }
        else {
            pImpl_->sizeChanged_ = false;
        }

        if (pImpl_->directBuffer_.bmi.bmiHeader.biWidth != buffer.width ||
            pImpl_->directBuffer_.bmi.bmiHeader.biHeight != -buffer.height) {
            pImpl_->directBuffer_.bmi.bmiHeader.biWidth = buffer.width;
            pImpl_->directBuffer_.bmi.bmiHeader.biHeight = -buffer.height;
        }

        SetDIBitsToDevice(
            REAL_HDC(pImpl_->persistentDC_),
            destX, destY,
            buffer.width, buffer.height,
            0, 0,
            0, buffer.height,
            buffer.color,
            &pImpl_->directBuffer_.bmi,
            DIB_RGB_COLORS
        );
        return *this;
    }

    Window& Window::render(const Canvas& canvas, int destX, int destY,
        int srcX, int srcY, int width, int height,
        bool clearBackground, COLORREF bgColor) {
        return render(canvas.getBuffer(), destX, destY, srcX, srcY, width, height, clearBackground, bgColor);
    }

    Window& Window::renderCentered(const Canvas& canvas, bool clearBackground, COLORREF bgColor) {
        return renderCentered(canvas.getBuffer(), clearBackground, bgColor);
    }

    Point getScreenSize() {
        return { (float)GetSystemMetrics(SM_CXSCREEN), (float)GetSystemMetrics(SM_CYSCREEN) };
    }

    Point getWorkAreaSize() {
        RECT rect;
        SystemParametersInfo(SPI_GETWORKAREA, 0, &rect, 0);
        return { (float)rect.right - rect.left, (float)rect.bottom - rect.top };
    }

    double getDpiScale() {
        HDC hdc = GetDC(nullptr);
        double scale = GetDeviceCaps(hdc, LOGPIXELSX) / 96.0;
        ReleaseDC(nullptr, hdc);
        return scale;
    }

    void Window::startMessageThread(int width, int height, const std::string& title) {
        pImpl_->running_ = true;
        pImpl_->messageThread_ = std::thread([this, width, height, title]() {
            pImpl_->threadId_ = GetCurrentThreadId();
            registerWindowClassOnce();
            RECT rect = { 0, 0, width, height };
            AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
            pImpl_->hwnd_ = CreateWindowEx(0, "PA2D", "", WS_OVERLAPPEDWINDOW,
                CW_USEDEFAULT, CW_USEDEFAULT,
                rect.right - rect.left, rect.bottom - rect.top,
                nullptr, nullptr, GetModuleHandle(nullptr), this);
            if (pImpl_->hwnd_) {
                SetWindowTextA(REAL_HWND(pImpl_->hwnd_), title.c_str());
                initializeRender();
                TRACKMOUSEEVENT tme = { sizeof(TRACKMOUSEEVENT) };
                tme.dwFlags = TME_LEAVE;
                tme.hwndTrack = REAL_HWND(pImpl_->hwnd_);
                TrackMouseEvent(&tme);
                pImpl_->trackingMouse_ = true;
                pImpl_->initPromise_.set_value(true);
            }
            else {
                pImpl_->initPromise_.set_value(false);
                return;
            }
            MSG msg;
            while (pImpl_->running_ && GetMessage(&msg, nullptr, 0, 0)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            });
    }

    void Window::stopMessageThread() {
        pImpl_->running_ = false;
        if (pImpl_->messageThread_.joinable()) {
            if (pImpl_->threadId_ != 0) {
                PostThreadMessage(pImpl_->threadId_, WM_QUIT, 0, 0);
            }
            pImpl_->messageThread_.join();
        }
    }

    void Window::handleMouse(UINT msg, WPARAM wparam, LPARAM lparam) {
        if (!pImpl_->mouseCb_) return;

        MouseEvent e;
        e.x = GET_X_LPARAM(lparam);
        e.y = GET_Y_LPARAM(lparam);
        e.wheelDelta = 0;

        switch (msg) {
        case WM_LBUTTONDOWN: case WM_LBUTTONUP:
            e.button = 0; e.pressed = (msg == WM_LBUTTONDOWN); break;
        case WM_RBUTTONDOWN: case WM_RBUTTONUP:
            e.button = 1; e.pressed = (msg == WM_RBUTTONDOWN); break;
        case WM_MBUTTONDOWN: case WM_MBUTTONUP:
            e.button = 2; e.pressed = (msg == WM_MBUTTONDOWN); break;
        case WM_MOUSEWHEEL:
            e.button = -1;
            e.wheelDelta = GET_WHEEL_DELTA_WPARAM(wparam);
            e.pressed = false;
            break;
        case WM_MOUSEMOVE:
            e.button = -2;
            e.pressed = false;
            if (!pImpl_->trackingMouse_) {
                TRACKMOUSEEVENT tme = { sizeof(TRACKMOUSEEVENT) };
                tme.dwFlags = TME_LEAVE;
                tme.hwndTrack = REAL_HWND(pImpl_->hwnd_);
                TrackMouseEvent(&tme);
                pImpl_->trackingMouse_ = true;
            }
            break;
        case WM_MOUSELEAVE: {
            e.button = -3;
            e.pressed = false;
            pImpl_->trackingMouse_ = false;
            break;
        }
        default:
            return;
        }
        pImpl_->mouseCb_(e);
    }

    LRESULT CALLBACK Window::windowProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
        Window* win = reinterpret_cast<Window*>(GetWindowLongPtr(REAL_HWND(hwnd), GWLP_USERDATA));

        if (msg == WM_NCCREATE) {
            win = reinterpret_cast<Window*>(((CREATESTRUCT*)lparam)->lpCreateParams);
            SetWindowLongPtr(REAL_HWND(hwnd), GWLP_USERDATA, reinterpret_cast<LONG_PTR>(win));
            if (win && win->pImpl_) {
                win->pImpl_->hwnd_ = hwnd;
            }
        }

        if (!win) return DefWindowProc(REAL_HWND(hwnd), msg, wparam, lparam);

        switch (msg) {
        case WM_DESTROY:
            win->pImpl_->hwnd_ = nullptr;
            {
                std::lock_guard<std::mutex> lock(win->pImpl_->closeMutex_);
                win->pImpl_->windowClosed_ = true;
            }
            win->pImpl_->closeCv_.notify_all();

            return 0;

        case WM_CLOSE:
            if (win->pImpl_->closeCb_ && !win->pImpl_->closeCb_()) {
                return 0;
            }
            break;

        case WM_SIZE:
            if (win->pImpl_->resizeCb_) win->pImpl_->resizeCb_(LOWORD(lparam), HIWORD(lparam));
            return 0;

        case WM_KEYDOWN: case WM_KEYUP:
            if (win->pImpl_->keyCb_) win->pImpl_->keyCb_({ static_cast<int>(wparam), msg == WM_KEYDOWN });
            break;

        case WM_CHAR:
            if (win->pImpl_->charCb_) {
                wchar_t ch = static_cast<wchar_t>(wparam);
                if (ch == '\b' || ch == '\r' || ch == '\t' || ch >= 32) {
                    win->pImpl_->charCb_(ch);
                }
            }
            break;

        case WM_SETFOCUS:
            if (win->pImpl_->focusCb_) win->pImpl_->focusCb_(true);
            break;

        case WM_KILLFOCUS:
            if (win->pImpl_->focusCb_) win->pImpl_->focusCb_(false);
            break;

        case WM_COMMAND:
            if (win->pImpl_->menuCb_) {
                int menuId = LOWORD(wparam);
                int notificationCode = HIWORD(wparam);
                if (notificationCode == 0 || notificationCode == 1) {
                    win->pImpl_->menuCb_(menuId);
                }
            }
            break;

        case WM_LBUTTONDOWN: case WM_LBUTTONUP:
        case WM_RBUTTONDOWN: case WM_RBUTTONUP:
        case WM_MBUTTONDOWN: case WM_MBUTTONUP:
        case WM_MOUSEWHEEL:
        case WM_MOUSEMOVE:
        case WM_MOUSELEAVE:
            win->handleMouse(msg, wparam, lparam);
            break;

        case WM_DROPFILES: {
            HDROP hDrop = (HDROP)wparam;
            UINT fileCount = DragQueryFile(hDrop, 0xFFFFFFFF, nullptr, 0);

            std::vector<std::string> files;
            for (UINT i = 0; i < fileCount; i++) {
                char filePath[MAX_PATH];
                DragQueryFileA(hDrop, i, filePath, MAX_PATH);
                files.push_back(filePath);
            }

            if (win->pImpl_->dropCb_) {
                win->pImpl_->dropCb_(files);
            }

            DragFinish(hDrop);
            return 0;
        }

        case WM_DRAWCLIPBOARD: {
            if (win && win->pImpl_->clipboardFilesCb_) {
                auto files = win->getClipboardFiles();
                if (!files.empty()) {
                    win->pImpl_->clipboardFilesCb_(files);
                }
            }
            if (win->pImpl_->nextClipboardViewer_) {
                SendMessage(REAL_HWND(win->pImpl_->nextClipboardViewer_), WM_DRAWCLIPBOARD, wparam, lparam);
            }
            break;
        }

        case WM_CHANGECBCHAIN: {
            if (win->pImpl_->nextClipboardViewer_ == (HWND)wparam) {
                win->pImpl_->nextClipboardViewer_ = (HWND)lparam;
            }
            else if (win->pImpl_->nextClipboardViewer_) {
                SendMessage(REAL_HWND(win->pImpl_->nextClipboardViewer_), WM_CHANGECBCHAIN, wparam, lparam);
            }
            break;
        }

        case WM_GETMINMAXINFO: {
            MINMAXINFO* minMaxInfo = (MINMAXINFO*)lparam;
            if (win->pImpl_->minWidth_ > 0 && win->pImpl_->minHeight_ > 0) {
                minMaxInfo->ptMinTrackSize.x = win->pImpl_->minWidth_;
                minMaxInfo->ptMinTrackSize.y = win->pImpl_->minHeight_;
            }
            if (win->pImpl_->maxWidth_ > 0 && win->pImpl_->maxHeight_ > 0) {
                minMaxInfo->ptMaxTrackSize.x = win->pImpl_->maxWidth_;
                minMaxInfo->ptMaxTrackSize.y = win->pImpl_->maxHeight_;
            }
            return 0;
        }
        case WM_NCHITTEST: {
            if (win->pImpl_->isTitlebarless_) {
                POINT pt = { GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam) };
                ScreenToClient(REAL_HWND(win->pImpl_->hwnd_), &pt);

                RECT clientRect;
                GetClientRect(REAL_HWND(win->pImpl_->hwnd_), &clientRect);

                if (pt.y < 30 && pt.x < clientRect.right - 100) {
                    return HTCAPTION;
                }
            }
            break;
        }
        }

        return DefWindowProc(REAL_HWND(hwnd), msg, wparam, lparam);
    }

} // namespace pa2d