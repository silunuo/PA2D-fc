#include "../include/image_loader.h"
#include <wincodec.h>
#include <immintrin.h>
#include <atomic>
#include <vector>
#include <memory>
#include <functional>
#include <string>
#pragma comment(lib, "WindowsCodecs.lib")
#pragma comment(lib, "ole32.lib")

namespace pa2d {

    // 内部实现细节 - 不暴露给用户
    namespace internal {
        bool EnsureCOMInitialized();
        void ConvertRGBAtoBGRA(Color* dest, const uint32_t* src, int pixelCount);
        bool LoadImageCommon(Buffer& buffer, std::function<bool(class WICResource&)> loader);
        bool LoadImageFromResourceInternal(Buffer& buffer, void* hInstancePtr, unsigned int resourceID);
        bool LoadImageFromFileInternal(Buffer& buffer, const char* filePath);

        class WICResource {
        public:
            IWICImagingFactory* factory = nullptr;
            IWICStream* stream = nullptr;
            IWICBitmapDecoder* decoder = nullptr;
            IWICBitmapFrameDecode* frame = nullptr;
            IWICFormatConverter* converter = nullptr;

            ~WICResource();
            bool InitializeFactory();
        };
    }

    // COM初始化
    bool internal::EnsureCOMInitialized() {
        static std::atomic<bool> com_initialized(false);
        if (!com_initialized.load(std::memory_order_acquire)) {
            HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
            if (SUCCEEDED(hr) || hr == RPC_E_CHANGED_MODE) {
                com_initialized.store(true, std::memory_order_release);
            }
            return SUCCEEDED(hr) || hr == RPC_E_CHANGED_MODE;
        }
        return true;
    }

    // 颜色转换
    void internal::ConvertRGBAtoBGRA(Color* dest, const uint32_t* src, int pixelCount) {
        const int avxCount = pixelCount / 8;
        for (int i = 0; i < avxCount; ++i) {
            __m256i rgba = _mm256_loadu_si256((__m256i const*)(src + i * 8));
            __m256i shuffled = _mm256_shuffle_epi8(rgba, _mm256_setr_epi8(
                2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15,
                18, 17, 16, 19, 22, 21, 20, 23, 26, 25, 24, 27, 30, 29, 28, 31));
            _mm256_storeu_si256((__m256i*) & dest[i * 8], shuffled);
        }

        // 处理剩余像素
        for (int i = avxCount * 8; i < pixelCount; ++i) {
            uint32_t rgba = src[i];
            dest[i].data = (rgba & 0xFF00FF00) | ((rgba & 0x00FF0000) >> 16) | ((rgba & 0x000000FF) << 16);
        }
    }

    // WICResource析构函数
    internal::WICResource::~WICResource() {
        if (converter) converter->Release();
        if (frame) frame->Release();
        if (decoder) decoder->Release();
        if (stream) stream->Release();
        if (factory) factory->Release();
    }

    // WICResource工厂初始化
    bool internal::WICResource::InitializeFactory() {
        return SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr,
            CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)));
    }

    // 主要加载逻辑
    bool internal::LoadImageCommon(Buffer& buffer, std::function<bool(WICResource&)> loader) {
        if (!EnsureCOMInitialized()) return false;

        WICResource wic;
        if (!wic.InitializeFactory()) return false;

        if (!loader(wic)) return false;
        if (!wic.frame) return false;

        UINT width = 0, height = 0;
        wic.frame->GetSize(&width, &height);

        if (SUCCEEDED(wic.factory->CreateFormatConverter(&wic.converter)) &&
            SUCCEEDED(wic.converter->Initialize(wic.frame, GUID_WICPixelFormat32bppRGBA,
                WICBitmapDitherTypeNone, nullptr, 0.0f, WICBitmapPaletteTypeCustom))) {

            if (buffer.width != width || buffer.height != height) {
                if (buffer.color) free(buffer.color);
                buffer.width = width;
                buffer.height = height;
                buffer.color = static_cast<Color*>(malloc(width * height * sizeof(Color)));
            }

            if (buffer.color) {
                std::vector<uint32_t> tempBuffer(width * height);
                UINT stride = width * sizeof(uint32_t);
                if (SUCCEEDED(wic.converter->CopyPixels(nullptr, stride,
                    static_cast<UINT>(tempBuffer.size() * sizeof(uint32_t)),
                    reinterpret_cast<BYTE*>(tempBuffer.data())))) {

                    ConvertRGBAtoBGRA(buffer.color, tempBuffer.data(), width * height);
                    return true;
                }
            }
        }
        return false;
    }

    // 资源文件加载
    bool internal::LoadImageFromResourceInternal(Buffer& buffer, void* hInstancePtr, unsigned int resourceID) {
        return LoadImageCommon(buffer, [&](WICResource& wic) {
            HINSTANCE hInstance = static_cast<HINSTANCE>(hInstancePtr);
            HRSRC hRes = FindResource(hInstance, MAKEINTRESOURCE(static_cast<UINT>(resourceID)), "PNG");
            if (!hRes) return false;

            HGLOBAL hResData = LoadResource(hInstance, hRes);
            if (!hResData) return false;

            const unsigned char* pResData = (const unsigned char*)LockResource(hResData);
            DWORD dataSize = SizeofResource(hInstance, hRes);
            if (!pResData || dataSize == 0) return false;

            return SUCCEEDED(wic.factory->CreateStream(&wic.stream)) &&
                SUCCEEDED(wic.stream->InitializeFromMemory(const_cast<BYTE*>(pResData), dataSize)) &&
                SUCCEEDED(wic.factory->CreateDecoderFromStream(wic.stream, nullptr,
                    WICDecodeMetadataCacheOnLoad, &wic.decoder)) &&
                SUCCEEDED(wic.decoder->GetFrame(0, &wic.frame));
            });
    }

    // 文件加载
    bool internal::LoadImageFromFileInternal(Buffer& buffer, const char* filePath) {
        if (GetFileAttributesA(filePath) == INVALID_FILE_ATTRIBUTES) return false;

        return LoadImageCommon(buffer, [&](WICResource& wic) {
            int wideLen = MultiByteToWideChar(CP_UTF8, 0, filePath, -1, nullptr, 0);
            if (wideLen == 0) return false;

            std::vector<wchar_t> widePath(wideLen);
            MultiByteToWideChar(CP_UTF8, 0, filePath, -1, widePath.data(), wideLen);

            return SUCCEEDED(wic.factory->CreateDecoderFromFilename(widePath.data(), nullptr,
                GENERIC_READ, WICDecodeMetadataCacheOnLoad, &wic.decoder)) &&
                SUCCEEDED(wic.decoder->GetFrame(0, &wic.frame));
            });
    }

    // 公共接口实现
    bool loadImage(Buffer& buffer, const char* filePath) {
        if (!internal::EnsureCOMInitialized()) return false;

        if (internal::LoadImageFromFileInternal(buffer, filePath)) return true;

        char currentDir[MAX_PATH];
        GetCurrentDirectoryA(MAX_PATH, currentDir);
        std::string fullPath = std::string(currentDir) + "\\" + filePath;
        if (internal::LoadImageFromFileInternal(buffer, fullPath.c_str())) return true;

        char exePath[MAX_PATH];
        GetModuleFileNameA(nullptr, exePath, MAX_PATH);
        std::string exeDir(exePath);
        size_t lastSlash = exeDir.find_last_of("\\/");
        if (lastSlash != std::string::npos) {
            std::string exeFullPath = exeDir.substr(0, lastSlash + 1) + filePath;
            if (internal::LoadImageFromFileInternal(buffer, exeFullPath.c_str())) return true;
        }

        return false;
    }

    bool loadImage(Buffer& buffer, int resourceID) {
        return internal::LoadImageFromResourceInternal(buffer, GetModuleHandle(nullptr), resourceID);
    }

    bool loadImage(Buffer& buffer, void* hInstance, int resourceID) {
        return internal::LoadImageFromResourceInternal(buffer, hInstance, resourceID);
    }
}