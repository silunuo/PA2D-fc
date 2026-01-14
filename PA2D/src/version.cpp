#include"../include/version.h"
#include <windows.h>
#include <string>
namespace pa2d {
    // ==================== 版本信息 ====================
    const char* VERSION_STRING = "1.0.0-beta.1";
    // 获取完整版本信息
    const char* getVersion() {
        // 使用静态局部变量保证线程安全和一次性初始化
        static const std::string fullVersion = []() {
            std::string version = "PA2D v" + std::string(VERSION_STRING);

            // 添加构建时间
            version += " (Build: " + std::string(__DATE__) + " " + __TIME__ + ")";

            // 构建配置
#ifdef _DEBUG
            version += " [Debug]";
#else
            version += " [Release]";
#endif
            return version;
            }();
        return fullVersion.c_str();
    }

    AVX2_Verifier::AVX2_Verifier() {
        int info[4];
        __cpuid(info, 0);
        if (info[0] < 7) fail();
        __cpuidex(info, 7, 0);
        if (!(info[1] & (1 << 5))) fail();
    }
    void AVX2_Verifier::fail() {
        MessageBoxA(nullptr,
            "PA2D requires AVX2 CPU support.\n\n"
            "Minimum: Intel Haswell (2013) / AMD Excavator (2015)",
            "CPU Not Supported",
            MB_OK | MB_ICONERROR);
        ExitProcess(1);
    }
}