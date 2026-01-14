namespace pa2d {
    // ==================== 版本信息 ====================
    extern const char* VERSION_STRING;
    // 获取完整版本信息
    extern const char* getVersion();
    // 硬件支持检测
    struct AVX2_Verifier {
        AVX2_Verifier();
    private:
        [[noreturn]] static void fail();
    };
}