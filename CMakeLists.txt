project ("PASDK")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)    # 静态库.lib
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)    # 动态库.dll
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)    # 可执行文件.exe

# 包含子项目：

# 编译 pa2d 库
add_subdirectory ("pa2d")
# pa2d 测试程序
add_subdirectory ("tests")
# pa2d 拓展渲染
add_subdirectory ("extensions")