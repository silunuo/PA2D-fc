project(pa2d)
# 自动查找源文件
file(GLOB SOURCES
    "src/*.cpp"
    "src/geometry/*.cpp"
    "src/draw/*.cpp"
    "src/internal/*.cpp"
)
# 创建静态库
add_library(pa2d STATIC ${SOURCES} )
# 设置包含目录
target_include_directories(pa2d
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/pa2d.h>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
# 设置输出目录
set_target_properties(pa2d PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    DEBUG_POSTFIX "d"
)