project(extensions)

file(GLOB SOURCE_FILES 
    "graphics/src/*.cpp"
    "extensions.cpp"
    "filter/filter.cpp"   # 添加了滤镜模块
)

add_executable(extensions ${SOURCE_FILES})

# 链接pa2d静态库
target_link_libraries(extensions PRIVATE pa2d)

target_include_directories(extensions PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})