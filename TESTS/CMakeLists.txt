# 将源代码添加到此项目的可执行文件。
add_executable (tests "tests.cpp" )

# 链接pa2d静态库
target_link_libraries(tests PRIVATE pa2d)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET tests PROPERTY CXX_STANDARD 20)
endif()