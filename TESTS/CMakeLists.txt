project(tests)

add_executable (tests "tests.cpp" )

# 链接pa2d静态库
target_link_libraries(tests PRIVATE pa2d)
