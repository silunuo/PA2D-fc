project(extensions)

file(GLOB SOURCE_FILES 
    "graphics/src/*.cpp"
    "extensions.cpp"
)

add_executable(extensions ${SOURCE_FILES})

# Á´½Ópa2d¾²Ì¬¿â
target_link_libraries(extensions PRIVATE pa2d)