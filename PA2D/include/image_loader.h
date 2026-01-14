#pragma once
#include "buffer.h"
namespace pa2d {
    bool loadImage(Buffer& buffer, const char* filePath);
    bool loadImage(Buffer& buffer, int resourceID);
    bool loadImage(Buffer& buffer, void* hInstance, int resourceID);
}