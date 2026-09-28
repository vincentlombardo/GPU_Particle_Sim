#pragma once
#include <vector_types.h>  // float2

// The only thing the renderer is allowed to consume.
// position points into a CUDA/OpenGL interop buffer -- there is no
// separate host or device copy of this data outside that buffer.
struct RenderState
{
    float2* position = nullptr;
    int     count    = 0;
};
