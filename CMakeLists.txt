cmake_minimum_required(VERSION 3.24)
project(particle-sim LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# GLAD is not fetched automatically. Generate it once from
# https://glad.dav1d.de (Language: C/C++, gl: Version 4.5, Profile: Core)
# and unpack it so this layout exists:
#   third_party/glad/include/glad/glad.h
#   third_party/glad/include/KHR/khrplatform.h
#   third_party/glad/src/glad.c
add_library(glad STATIC third_party/glad/src/glad.c)
target_include_directories(glad PUBLIC third_party/glad/include)

add_executable(particle-sim
    src/main.cpp
    src/Simulation.cu
    src/RenderStateGenerator.cu
    src/Renderer.cpp
    src/InteropBuffer.cpp
)

target_include_directories(particle-sim PRIVATE src)

target_link_libraries(particle-sim PRIVATE
    OpenGL::GL
    glfw
    glad
    CUDA::cudart
)

set_target_properties(particle-sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Keep shaders/ next to the built binary so the runtime relative paths
# ("shaders/particle.vert") resolve without extra setup.
add_custom_command(TARGET particle-sim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/shaders $<TARGET_FILE_DIR:particle-sim>/shaders)
