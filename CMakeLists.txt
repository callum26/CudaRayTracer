cmake_minimum_required(VERSION 3.18)
project(CudaRayTracer LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GLFW options
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(import/glfw-3.4)

# GLAD (OpenGL loader)
add_library(glad STATIC import/glad/src/glad.c)
target_include_directories(glad PUBLIC import/glad/include)

# Main CUDA ray tracer executable
add_executable(rayTracer
    main.cpp
    kernel.cu
)
target_include_directories(rayTracer PRIVATE
    import
    import/glad/include
    import/glfw-3.4/include
)
target_link_libraries(rayTracer PRIVATE glfw glad opengl32)
