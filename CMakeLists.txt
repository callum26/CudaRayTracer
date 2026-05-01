cmake_minimum_required(VERSION 3.18)

if(WIN32 AND NOT DEFINED CMAKE_SYSTEM_VERSION)
    set(CMAKE_SYSTEM_VERSION 10.0.22621.0 CACHE STRING "Windows SDK version" FORCE)
endif()

project(CudaRayTracer LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)

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
set_target_properties(rayTracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(rayTracer PRIVATE
    glfw
    glad
    CUDA::cudart
    OpenGL::GL
)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shaders DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
