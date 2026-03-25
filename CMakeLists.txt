cmake_minimum_required(VERSION 3.18)

if(WIN32)
    if(DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}")
        file(TO_CMAKE_PATH "$ENV{CUDA_PATH}" _cuda_toolkit_dir)
        set(CMAKE_VS_PLATFORM_TOOLSET_CUDA_CUSTOM_DIR "${_cuda_toolkit_dir}")
    elseif(EXISTS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1")
        set(CMAKE_VS_PLATFORM_TOOLSET_CUDA_CUSTOM_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1")
    endif()
endif()

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

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shaders DESTINATION ${CMAKE_CURRENT_BINARY_DIR})