#include <string>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <fstream>
#include <cstdio>
#include <sstream>
#include "raytracer.h"
#include <array>
#include <vector>
#include <iomanip>
#include <future>

// for image creating
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "glfw-3.4/deps/stb_image_write.h"

const int benchmarkFrameCount = 500;

struct FrameData
{
    int index;
    float currentFps;
    float gpuMs;
    float totalMs;
    SceneSettings settings;
    CurrentMode mode;
};

struct OutputNames
{
    std::string csvName;
    std::array<std::string, 4> convergePng;
};

void exportToCsv(const std::string &name, const std::vector<FrameData> &data)
{
    std::ofstream file(name);

    if (!file.is_open())
    {
        printf("failed to open for writing");
        return;
    }

    if (data.empty())
    {
        printf("no data to export");
        return;
    }

    unsigned long long totalBounces = 0;
    unsigned long long totalRays = 0;

    getBounceStats(totalBounces, totalRays);

    if (totalRays <= 0)
    {
        printf("no rays?");
        return;
    }

    double avgBounces = (double)totalBounces / totalRays;

    SceneSettings initSettings = data[0].settings;
    CurrentMode initMode = data[0].mode;
    file << "maxBounces,maxShadowBounces,samplesPerPixel,lightSamples,screenWidth,screenHeight,useBVH,perfTest,gridSize\n";
    file << initSettings.maxBounces << "," << initSettings.maxShadowBounces << "," << initSettings.samplesPerPixel
         << "," << initSettings.lightSamples << "," << initSettings.screenWidth << "," << initSettings.screenHeight
         << "," << (initMode.useBVH ? 1 : 0) << "," << (initMode.perfTest ? 1 : 0) << "," << initSettings.perfTestGridSize << "\n";
    file << "frameIndex,fps, gpuMs,totalMs\n";
    file << std::fixed << std::setprecision(2);

    for (const auto &f : data)
    {

        file << f.index << "," << f.currentFps << "," << f.gpuMs << "," << f.totalMs << "," << "\n";
    }

    file << "Total Rays,Total Bounces,Average Bounces Per Ray\n";
    file << totalRays << "," << totalBounces << "," << avgBounces << "\n";

    printf("Exported 500 frames to %s\n", name.c_str());
}

// shader read func from https://learnopengl.com/Getting-started/Shaders
std::string readShader(const char *path)
{
    std::ifstream in(path);
    if (!in)
        return std::string();

    std::stringstream ss;

    ss << in.rdbuf();
    return ss.str();
}

// shader compile func from https://learnopengl.com/Getting-started/Shaders
unsigned int compileShader(unsigned int type, const char *src)
{
    unsigned int s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);

    // check compilation status
    int ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char b[512];
        glGetShaderInfoLog(s, 512, NULL, b);
        printf("shader compile error %s\n", b);
    }
    return s;
}

// shader program creation func from https://learnopengl.com/Getting-started/Shaders
unsigned int createProgram(const char *vpath, const char *fpath)
{
    std::string vs = readShader(vpath);
    std::string fs = readShader(fpath);

    if (vs.empty() || fs.empty())
    {
        printf("missing shader files\n");
        return 0;
    }

    unsigned int vsId = compileShader(GL_VERTEX_SHADER, vs.c_str());
    unsigned int fsId = compileShader(GL_FRAGMENT_SHADER, fs.c_str());

    unsigned int p = glCreateProgram();
    glAttachShader(p, vsId);
    glAttachShader(p, fsId);
    glLinkProgram(p);

    int isOkay;
    glGetProgramiv(p, GL_LINK_STATUS, &isOkay);
    if (!isOkay)
    {
        char b[512];
        glGetProgramInfoLog(p, 512, NULL, b);
        printf("shader link error %s\n", b);
    }

    glDeleteShader(vsId);
    glDeleteShader(fsId);
    return p;
}

// move outside main we can use key proeprlly
namespace fullState
{
    // move the calculation of host pixel buffer solely with cpp
    SceneSettings settings = {
        15,  // maxBounces
        8,   // maxShadowBounces
        1,   // samplesPerPixel
        1,   // lightSamples
        800, // screenWidth
        800, // screenHeight
        10,  // grid size
    };

    CurrentMode mode = {
        false, // useBVH
        false, // perfTest
    };
}

// move keys to an event driven input callback
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (action != GLFW_PRESS)
        return;

    switch (key)
    {
    case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, true);
        break;
    case GLFW_KEY_B:
        fullState::mode.useBVH = !fullState::mode.useBVH;
        initScene(fullState::mode, fullState::settings);
        resetAccumulation(fullState::settings);
        break;
    case GLFW_KEY_T:
        fullState::mode.perfTest = !fullState::mode.perfTest;
        initScene(fullState::mode, fullState::settings);
        resetAccumulation(fullState::settings);
        break;
    }
}

OutputNames findFileNames(const SceneSettings &settings, const CurrentMode &mode, const int convergeAt[])
{
    OutputNames names;
    names.csvName = std::string("bench_") + (mode.perfTest ? "perftest" : "cornell") + "_" + (mode.useBVH ? "bvh" : "brute") + "_" + std::to_string(settings.screenWidth) + "x" + std::to_string(settings.screenHeight) + "_b" + std::to_string(settings.maxBounces) + "_sb" + std::to_string(settings.maxShadowBounces) + "_ls" + std::to_string(settings.lightSamples) + "_spp" + std::to_string(settings.samplesPerPixel) + (mode.perfTest ? "_gs" + std::to_string(settings.perfTestGridSize) : "") + ".csv";

    // change file name based on what we r doing
    for (size_t i = 0; i < names.convergePng.size(); ++i)
    {
        std::ostringstream name;

        name << (mode.perfTest ? "conv_perftest" : "conv_cornell")
             << "_b" << settings.maxBounces << "_sb" << settings.maxShadowBounces << "_spp" << settings.samplesPerPixel
             << "_ls" << settings.lightSamples << "_" << settings.screenWidth << "x" << settings.screenHeight
             << "_frame" << convergeAt[i] << ".png";

        names.convergePng[i] = name.str();
    }
    return names;
}

int main()
{
    SceneSettings settings = fullState::settings;
    CurrentMode mode = fullState::mode;

    int accumFrame = 0; // counts every rendered frame from startup

    const int screenWidth = settings.screenWidth;
    const int screenHeight = settings.screenHeight;
    const int convergeAt[] = {1, 64, 512, 4096};
    int convergeIdx = 0;

    bool benchmarkFinished = false;
    std::vector<FrameData> benchmarkData;
    benchmarkData.reserve(benchmarkFrameCount);

    OutputNames names = findFileNames(settings, mode, convergeAt);
    std::string csvName = names.csvName;
    std::array<std::string, 4> convergePng = names.convergePng;

    if (!glfwInit())
        return -1;

    // set opengl version 4.6
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *win = glfwCreateWindow(screenWidth, screenHeight, "CUDA Ray Tracer", NULL, NULL);

    if (!win)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(win);
    glfwSwapInterval(0); // turn off vsync

    glfwSetKeyCallback(win, keyCallback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        printf("glad init failed\n");
        return -1;
    }

    // creates/loads shader program
    unsigned int prog = createProgram("../shaders/vertex.glsl", "../shaders/fragment.glsl");

    // fullscreen quad vertices
    // position xy and texture coords uv
    float verts[] = {
        1, 1, 1, 1,   // top right
        1, -1, 1, 0,  // bottom right
        -1, -1, 0, 0, // bottom left
        -1, 1, 0, 1   // top left
    };

    // triangle indices for quad
    unsigned int idx[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };

    // create vertex array object vertex buffer object element buffer object
    unsigned int VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    // took from my graphics project in 2nd year and various other graphics module task sheets
    // upload vertex data
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    // upload index data
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // create texture for raytraced image
    unsigned int tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, screenWidth, screenHeight);

    // create PBO and register with CUDA
    GLuint pbo = 0;
    cudaGraphicsResource *cudaPbo = nullptr;

    // same as before
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, screenWidth * screenHeight * sizeof(uint8_t) * 4, nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // register PBO with CUDA
    cudaError_t cErr = cudaGraphicsGLRegisterBuffer(&cudaPbo, pbo, cudaGraphicsMapFlagsWriteDiscard);

    if (cErr != cudaSuccess)
    {
        fprintf(stderr, "cudaGraphicsGLRegisterBuffer failed: %s\n", cudaGetErrorString(cErr));
        cudaPbo = nullptr;
    }

    // cast to vector instead of using pointer
    std::vector<unsigned char> hostPixels(screenWidth * screenHeight * 4);
    stbi_flip_vertically_on_write(1);

    initDevicePixel(settings);
    initScene(mode, settings);

    float statsTimer = 0.0f;
    float frameCount = 0;
    float accumulatedGpuMs = 0.0f;
    float accumulatedTotalMs = 0.0f;
    float avgFps = 0.0f;

    while (!glfwWindowShouldClose(win))
    {
        // update settings/mode from fullState
        settings = fullState::settings;
        mode = fullState::mode;

        float frameStart = glfwGetTime();

        // clear screen
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        cudaGraphicsMapResources(1, &cudaPbo, 0);

        uchar4 *devPtr = nullptr;
        size_t mappedSize = 0;
        cudaGraphicsResourceGetMappedPointer((void **)&devPtr, &mappedSize, cudaPbo);

        // kernel writing directly into mapped PBO memory
        float gpuMs = launchRayTracerToDevice((void *)devPtr, settings, mode);

        // if we need to save this frame to PNG copy device to host while mapped
        if (convergeIdx < 4 && accumFrame == convergeAt[convergeIdx] && mode.useBVH)
        {
            // async copy frame data
            cudaMemcpy(hostPixels.data(), devPtr, screenWidth * screenHeight * 4, cudaMemcpyDeviceToHost);

            // local copy of the buffer so the background thread owns it
            std::vector<unsigned char> asyncPixels = hostPixels;
            // fetch png name
            std::string currentPngName = convergePng[convergeIdx];

            // before was casuing massive performance spikes skewing results
            // create seperate thread
            std::thread([asyncPixels, currentPngName, screenWidth, screenHeight]()
                        {
                if (stbi_write_png(currentPngName.c_str(), screenWidth, screenHeight, 4, asyncPixels.data(), screenWidth * 4)) {
                    printf("Saved %s\n", currentPngName.c_str());
                } })
                .detach();

            convergeIdx++;
        }

        cudaGraphicsUnmapResources(1, &cudaPbo, 0);

        // upload is now just a tex sub image from the bound PBO
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, screenWidth, screenHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // draw textured quad
        glUseProgram(prog);
        glBindVertexArray(VAO);
        glBindTexture(GL_TEXTURE_2D, tex);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // swap buffers and poll events
        glfwSwapBuffers(win);
        glfwPollEvents();

        // screenshots handled during PBO mapping when needed

        float totalMs = (float)((glfwGetTime() - frameStart) * 1000.0f);

        accumulatedGpuMs += gpuMs;
        accumulatedTotalMs += totalMs;
        frameCount++;
        accumFrame++;
        statsTimer += totalMs / 1000.0f;

        // reset accums before 100 warmup frames
        if (accumFrame == 100)
        {
            accumulatedGpuMs = 0.0f;
            accumulatedTotalMs = 0.0f;
            frameCount = 0;
            statsTimer = 0.0f;
        }

        float currentFps = 1000.0f / totalMs;

        if (!benchmarkFinished && accumFrame > 100)
        {
            benchmarkData.push_back({(int)benchmarkData.size(), currentFps, gpuMs, totalMs, settings, mode});
            if (benchmarkData.size() >= benchmarkFrameCount)
            {
                exportToCsv(csvName, benchmarkData);
                benchmarkFinished = true;
            }
        }

        // temp auto close once benchmark done and convergence images captured
        //  bool convDone = mode.perfTest || !mode.useBVH || convergeIdx >= 4;
        // if (benchmarkFinished && convDone)
        //     glfwSetWindowShouldClose(win, true);

        if (statsTimer >= 0.5)
        {
            float avgGpuMs = accumulatedGpuMs / frameCount;
            float avgTotalMs = accumulatedTotalMs / frameCount;
            float avgFps = 1000.0f / avgTotalMs;

            char title[96];
            // stores it in a suitabke buffer
            snprintf(title, sizeof(title), "CUDA Ray Tracer | %s | FPS: %6.0f | Frame: %6.2fms | GPU: %6.2fms", mode.useBVH ? "BVH" : "Brute", avgFps, avgTotalMs, avgGpuMs);
            glfwSetWindowTitle(win, title);

            statsTimer = 0.0f;
            frameCount = 0;
            accumulatedGpuMs = 0.0f;
            accumulatedTotalMs = 0.0f;
        }
    }

    // cleanup
    if (cudaPbo)
    {
        cudaGraphicsUnregisterResource(cudaPbo);
        cudaPbo = nullptr;
    }
    if (pbo)
    {
        glDeleteBuffers(1, &pbo);
        pbo = 0;
    }

    freeDevicePixels();

    glfwTerminate();
    return 0;
}
