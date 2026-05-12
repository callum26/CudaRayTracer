#include <string>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fstream>
#include <cstdio>
#include <sstream>
#include "raytracer.h"
#include <vector>
#include <iomanip>

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

    SceneSettings initSettings = data[0].settings;
    CurrentMode initMode = data[0].mode;
    file << "maxBounces,maxShadowBounces,samplesPerPixel,lightSamples,screenWidth,screenHeight,useBVH,perfTest\n";
    file << initSettings.maxBounces << "," << initSettings.maxShadowBounces << "," << initSettings.samplesPerPixel
         << "," << initSettings.lightSamples << "," << initSettings.screenWidth << "," << initSettings.screenHeight
         << "," << (initMode.useBVH ? 1 : 0) << "," << (initMode.perfTest ? 1 : 0) << "\n";

    file << "frameIndex,fps, gpuMs,totalMs\n";
    file << std::fixed << std::setprecision(2);

    for (const auto &f : data)
    {
        file << f.index << "," << f.currentFps << "," << f.gpuMs << "," << f.totalMs << "," << "\n";
    }

    printf("Exported 500 frames to %s\n", name.c_str());
}

const unsigned int screenWidth = 800;
const unsigned int screenHeight = 800;

// seperated the shader read file compiling and all other host cpp code into a seprate file was causing many issues when trying to boot the ray tracer
// seperating the code makes it much easier and probably i think more efficient as the cuda compiler was having to deal with
// both of the logic causing slow downs
// theses lines are only temporary was just trying to get it running
// will change later to my code
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

int main()
{
    if (!glfwInit())
        return -1;

    // set opengl version 3.3 core
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *win = glfwCreateWindow(screenWidth, screenHeight, "CUDA Ray Tracer", NULL, NULL);

    if (!win)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(win);

    // turn off vsync
    glfwSwapInterval(0);

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

    // move the calculation of host pixel buffer solely with cpp
    SceneSettings settings = {
        15,  // maxBounces
        8,   // maxShadowBounces
        1,   // samplesPerPixel
        1,   // lightSamples
        800, // screenWidth
        800  // screenHeight
    };

    CurrentMode mode = {
        true, // useBVH
        false // perfTest
    };

    const int screenWidth = settings.screenWidth;
    const int screenHeight = settings.screenHeight;

    unsigned char *hostPixels = new unsigned char[screenWidth * screenHeight * 4];

    initDevicePixel(settings);
    initScene(mode, settings);

    bool bKeyPressed = false;
    bool tKeyPressed = false;
    float statsTimer = 0.0f;
    float frameCount = 0;
    float accumulatedGpuMs = 0.0f;
    float accumulatedTotalMs = 0.0f;
    float avgFps = 0.0f;

    std::vector<FrameData> benchmarkData;
    benchmarkData.reserve(benchmarkFrameCount);

    bool benchmarkFinished = false;

    while (!glfwWindowShouldClose(win))
    {
        float frameStart = glfwGetTime();

        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(win, true);

        bool bKeyDown = glfwGetKey(win, GLFW_KEY_B) == GLFW_PRESS;
        if (bKeyDown && !bKeyPressed)
        {
            mode.useBVH = !mode.useBVH;
            resetAccumulation(settings);
        }
        bKeyPressed = bKeyDown;

        bool tKeyDown = glfwGetKey(win, GLFW_KEY_T) == GLFW_PRESS;
        if (tKeyDown && !tKeyPressed)
        {
            mode.perfTest = !mode.perfTest;
            initScene(mode, settings);
            resetAccumulation(settings);
        }
        tKeyPressed = tKeyDown;

        // clear screen
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // instead of only launching void function it now returns ms time for frames
        float gpuMs = launchRayTracer(hostPixels, settings, mode);

        // upload pixel data to texture took this from opengl graphics project in year2
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, screenWidth, screenHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, hostPixels);

        // draw textured quad
        glUseProgram(prog);
        glBindVertexArray(VAO);
        glBindTexture(GL_TEXTURE_2D, tex);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // swap buffers and poll events
        glfwSwapBuffers(win);
        glfwPollEvents();

        float totalMs = (float)((glfwGetTime() - frameStart) * 1000.0f);

        accumulatedGpuMs += gpuMs;
        accumulatedTotalMs += totalMs;
        frameCount++;
        statsTimer += totalMs / 1000.0f;
        float currentFps = 1000.0f / (accumulatedTotalMs / frameCount);

        if (!benchmarkFinished)
        {
            benchmarkData.push_back({(int)benchmarkData.size(), currentFps, gpuMs, totalMs, settings, mode});
            if (benchmarkData.size() >= benchmarkFrameCount)
            {
                exportToCsv("frame_metrics.csv", benchmarkData);
                benchmarkFinished = true;
            }
        }

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
    delete[] hostPixels;
    freeDevicePixels();

    glfwTerminate();
    return 0;
}