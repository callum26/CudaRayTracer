#include <string>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fstream>
#include <sstream>
#include <raytracer.h>

// seperated the shader read file compiling and all other host cpp code into a seprate file was causing many issues when trying to boot the ray tracer
// seperating the code makes it much easier and probably i think more efficient as the cuda compiler was having to deal with
// both of the logic causing slow downs

const unsigned int screenWidth = 800;
const unsigned int screenHeight = 800;

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

    unsigned char *hostPixels = new unsigned char[screenWidth * screenHeight * 4];
    initDevicePixel(screenWidth, screenHeight);

    while (!glfwWindowShouldClose(win))
    {
        float frameStart = glfwGetTime();

        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(win, true);

        // clear screen
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // instead of only launching void function it now returns ms time for frames
        float gpuMs = launchRayTracer(hostPixels, screenWidth, screenHeight);

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

        float frameEnd = glfwGetTime();

        // full frame time include GPU processing ime
        float totalMs = (float)((frameEnd - frameStart) * 1000.0f);
        float fps = 1000.0f / totalMs;

        char title[64];
        printf(title, "CUDA Ray Tracer | FPS: %", fps);
        glfwSetWindowTitle(win, title);
    }

    // cleanup
    delete[] hostPixels;
    freeDevicePixels();

    glfwTerminate();
    return 0;
}