#ifndef RAYTRACER_H
#define RAYTRACER_H

struct SceneSettings
{
    int maxBounces;
    int maxShadowBounces;
    int samplesPerPixel;
    int lightSamples;
    int screenWidth;
    int screenHeight;
};

struct CurrentMode
{
    bool useBVH;
    bool perfTest;
};

float launchRayTracer(void *hostPixels, SceneSettings settings, CurrentMode mode);

void initDevicePixel(int screenWidth, int screenHeight);
void freeDevicePixels();

// either new performance test or regualr scene
void initScene(bool perfTest);

void resetAccumulation();

#endif
