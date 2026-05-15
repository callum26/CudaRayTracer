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
    int perfTestGridSize;
};

struct CurrentMode
{
    bool useBVH;
    bool perfTest;
};

float launchRayTracer(void *hostPixels, SceneSettings settings, CurrentMode mode);

void initDevicePixel(SceneSettings settings);
void freeDevicePixels();

// either new performance test or regualr scene
void initScene(CurrentMode mode, SceneSettings settings);

void resetAccumulation(SceneSettings settings);

#endif
