#ifndef RAYTRACER_H
#define RAYTRACER_H

float launchRayTracer(void *hostPixels, int screenWidth, int screenHeight, bool useBVH);

void initDevicePixel(int screenWidth, int screenHeight);
void freeDevicePixels();

// either new performance test or regualr scene
void initScene(bool perfTest);

void resetAccumulation();

#endif
