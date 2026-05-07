#ifndef RAYTRACER_H
#define RAYTRACER_H

float launchRayTracer(void *hostPixels, int screenWidth, int screenHeight, bool useBVH);

void initDevicePixel(int screenWidth, int screenHeight);
void freeDevicePixels();

void initScene();

void resetAccumulation();

#endif
