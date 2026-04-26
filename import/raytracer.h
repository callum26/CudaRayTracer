#ifndef RAYTRACER_H
#define RAYTRACER_H

float launchRayTracer(void *hostPixels, int screenWidth, int screenHeight);

void initDevicePixel(int screenWidth, int screenHeight);
void freeDevicePixels();

void initScene();

#endif
