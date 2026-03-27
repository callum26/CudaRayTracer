#include <cuda_runtime.h>
#include <cmath>
#include "raytracer.h"

const unsigned int screenWidth = 800;
const unsigned int screenHeight = 800;

// device pixels declared frist point to empty memory addressi in gpu
static unsigned char *devicePixels = nullptr;

// now we have all of the basis of the ray tracer working
// introducing structs to simplify the length of code
// used to store 3 coords 3d scructyuire
// common namming convench online Vector 3
// later may put something for used  the pixel datas
struct Vec3
{
    float x, y, z;

    // calculations for vector maths

    __device__ float dot(const Vec3 &otherVec3) const
    {
        return (x * otherVec3.x) + (y * otherVec3.y) + (z * otherVec3.z);
    }

    __device__ Vec3 sub(const Vec3 &otherVec3) const
    {
        return {x - otherVec3.x, y - otherVec3.y, z - otherVec3.z};
    }

    __device__ Vec3 add(const Vec3 &otherVec3) const
    {
        return {x + otherVec3.x, y + otherVec3.y, z + otherVec3.z};
    }

    __device__ Vec3 div(const Vec3 &otherVec3) const
    {
        return {x / otherVec3.x, y / otherVec3.y, z / otherVec3.z};
    }

    __device__ Vec3 multi(const Vec3 &otherVec3) const
    {
        return {x * otherVec3.x, y * otherVec3.y, z * otherVec3.z};
    }

    // multi by a scalar instead of vec3
    __device__ Vec3 scale(float scalar) const
    {
        return {x * scalar, y * scalar, z * scalar};
    }

    // to normalise the vector = u / magnitude of (u)

    // magnitude equation is given as = sqrt(x^2 + y^2 + z^2)
    // this function takes all xyz components of the current vec3 structure
    // dots them them square roots it
    // leaving the scalar value of magnitude
    __device__ float magnitude() const
    {
        return sqrtf(dot(*this));
    }

    // this is magnitude function is then executed by the normalise function
    // stored as a var to prevent reexecution and divides each of the xyz components by the scalar value to give a normalise vector
    // this now meanss that the vector has a magnitude of 1 and therefore its components are solely the direction they are facing
    // useful in ray tracing as for a lot of things we need the direction to where any given ray whether that be light or rays are travelling
    __device__ Vec3 normalise() const
    {
        float mag = magnitude();
        return {x / mag, y / mag, z / mag};
    }
};

// introducing material storing values of different objects
struct Material
{
    Vec3 colour;
    // ambient/diffuse/specular reflectivity
    float ambientReflectivity, diffuseReflectivity, specularReflectivity;
    // shiniess
    float a;
};

struct Light
{
    Vec3 pos;
    // ia
    Vec3 ambientIntensity;
    // imd
    Vec3 diffuseIntensity;
    // ims
    Vec3 specularIntensity;
    // use to control overall intensity of light
    float intensity;
};

struct Object
{
    Vec3 pos;
    Material mat;
};

__device__ void writePixels(unsigned char *pixels, int pixelIndex, Object object, Vec3 phongShading)
{
    pixels[pixelIndex + 0] = (unsigned char)((fminf(fmaxf(object.mat.colour.x * phongShading.x, 0.0f), 1.0f)) * 255.0f);
    pixels[pixelIndex + 1] = (unsigned char)((fminf(fmaxf(object.mat.colour.y * phongShading.y, 0.0f), 1.0f)) * 255.0f);
    pixels[pixelIndex + 2] = (unsigned char)((fminf(fmaxf(object.mat.colour.z * phongShading.z, 0.0f), 1.0f)) * 255.0f);
    pixels[pixelIndex + 3] = 255;
}

// from my reading on https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
// a ray can be defed as a ray(t) = Origin + t * Direction
// Origin is (camX, camY, camZ) and e Direction as (rayDirX, rayDirY, rayDirZ)
// t = distance from point of intersection to the origin of the ray
// goal is to find value of t (if intersects) use it to find the point of intersect with spehere
// then to shade that specific pixel based on point of intsect along with normal of surface at that point
// also taking account the position of the light source in order to calculate the lighting components and therrefore colour of pixel

__device__ bool raySphereIntersection(Vec3 camPos, Object sphere, float sphereRadius, Vec3 rayDir, float &sphereDistance)
{

    // implicit equation of a sphere with radius r centred at C = 0,0,0
    // is given by (x-x0)^2 + (y-y0)^2 + (z-z0)^2 = r^2
    // this means that any piont P  on the sphere must be the following
    // (P-C) . (P-C) = r^2

    // but in order to find the goal (t) (origin to distance from point of intersection) we need to find its intersection
    // to do this we must subsitue the point P into the equation of a ray equation to find the point at which it intersects the spehre
    // this makes our point P = Origin + t * Direction
    // now we have a new definition of P we can put that in the implicit equation of the spehere from above
    // (O + tD - C) . (O + tD - C) = r^2
    // befre expanding it to make it simplier we can rearrange and sub in a new var for the vars we already know (O and C)
    // (O - C + tD) . (O - C + tD) = r^2
    // ((as explained on scratchapixel)) O - C is the vector to get from sphere center to ray origin
    // in the doc they assign this to the var L hence L = O - C

    // slightly confusing bcos of the way the variables are named
    // O is the camPos
    // C is spherePos
    // camPos - spherepos gives us L aka the vectro from sphere center to orgin
    Vec3 sphereToOrigin = camPos.sub(sphere.pos);

    // equation now becomes (L + tD) . (L + tD) = r^2
    // = L . (L + tD) + tD . (L + tD)
    // = (L . L) + (L . tD) + (tD . L) + (tD . tD)
    // as t is a scalar dot product distributes seperately from the vecotrs we can remove them out with no issue
    // = (L . L) + t(L . D) + t(D . L) + t^2(D . D)
    // as t is multipled within the dot product of (D . D) twice we can use t^2
    // L . D = D. L meaning t(L . D) and t(D . L) are indentical so we can multi them by 2
    // = (L . L) + 2t(L . D) + t^2(D . D)
    // rearranging we get
    // t^2(D . D) + 2t(L . D) + (L . L) = r^2
    // we can then assign all 3 distinct dot products their own vars
    // a = D . D
    // b = L . D
    // c = L . L
    // giviing us
    // at^2 + 2bt + c = r^2

    // a = D . D
    // dot product of the ray direction with itself
    float a = rayDir.dot(rayDir);

    // b = L . D
    // dot product of the length of vector from origin to center with the rayDirection
    // multipy b by 2 explained below
    float b = 2.0f * sphereToOrigin.dot(rayDir);

    // c = L . L
    // dot product of the length of vector from origin to center with inself sub the sphere radius sphere to make sure its equal to zero explained below
    float c = sphereToOrigin.dot(sphereToOrigin) - (sphereRadius * sphereRadius);

    // obviously quadratic equation is given as
    // x = (-b (+/-) sqr(b^2 - 4ac)) / 2a

    // solving the discrimiant gives us the amount of solutions
    // b^2 - 4ac > 0 two real solutions
    // b^2 - 4ac = 0 one real solution
    // b^2 - 4ac < 0 two complex solutions
    // we are dealing with strictly real solutions we can rule out where the discriminant is less than 0

    // b^2 - 4ac > 0 two real solutions
    // b^2 - 4ac = 0 one real solution
    // either the ray is intersecting twice at b^2 - 4ac > 0
    // or its intersecting once b^2 - 4ac = 0
    // one intersection is less comon it would be arond clipping edges of sphere
    // two intersection pass throughs spehere coming out another side

    // discrim = b^2 - 4ac
    // at^2 + 2bt + c = r^2
    // quadratic equation ax^2 + bx + c = 0
    // at^2 + 2bt + (c - r^2) = 0
    // d = a
    // e = 2b
    // f = (c - r^2)
    // dt^2 + et + f = 0
    // now in quadratic form

    // solve for the discriminant
    // b^2 - 4 * a * c
    float discriminant = b * b - (4.0f * a * c);

    // mentioned above check for only real solutions either discrim is greater than 0 or exactly 0
    if (discriminant >= 0)
    {
        // (- b - sqrt(b^2-4ac)) / 2a for nearest intersection to the camera  enter spehere
        // (- b - sqrt(b^2-4ac)) / 2a for far intsect to cam  exit spehere
        float closeIntersection = (-b - sqrtf(discriminant)) / (2.0f * a);
        // float farIntersection = (-b + sqrtf(discriminant)) / (2.0f * a);
        // for now farIntersection isnt actually used but it could be if cam movement was added

        // as long as the intersection is infront of the camera then set the distance of this specific ray to the distance of the clostinerscection
        if (closeIntersection > 0.0f)
        {
            sphereDistance = closeIntersection;
            return true;
        }
    }
    return false;
}

__device__ bool rayPlaneIntersection(Vec3 camPos, Vec3 rayDir, float groundHeightY, float &groundDistance)
{
    if (abs(rayDir.y) < 0.0001f)
    {
        return false;
    }

    // for the ray plane it is much easier the intersection distance is simply
    // the distrance from the ground to the camera divided by the y of the ray direction
    float intersectionDistance = (groundHeightY - camPos.y) / rayDir.y;
    // make sure the intersection distance is pos we shouldnt worry about intersections behind the camera
    if (intersectionDistance > 0.0f)
    {
        // then we can set the ground distance to the intersection distance
        groundDistance = intersectionDistance;
        return true;
    }
    return false;
}

// its described as the following
// ks is the specular reflection const ratop pf reflection of speciular incoming light
// kd how stong surface scatters diffuse light
// ka is reflectiveness of the material to ambient light
// a is a shinness factor for the material larger surfaces that are smoother
// ia interensity of ambient lighting in the scene

// it also defines
// lights what are the set of all light soruces
// Lm is the direcitiopn vector from the point on the surface towards each light source m being to specificy the light source
// N is a normal at the point on the surface
// Rm is direction of a perfect reflected ray would take from the pooint on the surface
// V is the direction pointing towards the camera

// each point is given by the following equation of surface point Ip
// Ip = ka * ia + Sum (of all light soruces)(kd * (Lm . N)* im,d + ks * (Rm . V)^a * im,s))
// (ka * ia) is for the ambient light
// (kd * (Lm . N)* im,d) is for diffuse light
// (ks * (Rm . V)^a * im,s) is for specular
// we can similpfy them by assigning them variables to make the equation look easier
// A = ka * ia
// D = kd * (Lm . N)* im,d
// S = ks * (Rm . V)^a * im,s
// Ip = A + Sum (of all light soruces)(D + S)
__device__ Vec3 calculateAmbient(Light light, Object object)
{
    Vec3 ambientStrength = light.ambientIntensity.scale(object.mat.ambientReflectivity);
    return ambientStrength;
}

__device__ Vec3 calcLightDir(Vec3 hitPoint, Light light)
{

    // now we have the hit coords we can work out light distance and direction
    // lightToHit vector now contains distance and direction from hit point to light source
    Vec3 lightToHit = light.pos.sub(hitPoint);

    // we normalise the lightToHit vector to only have direction vecotr
    Vec3 lightDirection = lightToHit.normalise();

    return lightDirection;
}

__device__ Vec3 calcSurfaceNormal(Vec3 hitPoint, Object object)
{
    // to calculate N (surface normal) we calculate vector from center of sphere to hit point then normal it
    // again surfaceNormal currently is both direction and distance normalising will give us soley the direction
    Vec3 objectToHit = hitPoint.sub(object.pos);

    // normalise it in order to get the surface normal
    Vec3 surfaceNormal = objectToHit.normalise();

    return surfaceNormal;
}

__device__ Vec3 calculateDiffuse(Vec3 camPos, Vec3 rayDir, Light light, float distanceToObject, Object object)
{
    // D = kd * (Lm . N)* im,d
    Vec3 hitPoint = camPos.add(rayDir.scale(distanceToObject));

    Vec3 lightDir = calcLightDir(hitPoint, light);
    Vec3 surfaceNormal = calcSurfaceNormal(hitPoint, object);
    float lightDirDotSurfaceNormal = lightDir.dot(surfaceNormal);

    // (Lm. N) is calc in calclightDirDotSurfaceNormal
    // clamp any negative values are 0 bcos they woukd facing qwaay from light hence no lit
    float diffuseFactor = fmaxf(lightDirDotSurfaceNormal, 0.0f);

    // im,d represents light scatter in all dirs when a light source hits suface this must be done for all light source
    Vec3 imd = light.diffuseIntensity.scale(light.intensity);

    // we can now cal D
    // D = kd * (Lm . N) * im,d
    Vec3 diffuseStrength = imd.scale(object.mat.diffuseReflectivity * diffuseFactor);

    return diffuseStrength;
}

// was cuasingh issues using both the same equations for calculating the diffuse
// the surface normal cannot be calculated the same way as for the sphere it was sillyto d it
// this is temporary solution for now
/* COULD INSTEAD PASS IN THE NORMAL IN FUNC TO SAVE SPACE*/
__device__ Vec3 calculateGroundDiffuse(Vec3 camPos, Vec3 rayDir, Light light, float groundDistance, Object object)
{
    Vec3 hitPoint = camPos.add(rayDir.scale(groundDistance));

    Vec3 lightDirection = light.pos.sub(hitPoint).normalise();

    // no need to work out the ground normal as its the same everytime
    Vec3 groundNormal = {0.0f, 1.0f, 0.0f};
    float diffuseFactor = fmaxf(lightDirection.dot(groundNormal), 0.0f);

    // im,d represents light scatter in all dirs when a light source hits suface this must be done for all light source
    Vec3 imd = light.diffuseIntensity.scale(light.intensity);

    // we can now cal D
    // D = kd * (Lm . N) * im,d
    Vec3 diffuseStrength = imd.scale(object.mat.diffuseReflectivity * diffuseFactor);

    return diffuseStrength;
}

__device__ Vec3 calculateSpecular(Vec3 camPos, Vec3 rayDir, Light light, float distanceToObject, Object object)
{
    // S = ks * (Rm . V)^a * im,s
    // Rm is the direction of the reflect ray from the point on the surface occuring from light source m
    // we can calc this by reflecting the light direction across the surface normal
    // incident vector reflection form is
    // R = 2 * (L . N) * N - L

    // L is lightDirection
    // N is surfaceNormal

    // where the ray intersects the obj
    Vec3 hitPoint = camPos.add(rayDir.scale(distanceToObject));

    Vec3 lightDir = calcLightDir(hitPoint, light);
    Vec3 surfaceNormal = calcSurfaceNormal(hitPoint, object);

    float lightDirDotSurfaceNormal = lightDir.dot(surfaceNormal);

    // R = N * 2 * (L . N) - L
    Vec3 reflectDir = surfaceNormal.scale(2.0f * lightDirDotSurfaceNormal).sub(lightDir);

    // now we need V which is dir pointing towards cam from hit point
    Vec3 camToHit = camPos.sub(hitPoint);

    // normalise camToHit to get V direction
    Vec3 camToHitDirection = camToHit.normalise();

    // Rm and V we can calc (Rm . V)
    float reflectDotcamToHit = reflectDir.dot(camToHitDirection);
    reflectDotcamToHit = fmaxf(reflectDotcamToHit, 0.0f); // again like diffuse we clamp to 0

    // im,s is intesity of light scatter in all directions when the light hits surface for specular reflection
    Vec3 ims = light.specularIntensity.scale(light.intensity);

    // we can now cal S
    // S = ks * (Rm . V)^a * im,s`

    Vec3 specularStrength = ims.scale(object.mat.specularReflectivity * powf(reflectDotcamToHit, object.mat.a));
    return specularStrength;
}

__device__ bool isInShadow(Vec3 camPos, Vec3 rayDir, float distanceToObject, Light light, Object object)
{
    Vec3 hitPoint = camPos.add(rayDir.scale(distanceToObject));
    Vec3 lightDir = calcLightDir(hitPoint, light);
    Vec3 surfaceNormal = calcSurfaceNormal(hitpoint, object);

    Vec3 shadowOrigin = hitpoint.add(surfaceNormal);
    float shadowToLight = light.pos.sub(shadowOrigin).magnitude();

    float shadowSphereDistance;
    if (raySphereIntersection(shadowOrigin, sphere, sphereRadius, rayDir, shadowSphereDistance))
    {
        if (shadowDistance < shadowToLight)
            return true;
    }

    float shadowGroundDistance;
    if (rayPlaneIntersection(shadowOrigin, sphere, sphereRadius, rayDir, shadowGroundDistance))
    {
        if (shadowDistance < shadowToLight)
            return true;
    }

    return false;
}

// in order to shade the sphere we are using the phong shading model
// it combines three different terms to create realistic reflections
// this sis ambient, diffusal and specular
// ambient is the soft light that illuminates all parts of a surface regardless of direct light sourecs,
// diffusal simulates light scattering when striking a surface, a matte appearance depending on angle of light source and surface normal surfaces facing light appear bright than ones not facing
// specular uses bright highlights occuring when light reflects off smoth ir rough surface. a lot more dynamic than thte others
// for now only ambient and diffual are used specular is more cmoplex and will be done later
__device__ void shadeSphere(unsigned char *pixels, int pixelIndex, float sphereDistance, Vec3 rayDir, Vec3 camPos, Light light, Object sphere)
{
    Vec3 hitPoint = camPos.add(rayDir.scale(distanceToObject));
    Vec3 surfaceNormal = calcSurfaceNormal(hitpoint, object);

    if (isInShadow(camPos, rayDir, distanceToObject, light, object))
    {
        // if its in the shadow only use ambient
        Vec3 ambientStrength = calculateAmbient(light, sphere);
        writePixels(pixels, pixelIndex, sphere, ambientStrength)
    }
    else
    {
        Vec3 ambientStrength = calculateAmbient(light, sphere);
        Vec3 diffuseStrength = calculateDiffuse(camPos, rayDir, light, sphereDistance, sphere);
        Vec3 specularStrength = calculateSpecular(camPos, rayDir, light, sphereDistance, sphere);

        // Ip = A + D + S
        Vec3 phongShading = {
            fminf((ambientStrength.x + diffuseStrength.x + specularStrength.x), 1.0f),
            fminf((ambientStrength.y + diffuseStrength.y + specularStrength.y), 1.0f),
            fminf((ambientStrength.z + diffuseStrength.z + specularStrength.z), 1.0f)};

        writePixels(pixels, pixelIndex, sphere, phongShading);
    }
}

// shading of the ground is similar to sphere
// we dont really need specular for the ground as its a matte surface
__device__ void shadeGround(unsigned char *pixels, int pixelIndex, float groundDistance, Vec3 rayDir, Vec3 camPos, Light light, Object ground)
{

    // implentning checkboard pattern to show off ground more clearly

    float tileSize = 1.0f;
    // had to introduced an offset for the tiles as they were mirrored centring coming from the middle
    // meaning the there was two of the same tiles in the middle
    float checkerOffsetX = 0.5f;
    float checkerOffsetZ = 0.5f;

    Vec3 hitPoint = camPos.add(rayDir.scale(groundDistance));
    // using the hit point coords we can determine which tile we are on diving hit point by tile size
    // creating an int for the tiles in x and z axis as the ground is flat on the xz plane
    // even or odd tiles will be different colours to create a pattern

    // solved the issue by adding an offset 1/2 a tile
    // changed it to floor instead to round down
    int checkX = floor((hitPoint.x + checkerOffsetX) / tileSize);
    int checkZ = floor((hitPoint.z + checkerOffsetZ) / tileSize);

    // if both are even or both are odd we make one colour otherwise we make the other colour

    if ((checkX % 2 == 0 && checkZ % 2 == 0) || (checkX % 2 != 0 && checkZ % 2 != 0))
    {
        // just liek sphjerw shade xyz repsect rgb
        ground.mat.colour = {0.3f, 0.3f, 0.3f};
    }
    else
    {
        ground.mat.colour = {0.7f, 0.7f, 0.7f};
    }

    Vec3 ambientStrength = calculateAmbient(light, ground);
    Vec3 diffuseStrength = calculateGroundDiffuse(camPos, rayDir, light, groundDistance, ground);

    // as we ignore specular
    // the equation simplifies to
    // Ip = A + D
    Vec3 phongShading = {
        fminf((ambientStrength.x + diffuseStrength.x), 1.0f),
        fminf((ambientStrength.y + diffuseStrength.y), 1.0f),
        fminf((ambientStrength.z + diffuseStrength.z), 1.0f)};

    writePixels(pixels, pixelIndex, ground, phongShading);
}

// for the background gradient based on the ray direction

__device__ void shadeBackground(unsigned char *pixels, int pixelIndex, Vec3 rayDir, Object background)
{
    // we can use the y of the ray direction to determine how much of the background colour to show
    // this works bcos y is btween -1 and 1 when normalised
    // rayDirY -1 its pointing down towards the ground its 1 pointing up towards the sky
    Vec3 backgroundDiffuse = {
        0.5f * (rayDir.x + 1),
        0.5f * (rayDir.y + 1),
        0.5f * (rayDir.z + 1),
    };

    writePixels(pixels, pixelIndex, background, backgroundDiffuse);
}

__global__ void renderKernel(unsigned char *pixels, int screenWidth, int screenHeight)
{
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    float viewPortHeight = 2.0f;
    float viewPortWidth = ((float)screenWidth / (float)screenHeight) * viewPortHeight;

    if (pixelX >= screenWidth || pixelY >= screenHeight)
        return;

    Vec3 camPos = {0.0f, 0.0f, 0.0f};

    Vec3 backgroundRGB = {0.1f, 0.1f, 0.1f};

    Light light = {
        {3.0f, 3.0f, 3.0f},    // pos
        {0.25f, 0.25f, 0.25f}, // ambientIntensity
        {0.8f, 0.8f, 0.8f},    // diffuseIntensity
        {0.6f, 0.6f, 0.6f},    // specularIntensity
        1.0f                   // intensity
    };

    Object sphere = {
        {0.0f, 0.0f, -3.0f}, // pos
        {
            {0.2f, 0.5f, 1.0f}, // colour
            0.2f,               // ambientReflectivity
            0.7f,               // diffuseReflectivity
            0.5f,               // specularReflectivity
            24.0f               // shininess
        }};

    Object ground = {
        {0.0f, -2.0f, 0.0f}, // pos
        {
            {0.2f, 0.2f, 0.2f}, // colour
            0.2f,               // ambientReflectivity
            0.6f,               // diffuseReflectivity
            0.5f,               // specularReflectivity
            24.0f               // shininess
        }};

    Object background = {
        {0.0f, 0.0f, 0.0f}, // pos // irrelasvant
        {
            {0.6f, 0.6f, 0.6f}, // colour
            0.2f,               // ambientReflectivity
            0.7f,               // diffuseReflectivity
            0.5f,               // specularReflectivity
            24.0f               // shininess
        }};

    float sphereRadius = 1.5f;
    float groundNormalY = 1.0f;

    // map pixel coords between -0.5/0.5 for x and 0.5/-0.5 Y
    float normalX = ((float)pixelX / (screenWidth - 1)) - 0.5f;
    float normalY = 0.5f - ((float)pixelY / (screenHeight - 1));

    // these are then scaled by view port to get correct aspect ratio
    Vec3 rayDir = {normalX * viewPortWidth, normalY * viewPortHeight, -1.0f};

    // normalise the ray direction
    rayDir = rayDir.normalise();

    float sphereDistance = INFINITY;
    bool hitSphere = raySphereIntersection(camPos, sphere, sphereRadius, rayDir, sphereDistance);

    float groundDistance = INFINITY;
    bool hitGround = rayPlaneIntersection(camPos, rayDir, ground.pos.y, groundDistance);

    float shadowDistance;
    bool isInShadow = raySphereIntersection(shadowOrigin, sphere, sphereRadius, light, shadowDistance);

    int writeRow = (screenHeight - 1 - pixelY);
    int pixelIndex = (writeRow * screenWidth + pixelX) * 4;

    // hitSphere and hitGround are bools for determining whether a specifced ray hit the objects
    // they both also update their value for their respective distances whenever they run and return true
    if (hitSphere && (!hitGround || sphereDistance < groundDistance))
    {

        shadeSphere(pixels, pixelIndex, sphereDistance, rayDir, camPos, light, sphere);
    }
    else if (hitGround)
    {
        shadeGround(pixels, pixelIndex, groundDistance, rayDir, camPos, light, ground);
    }
    else
    {
        shadeBackground(pixels, pixelIndex, rayDir, background);
    }
}

// these functions now help avoid mem being allocated everyframe
// mem init at the start before launchraytracer loop is executed with main.cpp
// and cleared when it ends
void initDevicePixel(int screenWidth, int screenHeight)
{
    // cuda malloc takes address of devicepixels storing the size needed as W * H * 4 as RGBA of each pixel
    cudaMalloc(&devicePixels, screenWidth * screenHeight * 4);
}

void freeDevicePixels()
{
    // fres up the g
    cudaFree(devicePixels);
}

float launchRayTracer(unsigned char *hostPixels, int screenWidth, int screenHeight)
{
    // going to start implementation of performance stats
    // https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
    // as mentioned on the nvidia blog its better to use the inbuilt functions for timings in cuda instead of
    // cpu timings
    // the way on the blog is the best way to go about it

    // starting both the cuda events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 256 threads per block (16x16)
    dim3 blockSize(16, 16);
    dim3 gridSize((screenWidth + blockSize.x - 1) / blockSize.x, (screenHeight + blockSize.y - 1) / blockSize.y);

    // begins once the kernel is launch
    cudaEventRecord(start);
    renderKernel<<<gridSize, blockSize>>>(devicePixels, screenWidth, screenHeight);
    cudaEventRecord(stop);
    // stops and fills records once its finished

    cudaDeviceSynchronize();

    cudaMemcpy(hostPixels, devicePixels, screenWidth * screenHeight * 4, cudaMemcpyDeviceToHost);

    // forcing cpu to halt until gpu finishes
    cudaEventSynchronize(stop);
    float ms = 0;
    // calcs the difference
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}
