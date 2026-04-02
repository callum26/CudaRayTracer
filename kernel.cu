#include <cuda_runtime.h>
#include <cmath>
#include "raytracer.h"

const unsigned int screenWidth = 800;
const unsigned int screenHeight = 800;

// device pixels declared frist point to empty memory additionressi in gpu
static uchar4 *devicePixels = nullptr;

// stores a 3D vector with coords xyz 
// vital part of the whole program
struct Vec3{
    float x, y, z;

    // calculations for commonly needed vector maths
    // otherVec3 passed in via reference to prevent not needed copies
    // const so it doesnt get changed while calcs r running
    // trailing const to not modifiy obj called
    __device__ Vec3 subtract(const Vec3 &otherVec3) const
    {
        return {x - otherVec3.x, y - otherVec3.y, z - otherVec3.z};
    }

    __device__ Vec3 addition(const Vec3 &otherVec3) const
    {
        return {x + otherVec3.x, y + otherVec3.y, z + otherVec3.z};
    }

    __device__ Vec3 divide(const Vec3 &otherVec3) const
    {
        return {x / otherVec3.x, y / otherVec3.y, z / otherVec3.z};
    }

    __device__ Vec3 multiply(const Vec3 &otherVec3) const
    {
        return {x * otherVec3.x, y * otherVec3.y, z * otherVec3.z};
    }

    // more complex vector maths

    // a dotProduct b = a.x * b.x + a.y * b.y + a.z * b.z
    __device__ float dotProduct(const Vec3 &otherVec3) const
    {
        return (x * otherVec3.x) + (y * otherVec3.y) + (z * otherVec3.z);
    }

    // a(vector) * b(scalar) = a.x * b, a.y * b, a.z * b
    __device__ Vec3 scale(float scalar) const
    {
        return {x * scalar, y * scalar, z * scalar};
    }

    // normalise = u / magnitude of (u)

    // magnitude equation is given as = sqrt(x^2 + y^2 + z^2)
    // this takes all xyz components of vec3 passed in
    // dots them with themself return mag as result
    __device__ float magnitude() const
    {
        return sqrtf(dotProduct(*this));
    }

    // normalinng making vector have a magnitude of 1 (unit vector) (for not 0 vectors) therefore its components r direction only
    // direction is needed a lot regardless of magnitude so its better to normalise them then use for calculations
    __device__ Vec3 normalise() const
    {
        float mag = magnitude();
        if (mag != 0){
        return {x / mag, y / mag, z / mag};
        }
        return {0.0f, 0.0f, 0.0f};
    }
};

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
    Vec3 position;
    // ia, imd, ims
    Vec3 ambientIntensity; Vec3 diffuseIntensity; Vec3 specularIntensity;
    float lightIntensity;
};

// read about online find the source i think it was a yt video
// seperating objects into types
enum ObjectType{
    sphereObject,
    groundObject,
};

/* CHANGE RADIUS AND NORMAL SO ITS SEPERATE MAYBE SPLIT OBJECT INTO SPHERE GROUND STRUCTS idk*/
struct Object
{
    Vec3 position;
    Material material;
    ObjectType type;

    float radius; 
    Vec3 normal; 
    
};

struct Ray{
    Vec3 origin;
    Vec3 direction;
    Vec3 hitPoint;
};

// consts for gpu prevents recopying 
/* MAYBE BETTER WAY TO DO THIS?*/
__constant__ Light light;
__constant__ Object spheres[2];
__constant__ Object ground;
__constant__ Object background;


// from my reading on https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
// ray defoed as ray(t) = Origin + t * Direction
// Origin (ray.origin)
// Direction (ray.direction)
// t = distance from point of intersection to origin of the ray
// find value of t (if intersects) use it to find the point of intersect with spehere
/* MIGHT SPLIT THIS BACK UP INTO TWO EQUATIONS USING IFS IS INEFFICIENT*/
__device__ bool rayIntersect(Ray ray, Object object, float &distance){
    if (object.type == sphereObject){

    // implicit equation of sphere w/ radius r centred at C = 0,0,0
    // given by (x-x0)^2 + (y-y0)^2 + (z-z0)^2 = r^2
    // must mean any point P must satisfy
    // (P-C) . (P-C) = r^2
    // we have C and r we need to calc P 

    // to find t (intersection to origin of the ray) we need to find its intersection (point P)
    // if we subsitute our ray equation into point P we can work its intersection
    // P = Origin + t * Direction lets saying P = O + tD
    // putting it into our implicit equation of sphere
    // (O + tD - C) . (O + tD - C) = r^2
    // to reduce rearrange grouping togther vars we already have O and C and we also have r
    // (O - C + tD) . (O - C + tD) = r^2
    // O - C is the vector to get from sphere center to ray origin
    // in doc they refer to it as L = O - C 

    // L = ray.origin - object.position 
    Vec3 sphereToOrigin = ray.origin.subtract(object.position);
    
    // subsitute into equation 
    // (L + tD) . (L + tD) = r^2
    // expand out 
    // L . (L + tD) + tD . (L + tD)
    // (L . L) + (L . tD) + (tD . L) + (tD . tD)
    // t (scalar) dot product distros seperately from vecotrs remove them out with no issue
    // (L . L) + t(L . D) + t(D . L) + t^2(D . D)
    // group identical factors
    // = (L . L) + 2t(L . D) + t^2(D . D)
    // rearranging in power order
    // t^2(D . D) + 2t(L . D) + (L . L) = r^2
    // assign all to our vars
    // a = D . D
    // b = L . D
    // c = L . L
    // subsitute in
    // at^2 + 2bt + c = r^2
    // exactly like quadratic equation ax^2 + bx + c = 0 we want it equal to 0 subtract only none t component c by r^2
    // c = L . L - r^2
    // at^2 + 2bt + c = 0
    

    // a = D . D
    // dot ray direction with itself
    float a = ray.direction.dotProduct(ray.direction);

    // b = L . D
    // dot sphereToOrigin with ray.direction
    // multiplypy b by 2 explained below
    float b = 2.0f * sphereToOrigin.dotProduct(ray.direction);

    // c = L . L - r^2
    // dot sphereToOrigin with itself subtract sphere radius sphere to make sure its equal to zero explained below
    float c = sphereToOrigin.dotProduct(sphereToOrigin) - (object.radius * object.radius);

    // x = (-b (+/-) sqr(b^2 - 4ac)) / 2a

    // solving the discrimiant gives us the amount of solutions ignore complex solutions
    // b^2 - 4ac > 0 two real solutions
    // b^2 - 4ac = 0 one real solution

    // either ray is intersecting twice at b^2 - 4ac > 0
    // or intersecting once b^2 - 4ac = 0
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
        // for now farIntersection isnt actually used but it could be if cam movement was additioned

        // as long as the intersection is infront of the camera then set the distance of this specific ray to the distance of the clostinerscection
        if (closeIntersection > 0.001f)
        {
            distance = closeIntersection;
            return true;
        }
    }
        return false;
    
    } 
    else if (object.type == groundObject){
        if (fabsf(ray.direction.y) < 0.0001f)
    {
        return false;
    }

    // for the ray plane it is much easier the intersection distance is simply
    // the distrance from the ground to the camera divideided by the y of the ray direction
    float intersectionDistance = (object.position.y - ray.origin.y) / ray.direction.y;
    // make sure the intersection distance is position we shouldnt worry about intersections behind the camera
    if (intersectionDistance > 0.001f)
    {
        // then we can set the ground distance to the intersection distance
        distance = intersectionDistance;
        return true;
    }
    return false;
    }
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
    Vec3 ambientStrength = light.ambientIntensity.scale(object.material.ambientReflectivity);
    return ambientStrength;
}

__device__ Vec3 calcLightDir(Vec3 hitPoint, Light light)
{

    // now we have the hit coords we can work out light distance and direction
    // lightToHit vector now contains distance and direction from hit point to light source
    Vec3 lightToHit = light.position.subtract(hitPoint);

    // we normalise the lightToHit vector to only have direction vecotr
    Vec3 lightDirection = lightToHit.normalise();

    return lightDirection;
}

__device__ Vec3 calcSurfaceNormal(Vec3 hitPoint, Object object)
{
    // to calculate N (surface normal) we calculate vector from center of sphere to hit point then normal it
    // again surfaceNormal currently is both direction and distance normalising will give us soley the direction
    Vec3 objectToHit = hitPoint.subtract(object.position);

    // normalise it in order to get the surface normal
    Vec3 surfaceNormal = objectToHit.normalise();

    // pass in const of groundobject
    if (object.type == groundObject){
        return ground.normal;
    } else if (object.type == sphereObject){
        return surfaceNormal;
    }
}

__device__ Vec3 calculateDiffuse(Ray ray, Light light, float distanceToObject, Object object)
{
    // D = kd * (Lm . N)* im,d
    Vec3 hitPoint = ray.origin.addition(ray.direction.scale(distanceToObject));

    Vec3 lightDir = calcLightDir(hitPoint, light);
    Vec3 surfaceNormal = calcSurfaceNormal(hitPoint, object);

    float lightDirdotProductSurfaceNormal = lightDir.dotProduct(surfaceNormal);

    // (Lm. N) is calc in calclightDirdotProductSurfaceNormal
    // clamp any negative values are 0 bcos they woukd facing qwaay from light hence no lit
    float diffuseFactor = fmaxf(lightDirdotProductSurfaceNormal, 0.0f);

    // im,d represents light scatter in all dirs when a light source hits suface this must be done for all light source
    Vec3 imd = light.diffuseIntensity.scale(light.lightIntensity);

    // we can now cal D
    // D = kd * (Lm . N) * im,d
    Vec3 diffuseStrength = imd.scale(object.material.diffuseReflectivity * diffuseFactor);

    return diffuseStrength;
}

__device__ Vec3 calculateSpecular(Ray ray, Light light, float distanceToObject, Object object)
{
    // S = ks * (Rm . V)^a * im,s
    // Rm is the direction of the reflect ray from the point on the surface occuring from light source m
    // we can calc this by reflecting the light direction across the surface normal
    // incident vector reflection form is
    // R = 2 * (L . N) * N - L

    // L is lightDirection
    // N is surfaceNormal

    // where the ray intersects the obj
    Vec3 hitPoint = ray.origin.addition(ray.direction.scale(distanceToObject));

    Vec3 lightDir = calcLightDir(hitPoint, light);
    Vec3 surfaceNormal = calcSurfaceNormal(hitPoint, object);

    float lightDirdotProductSurfaceNormal = lightDir.dotProduct(surfaceNormal);

    // R = N * 2 * (L . N) - L
    Vec3 reflectDir = surfaceNormal.scale(2.0f * lightDirdotProductSurfaceNormal).subtract(lightDir);

    // now we need V which is dir pointing towards cam from hit point
    Vec3 camToHit = ray.origin.subtract(hitPoint);

    // normalise camToHit to get V direction
    Vec3 camToHitDirection = camToHit.normalise();

    // Rm and V we can calc (Rm . V)
    float reflectdotProductcamToHit = reflectDir.dotProduct(camToHitDirection);
    reflectdotProductcamToHit = fmaxf(reflectdotProductcamToHit, 0.0f); // again like diffuse we clamp to 0

    // im,s is intesity of light scatter in all directions when the light hits surface for specular reflection
    Vec3 ims = light.specularIntensity.scale(light.lightIntensity);

    // we can now cal S
    // S = ks * (Rm . V)^a * im,s`

    Vec3 specularStrength = ims.scale(object.material.specularReflectivity * powf(reflectdotProductcamToHit, object.material.a));
    return specularStrength;
}

__device__ bool isInShadow(Vec3 hitPoint, Vec3 surfaceNormal, Light light)
{
    // passing in hitpoint surfacenormal easier
    Vec3 lightDir = calcLightDir(hitPoint, light);
    
    // origin of the shadow is technically just hitpoint as thats where the light strikes the surface
    // it is common practise to adjust the origin just sligthly to prevent artifcats
    Vec3 shadowOrigin = hitPoint.addition(surfaceNormal.scale(0.001f));

    // distance from the shadow ray to the light source
    float shadowDistanceToLight = (light.position.subtract(shadowOrigin)).magnitude();

    float shadowSphereDistance;
    // reruns both intersection maths with shadow orgin instead and also initing a new shadowDistance var
    // which will be return and compared to shadowDistanceToLight
    // if shadowDistanceToLight greater than the intersection distance (shadowSphereDistance)
    // then it must mean its in shaded area

    // init shadow ray
    Ray shadowRay = {shadowOrigin, lightDir, {0.0f, 0.0f, 0.0f}};

    // loop through all speheres 
    // change this to introduce spherenum for scability
    for (int i = 0; i < 2; i++){

    float shadowSphereDistance;
    if (rayIntersect(shadowRay, spheres[i], shadowSphereDistance))
    {
        if (shadowSphereDistance < shadowDistanceToLight)
            return true;
    }
    }

    float shadowGroundDistance;
    if (rayIntersect(shadowRay, ground, shadowGroundDistance))
    {
        if (shadowGroundDistance < shadowDistanceToLight)
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
__device__ Vec3 shadeSphere(float sphereDistance, Ray ray, Light light, Object sphere)
{
    Vec3 hitPoint = ray.origin.addition(ray.direction.scale(sphereDistance));
    Vec3 surfaceNormal = calcSurfaceNormal(hitPoint, sphere);

    if (isInShadow(hitPoint, surfaceNormal, light))
    {
        // if its in the shadow only use ambient
        return calculateAmbient(light, sphere).multiply(sphere.material.colour);
    }
    else
    {
        Vec3 ambientStrength = calculateAmbient(light, sphere);
        Vec3 diffuseStrength = calculateDiffuse(ray, light, sphereDistance, sphere);
        Vec3 specularStrength = calculateSpecular(ray, light, sphereDistance, sphere);

        // Ip = A + D + S
        Vec3 phongShading = {
            fminf((ambientStrength.x + diffuseStrength.x + specularStrength.x), 1.0f),
            fminf((ambientStrength.y + diffuseStrength.y + specularStrength.y), 1.0f),
            fminf((ambientStrength.z + diffuseStrength.z + specularStrength.z), 1.0f)};

        return phongShading.multiply(sphere.material.colour);
    }
}

// shading of the ground is similar to sphere
// we dont really need specular for the ground as its a matte surface
__device__ Vec3 shadeGround(float groundDistance, Ray ray, Light light, Object ground)
{
    // implentning checkboard pattern to show off ground more clearly

    float tileSize = 1.0f;
    // had to introduced an offset for the tiles as they were mirrored centring coming from the middle
    // meaning the there was two of the same tiles in the middle
    float checkerOffsetX = 0.5f;
    float checkerOffsetZ = 0.5f;

    Vec3 hitPoint = ray.origin.addition(ray.direction.scale(groundDistance));
    // using the hit point coords we can determine which tile we are on divideing hit point by tile size
    // creating an int for the tiles in x and z axis as the ground is flat on the xz plane
    // even or odd tiles will be different colours to create a pattern

    // solved the issue by additioning an offset 1/2 a tile
    // changed it to floor instead to round down
    int checkX = floor((hitPoint.x + checkerOffsetX) / tileSize);
    int checkZ = floor((hitPoint.z + checkerOffsetZ) / tileSize);

    // if both are even or both are odd we make one colour otherwise we make the other colour

    if ((checkX % 2 == 0 && checkZ % 2 == 0) || (checkX % 2 != 0 && checkZ % 2 != 0))
    {
        // just liek sphjerw shade xyz repsect rgb
        ground.material.colour = {0.3f, 0.3f, 0.3f};
    }
    else
    {
        ground.material.colour = {0.7f, 0.7f, 0.7f};
    }

    
    Vec3 surfaceNormal = calcSurfaceNormal(hitPoint, ground);

    if (isInShadow(hitPoint, surfaceNormal, light))
    {
        // if its in the shadow only use ambient`
        return calculateAmbient(light, ground).multiply(ground.material.colour);
    }
    else
    {
        Vec3 ambientStrength = calculateAmbient(light, ground);
        Vec3 diffuseStrength = calculateDiffuse(ray, light, groundDistance, ground);

        // Ip = A + D + S
        Vec3 phongShading = {
            fminf((ambientStrength.x + diffuseStrength.x), 1.0f),
            fminf((ambientStrength.y + diffuseStrength.y), 1.0f),
            fminf((ambientStrength.z + diffuseStrength.z), 1.0f)
        };

        return phongShading.multiply(ground.material.colour);
    }
}

// for the background gradient based on the ray direction

__device__ Vec3 shadeBackground(Ray ray, Object background)
{
    // we can use the y of the ray direction to determine how much of the background colour to show
    // this works bcos y is btween -1 and 1 when normalised
    // directionY -1 its pointing down towards the ground its 1 pointing up towards the sky
    float positionition = 0.5f * (ray.direction.y + 1.0f);

    // gonna do blue white gradient
    Vec3 white = {1.0f, 1.0f, 1.0f};
    Vec3 blue = {0.1f, 0.1f, 1.0f};
    
    Vec3 gradient = {
        (1.0f - positionition) * white.x + positionition * blue.x,
        (1.0f - positionition) * white.y + positionition * blue.y,
        (1.0f - positionition) * white.z + positionition * blue.z
    };

    return gradient.multiply(background.material.colour);
}

__global__ void renderKernel(uchar4 *pixels, int screenWidth, int screenHeight)
{
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    float viewPortHeight = 2.0f;
    float viewPortWidth = ((float)screenWidth / (float)screenHeight) * viewPortHeight;

    if (pixelX >= screenWidth || pixelY >= screenHeight)
        return;
    

    // map pixel coords between -0.5/0.5 for x and 0.5/-0.5 Y
    float normalX = ((float)pixelX / (screenWidth - 1)) - 0.5f;
    float normalY = 0.5f - ((float)pixelY / (screenHeight - 1));

    // these are then scaled by view port to get correct aspect ratio
    Vec3 direction = {normalX * viewPortWidth, normalY * viewPortHeight, -1.0f};
    // normalise the ray direction
    direction = direction.normalise();

    // init hitPoint for now but wont be used till later
    Vec3 hitPoint = {0.0f, 0.0f, 0.0f};

    Ray ray = {
        {0.0f, 0.0f, 0.0f},
        direction,
        hitPoint

    };

    int writeRow = (screenHeight - 1 - pixelY);
    int pixelIndex = (writeRow * screenWidth + pixelX);

    // additioning reflection rays
    // every time ray bounces its strength is reduced
    Vec3 pixelColour = {0.0f, 0.0f, 0.0f};
    Vec3 strengthOfRay = {1.0f, 1.0f, 1.0f};
    int maxBounce = 3;

    for (int i = 0; i < maxBounce; i++){

    float sphereDistance = INFINITY;
    int hitIndex = -1;

    for (int s = 0; s < 2; s++){
        float closestSphereDistance = INFINITY;
        if (rayIntersect(ray, spheres[s], closestSphereDistance)){
            if (closestSphereDistance < sphereDistance){
                sphereDistance = closestSphereDistance;
                hitIndex = s;
            }
        }
    }
    bool hitSphere = (hitIndex != -1);

    float groundDistance = INFINITY;
    bool hitGround = rayIntersect(ray, ground, groundDistance);

    // hitSphere and hitGround are bools for determining whether a specifced ray hit the objects
    // they both also update their value for their respective distances whenever they run and return true
    if (hitSphere && (!hitGround || sphereDistance < groundDistance))
    {
        Object hitObject = spheres[hitIndex];
        // okay now we have to update each var depending on the results of the ray 
        // then we can calc the actual final value
        Vec3 hitColour = shadeSphere(sphereDistance, ray, light, hitObject);
        // as its recalling multiplyple addition next hit colour to previous multiplyplied by the current strenght
        pixelColour = pixelColour.addition(hitColour.multiply(strengthOfRay));

        strengthOfRay = strengthOfRay.scale(hitObject.material.specularReflectivity);
        
        // update hitpoin surface normal ray dir making sure it runs shadeSphere with its new values
        Vec3 hitPoint = ray.origin.addition(ray.direction.scale(sphereDistance));
        Vec3 surfaceNormal = calcSurfaceNormal(hitPoint, hitObject);

        // do same as shadow ray making sure its not actually in the same point as it can cause artifcats
        ray.origin = hitPoint.addition(surfaceNormal.scale(0.001f));
        // then find the reflected ray 
        // different from the R = 2 * (N . L) * N - L
        // as we are dealing with ray point from camera to hitpoint
        // rather than the og where it from from the hitpoint to the light we use
        // R = I - 2 * (I . N) * N
        // I incident vector
        // N = surfacenormal
        // R = I - (N * (I. N) * 2)
        ray.direction = ray.direction.subtract(surfaceNormal.scale((ray.direction.dotProduct(surfaceNormal)) * 2.0f)).normalise();


    }
    else if (hitGround)
    {
        // with ground and background not reflective currently so break out loop 
        Vec3 hitColour = shadeGround(groundDistance, ray, light, ground);
        pixelColour = pixelColour.addition(hitColour.multiply(strengthOfRay));
        break; 
    }
    else
    {
        Vec3 hitColour = shadeBackground(ray, background);
        pixelColour = pixelColour.addition(hitColour.multiply(strengthOfRay));
        break; 
    }
}

    // change storing of pixel buffer to use the cuda uchar4
    // storing rgba values 

    unsigned char r = (unsigned char)((fminf(fmaxf(pixelColour.x, 0.0f), 1.0f)) * 255.0f);
    unsigned char g = (unsigned char)((fminf(fmaxf(pixelColour.y, 0.0f), 1.0f)) * 255.0f);
    unsigned char b = (unsigned char)((fminf(fmaxf(pixelColour.z, 0.0f), 1.0f)) * 255.0f);

    pixels[pixelIndex] = make_uchar4(r, g, b, 255);
}



// these functions now help avoid mem being allocated everyframe
// mem init at the start before launchraytracer loop is executed with main.cpp
// and cleared when it ends
void initDevicePixel(int screenWidth, int screenHeight)
{
    // cuda malloc takes additionress of devicepixels storing the size needed as W * H * 4 as RGBA of each pixel
    // 4 removed as changed to uchar4 storage need size of it tho
    cudaMalloc(&devicePixels, screenWidth * screenHeight * sizeof(uchar4));
}

void freeDevicePixels()
{
    // fres up the g
    cudaFree(devicePixels);
}

// additioned init scene to prevent reloading the scene
void initScene(){
    Vec3 camPos = {0.0f, 0.0f, 0.0f};

    Light Hlight = {
        {3.0f, 3.0f, 3.0f},    // position
        {0.25f, 0.25f, 0.25f}, // ambientIntensity
        {0.8f, 0.8f, 0.8f},    // diffuseIntensity
        {0.6f, 0.6f, 0.6f},    // specularIntensity
        1.0f                   // lightIntensity
    };

    Object Hspheres[2];        
    
    Hspheres[0].position = {-1.5f, -0.5f, -4.0f};
    Hspheres[0].material = {
        {1.0f, 1.0f, 1.0f},
        0.1f, // ambient
        0.1f, // diffuse 
        0.9f, // specular 
        64.0f // shininess
    }; 
    Hspheres[0].type = sphereObject;
    Hspheres[0].radius = 1.0f; 
    
    Hspheres[1].position = {1.5f, -0.5f, -4.0f}; 
    Hspheres[1].material = {
        {1.0f, 0.2f, 0.2f}, 
        0.2f, // ambient
        0.7f, // diffuse
        0.3f, // specular 
        32.0f // shininess
    }; 
    Hspheres[1].type = sphereObject;
    Hspheres[1].radius = 1.0f;
    

    Object Hground;
    Hground.position = {0.0f, -2.0f, 0.0f}; 
    Hground.material = {
    {0.7f, 0.7f, 0.7f}, // colour
    0.2f, // ambientReflectivity
    0.6f, // diffuseReflectivity
    0.1f, // specularReflectivity
    24.0f // shininess
    };
    Hground.type = groundObject;
    Hground.normal = {0.0f, 1.0f, 0.0f};

    Object Hbackground;
    Hbackground.material = {
    {1.0f, 1.0f, 1.0f}, // colour
    0.2f, // ambientReflectivity
    0.7f, // diffuseReflectivity
    0.5f, // specularReflectivity
    24.0f  // shininess
    };

    cudaMemcpyToSymbol(light, &Hlight, sizeof(Light));
    cudaMemcpyToSymbol(spheres, &Hspheres, sizeof(Object) * 2);
    cudaMemcpyToSymbol(ground, &Hground, sizeof(Object));
    cudaMemcpyToSymbol(background, &Hbackground, sizeof(Object));

}

float launchRayTracer(void* hostPixels, int screenWidth, int screenHeight)
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
