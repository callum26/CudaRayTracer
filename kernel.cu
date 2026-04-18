#include <cuda_runtime.h>
#include <cmath>
#include <curand_kernel.h>
#include "raytracer.h"

const unsigned int screenWidth = 800;
const unsigned int screenHeight = 800;

// device pixels declared frist point to empty memory additionressi in gpu
static uchar4 *devicePixels = nullptr;

// stores a 3D vector with coords xyz
// vital part of the whole program
struct Vec3
{
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
        if (mag != 0)
        {
            return {x / mag, y / mag, z / mag};
        }
        return {0.0f, 0.0f, 0.0f};
    }

    /*EXPLAIN THIS LATER*/
    __device__ Vec3 cross(const Vec3 &otherVec3) const
    {
        return {
            y * otherVec3.z - z * otherVec3.y,
            z * otherVec3.x - x * otherVec3.z,
            x * otherVec3.y - y * otherVec3.x};
    }
};

// stores lighting properties of material
struct Material
{
    Vec3 colour;
    // ka, kd, ks
    float ambientReflectivity, diffuseReflectivity, specularReflectivity;
    // a
    float shininess;
};

struct Light
{
    Vec3 position;
    // ia, imd, ims
    Vec3 ambientIntensity;
    Vec3 diffuseIntensity;
    Vec3 specularIntensity;
    float lightIntensity;
};

// read about online find the source i think it was a yt video
// seperating objects into types
enum ObjectType
{
    sphereObject,
    triangleObject
};

/* CHANGE RADIUS AND NORMAL SO ITS SEPERATE MAYBE SPLIT OBJECT INTO SPHERE GROUND STRUCTS idk*/
struct Object
{
    Vec3 position;
    Material material;
    ObjectType type;

    float radius;
    Vec3 normal;
    Vec3 v0, v1, v2;
};

struct Ray
{
    Vec3 origin;
    Vec3 direction;
    Vec3 hitPoint;
};

// consts for gpu prevents recopying
/* MAYBE BETTER WAY TO DO THIS?*/
__constant__ Light light;
__constant__ Object objects[20];
__constant__ int objectCount;

// from my reading on https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
// ray defoed as ray(t) = Origin + t * Direction
// Origin (ray.origin)
// Direction (ray.direction)
// t = distance from point of intersection to origin of the ray
// find value of t (if intersects) use it to find the point of intersect with spehere
/* MIGHT SPLIT THIS BACK UP INTO TWO EQUATIONS USING IFS IS INEFFICIENT CREATES SLOW DOWNS*/
__device__ bool rayIntersect(const Ray &ray, const Object &object, float &distance)
{
    if (object.type == sphereObject)
    {

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
    // using Moller Trumbore triangle intersection
    // https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf

    // in the paper equation described as
    // T(u, v) = (1-u-v)V0 + uV1 + uV2
    // u,v are barycentric coords, to express poition of any point using scalars
    // the intersection point is T(u, v) which would be our ray equation just like sphere
    // R(t) = O + tD

    // sub that in
    // O + tD = (1-u-v)V0 + uV1 + vV2

    // expanding this out (they do tell you in the paper but doesnt work it out so im doing it)
    // O + tD = 1V0 - uV0 -vV0 + uV1 + vV2
    // O + tD = V0 - uV0 -vV0 + uV1 + vV2
    // collecting no scalar multiplied terms
    // O - V0 = - tD - uV0 -vV0 + uV1 + vV2
    // focusing on the right collect like terms
    // O - V0 = - tD + uV1 - uV0 + vV2 - vV0
    // then we can extract out scalars
    // O - V0 =  [- D, V1 - V0, + V2 - V0][t u v]
    // giving us the same as whats in the paper

    // to simplify
    // T = O - V0
    // E1 = V1 - V0
    // E2 = V2 - V0
    // T =  [- D, E1, E2][t u v]
    // just to mentin
    // [- D, E1, E2] is a 1X3 vector
    // [t u v] is 3X1
    // T is vector from traingles vertex to origin
    // E1, E2 edge vectors of triangle
    // we are trying to sole for t u v (distance, barycentric coords) respectively

    // using cramers rule we can work out [t u v]
    // first we need the determinant of [- D, E1, E2]
    // for now we can say
    // det([- D, E1, E2])
    // imagine we are multiplying this out as a 3x3 matrix
    // [- D, E1, E2] [t, 0, 0]
    // [- D, E1, E2] [0, u, 0]    =   T
    // [- D, E1, E2] [0, 0, v]

    // we swap T into each coloumn seperately
    // [T, E1, E2]
    // [- D, T, E2]
    // [- D, E1, T]

    // then we take the determaints of all of them and this creates a new 3x1 matrix
    // [det(T, E1, E2)]
    // [det(-D, T, E2)]
    // [det(-D, E1, T)]

    // then in order to get [t u v] we divide by the determinant of the base matrix hence
    // (tuv is one 3x1 matrix )
    // [t]     [det(T, E1, E2)]
    // [u]  =  [det(-D, T, E2)]  /  det([- D, E1, E2])
    // [v]     [det(-D, E1, T)]

    // we can simplify further to remove deting everything
    // for any three vectors A, B, C we can rewrite the det as
    // det(A,B,C) = (A x B) . C
    // the same is true for every variation of this (useful for 3d gemotry)
    // (A x B) . C = (B x C) . A = (C x A) . B

    // they identified key pairs of P = (D x E2) and Q = (T x E1)
    // were useful at reducing unnesscary math so we will focus on isolating those

    // starting with
    // [(T x E1) . E2]
    // [(E2 x -D) . T]
    // [(E1 x T) . -D]
    // P and Q are defined the other way round useful bcos we can cancel out the negatives
    // we will pull them out first
    // [(T x E1) . E2]
    // [-((E2 x D) . T)]
    // [-((E1 x T) . D)]
    // once pulled out we can flip terms to make it negative
    // [(T x E1) . E2]
    // [-(-(D x E2) . T)]
    // [-(-(T x E1) . D)]
    // double negatives cancel out
    // [(T x E1) . E2]
    // [(D x E2) . T]
    // [(T x E1) . D]
    // the final part of the dividor
    // det([- D, E1, E2]) = (E2 x -D) . E1
    // we do same as above pull out minus and flip terms so it fits our P definition
    //(D x E2) . E1

    // just to repeat we now have
    // [t]     [(T x E1) . E2]
    // [u]  =  [(D x E2) . T]   /  (D x E2) . E1
    // [v]     [(T x E1) . D]

    // then sub in P and Q
    // [t]     [Q . E2]
    // [u]  =  [P . T ]   / P . E1
    // [v]     [Q . D ]

    // t = Q . E2 / P . E1
    // u = P . T  / P . E1
    // v = Q . D  / P . E1

    // sumarry
    // P = (D x E2)
    // Q = (T x E1)

    // T = O - V0
    // E1 = V1 - V0
    // E2 = V2 - V0

    // D ray direction
    else if (object.type == triangleObject)
    {
        // both edges of the triangle
        Vec3 edge1 = object.v1.subtract(object.v0);
        Vec3 edge2 = object.v2.subtract(object.v0);

        Vec3 T = ray.origin.subtract(object.v0);
        /* COME UP WITH BETTER VAR NAMES LATER*/
        Vec3 P = ray.direction.cross(edge2);
        Vec3 Q = T.cross(edge1);

        float baseDet = P.dotProduct(edge1);

        // just like ground if parallel no bother or near zero
        if (fabsf(baseDet) < 0.0001f)
        {
            return false;
        }

        float t = Q.dotProduct(edge2) / baseDet;
        float u = P.dotProduct(T) / baseDet;
        float v = Q.dotProduct(ray.direction) / baseDet;

        // also coords u, v must be between 0 and 1
        // and also must add between 0 and 1 so we check
        if (u < 0.0f || u > 1.0f)
            return false; // false if outside range
        if (v < 0.0f || u + v > 1.0f)
            return false; // check for less than 0 and combined check for u + v

        if (t > 0.001f)
        {
            distance = t;
            return true;
        }
        return false;
    }
    else
    {
        return false;
    }
}

// lightDirection - normalised direction vector from hit point to light position
__device__ Vec3 calcLightDirection(const Ray &ray, const Light &light)
{
    Vec3 lightToHit = light.position.subtract(ray.hitPoint);
    return lightToHit.normalise();
}

// surfaceNormal - normalised direction vector of a vector perpendicular to surface on that point
__device__ Vec3 calcSurfaceNormal(const Ray &ray, const Object &object)
{
    // for sphere it is vector from centre of the sphere pointing outwards
    // beginning at hitPoint subtract by centre to get vector from centre to hit point
    // now gonna update surface normal as goes instead of returning it multiple times
    Vec3 surfaceNormal;

    if (object.type == sphereObject)
    {
        // normalising getting direction vector
        surfaceNormal = ray.hitPoint.subtract(object.position).normalise();
    }
    else if (object.type == triangleObject)
    {
        // cross product of E2 X E1 to get the normal
        Vec3 edge1 = object.v1.subtract(object.v0);
        Vec3 edge2 = object.v2.subtract(object.v0);
        surfaceNormal = edge2.cross(edge1).normalise();
    }
    else
    {
        return {0.0f, 1.0f, 0.0f}; // facllback
    }

    // light was causing issues on the ceiling as it is illuminating whats above it
    // calculating the surface normal was giving negative surface normals which the calcs didnt like
    // change to flip to pos
    if (surfaceNormal.dotProduct(ray.direction) > 0.0f)
    {
        surfaceNormal = surfaceNormal.scale(-1.0f);
    }

    return surfaceNormal;
}

__device__ float calcLightDirDotNormal(const Ray &ray, const Light &light, const Object &object)
{
    return calcLightDirection(ray, light).dotProduct(calcSurfaceNormal(ray, object));
}

// must be executed on the set of all light soruces
// Lm direction vector from point on the surface towards each light source
// N surface normal at the point
// Rm direction vector of reflected ray would take from the point
// V direction vector pointing towards the camera

// each point is given by the following equation of surface point Ip
// Ip = ka * ia + Sum (of all light soruces)(kd * (Lm . N)* im,d + ks * (Rm . V)^a * im,s))
// similfing them assigning them variables for their unique purposes
// A = ka * ia
// D = kd * (Lm . N)* im,d
// S = ks * (Rm . V)^a * im,s
// Ip = A + Sum (of all light soruces)(D + S)

__device__ Vec3 calculateAmbient(const Light &light, const Object &object)
{
    Vec3 ambientStrength = light.ambientIntensity.scale(object.material.ambientReflectivity);
    return ambientStrength;
}

__device__ Vec3 calculateDiffuse(const Ray &ray, const Light &light, float distanceToObject, const Object &object)
{
    // D = kd * (Lm . N)* im,d

    // (Lm. N) is calc in calcLightDirectiondotProductSurfaceNormal
    // clamp any negative values are 0 bcos they woukd facing qwaay from light hence no lit
    float diffuseFactor = fmaxf(calcLightDirDotNormal(ray, light, object), 0.0f);

    // im,d represents light scatter in all dirs when a light source hits suface this must be done for all light source
    Vec3 imd = light.diffuseIntensity.scale(light.lightIntensity);

    // we can now cal D
    // D = kd * (Lm . N) * im,d
    Vec3 diffuseStrength = imd.scale(object.material.diffuseReflectivity * diffuseFactor);

    return diffuseStrength;
}

__device__ Vec3 calculateSpecular(const Ray &ray, const Light &light, float distanceToObject, const Object &object)
{
    // S = ks * (Rm . V)^a * im,s

    // incident vector reflection from is
    // R = 2 * (L . N) * N - L

    // we can use our calc funcs to work out all the required elements

    // R = N * 2 * (L . N) - L
    Vec3 reflectDir = calcSurfaceNormal(ray, object).scale(2.0f * calcLightDirDotNormal(ray, light, object)).subtract(calcLightDirection(ray, light));

    // originToHit - V direction vector pointing from hit point to origin
    Vec3 originToHit = (ray.origin.subtract(ray.hitPoint)).normalise();

    // Rm and V -> (Rm . V)
    float reflectdotOriginToHit = reflectDir.dotProduct(originToHit);
    reflectdotOriginToHit = fmaxf(reflectdotOriginToHit, 0.0f); // again like diffuse we clamp to 0

    // im,s is intesity of light scatter in all directions when the light hits surface for specular reflection
    Vec3 ims = light.specularIntensity.scale(light.lightIntensity);

    // we can now cal S
    // S = ks * (Rm . V)^a * im,s`
    // S = im,s * ks * (Rm . V)^a
    return ims.scale(object.material.specularReflectivity * powf(reflectdotOriginToHit, object.material.shininess));
}

__device__ bool isInShadow(const Ray &ray, const Light &light, const Object &object, int objectIndex, float hitDistance)
{
    // origin of shadow is hitpoint as thats where the light strikes the surface
    // saw online common practise to adjust the origin just sligthly to prevent artifcats
    float shadowOffset = fmaxf(0.001f, 0.001f * hitDistance);
    Vec3 shadowNormal = calcSurfaceNormal(ray, object);
    Vec3 shadowToLight = light.position.subtract(ray.hitPoint);
    float shadowToLightDistance = shadowToLight.magnitude();

    Vec3 shadowOrigin = ray.hitPoint.addition(shadowNormal.scale(shadowOffset));
    Vec3 shadowDirection = shadowToLight.normalise();

    // reruns both intersection maths with shadow orgin instead and also initing a new shadowDistance var
    // which will be return and compared to shadowDistanceToLight
    // if shadowDistanceToLight greater than the intersection distance (shadowSphereDistance)
    // then it must mean its in shaded area
    Ray shadowRay = {shadowOrigin, shadowDirection, {0.0f, 0.0f, 0.0f}};

    // loop through all objects

    for (int o = 0; o < objectCount; o++)
    {
        if (o == objectIndex)
            continue;

        float shadowObjectDistance;
        // if shadowToLightDistance is greater than shadowObjectDistance
        // then point must be in shadow
        if (rayIntersect(shadowRay, objects[o], shadowObjectDistance))
        {
            if (shadowObjectDistance > shadowOffset && shadowObjectDistance < shadowToLightDistance)
                return true;
        }
    }

    return false;
}

__device__ Vec3 shadeSphere(float sphereDistance, const Ray &ray, const Light &light, const Object &sphere, int sphereIndex)
{
    // compute hit point for shading
    Ray shadeRay = ray;
    shadeRay.hitPoint = shadeRay.origin.addition(shadeRay.direction.scale(sphereDistance));

    if (isInShadow(shadeRay, light, sphere, sphereIndex, sphereDistance))
    {
        // if its in the shadow only use ambient
        return calculateAmbient(light, sphere).multiply(sphere.material.colour);
    }
    else
    {
        Vec3 ambientStrength = calculateAmbient(light, sphere);
        Vec3 diffuseStrength = calculateDiffuse(shadeRay, light, sphereDistance, sphere);
        Vec3 specularStrength = calculateSpecular(shadeRay, light, sphereDistance, sphere);

        // Ip = A + D + S
        Vec3 phongShading = {
            fminf((ambientStrength.x + diffuseStrength.x + specularStrength.x), 1.0f),
            fminf((ambientStrength.y + diffuseStrength.y + specularStrength.y), 1.0f),
            fminf((ambientStrength.z + diffuseStrength.z + specularStrength.z), 1.0f)};

        return phongShading.multiply(sphere.material.colour);
    }
}

__device__ Vec3 shadeTriangle(float triangleDistance, const Ray &ray, const Light &light, const Object &triangle, int triangleIndex)
{
    Object shadedTriangle = triangle;

    // compute hit point for shading
    Ray shadeRay = ray;
    shadeRay.hitPoint = shadeRay.origin.addition(shadeRay.direction.scale(triangleDistance));

    // keep ground matte like before
    shadedTriangle.material.specularReflectivity = 0.0f;

    if (isInShadow(shadeRay, light, shadedTriangle, triangleIndex, triangleDistance))
    {
        // if its in the shadow only use ambient
        return calculateAmbient(light, shadedTriangle).multiply(shadedTriangle.material.colour);
    }
    else
    {
        Vec3 ambientStrength = calculateAmbient(light, shadedTriangle);
        Vec3 diffuseStrength = calculateDiffuse(shadeRay, light, triangleDistance, shadedTriangle);
        Vec3 specularStrength = calculateSpecular(shadeRay, light, triangleDistance, shadedTriangle);

        // Ip = A + D + S
        Vec3 phongShading = {
            fminf((ambientStrength.x + diffuseStrength.x + specularStrength.x), 1.0f),
            fminf((ambientStrength.y + diffuseStrength.y + specularStrength.y), 1.0f),
            fminf((ambientStrength.z + diffuseStrength.z + specularStrength.z), 1.0f)};

        return phongShading.multiply(shadedTriangle.material.colour);
    }
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

    Vec3 origin = {0.0f, 0.0f, 0.0f};

    // these are then scaled by view port to get correct aspect ratio then normalise it
    Vec3 direction = {normalX * viewPortWidth, normalY * viewPortHeight, -1.0f};
    direction = direction.normalise();

    // init hitPoint will be calced later
    Vec3 hitPoint = {0.0f, 0.0f, 0.0f};

    Ray ray = {
        origin,
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

    for (int i = 0; i < maxBounce; i++)
    {
        float objectDistance = INFINITY;
        int objectIndex = -1;

        for (int o = 0; o < objectCount; o++)
        {
            float closestObjectDistance = INFINITY;
            if (rayIntersect(ray, objects[o], closestObjectDistance))
            {
                if (closestObjectDistance < objectDistance)
                {
                    objectDistance = closestObjectDistance;
                    objectIndex = o;
                }
            }
        }
        bool hitObject = (objectIndex != -1);

        // hitSphere and hitGround are bools for determining whether a specifced ray hit the objects
        // they both also update their value for their respective distances whenever they run and return true
        if (hitObject && objects[objectIndex].type == sphereObject)
        {
            Object hitObject = objects[objectIndex];
            // okay now we have to update each var depending on the results of the ray
            // then we can calc the actual final value
            Vec3 hitColour = shadeSphere(objectDistance, ray, light, hitObject, objectIndex);
            // as its recalling multiplyple addition next hit colour to previous multiplyplied by the current strenght
            pixelColour = pixelColour.addition(hitColour.multiply(strengthOfRay));

            strengthOfRay = strengthOfRay.scale(hitObject.material.specularReflectivity);

            // update hitpoin surface normal ray dir making sure it runs shadeSphere with its new values
            Vec3 hitPoint = ray.origin.addition(ray.direction.scale(objectDistance));
            ray.hitPoint = hitPoint;
            Vec3 surfaceNormal = calcSurfaceNormal(ray, hitObject);

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
        else if (hitObject && objects[objectIndex].type == triangleObject)
        {
            // copy same from sphere
            Object hitObject = objects[objectIndex];
            // okay now we have to update each var depending on the results of the ray
            // then we can calc the actual final value
            Vec3 hitColour = shadeTriangle(objectDistance, ray, light, hitObject, objectIndex);
            // as its recalling multiplyple addition next hit colour to previous multiplyplied by the current strenght
            pixelColour = pixelColour.addition(hitColour.multiply(strengthOfRay));

            strengthOfRay = strengthOfRay.scale(hitObject.material.specularReflectivity);

            // update hitpoin surface normal ray dir making sure it runs shadeSphere with its new values
            Vec3 hitPoint = ray.origin.addition(ray.direction.scale(objectDistance));
            ray.hitPoint = hitPoint;
            Vec3 surfaceNormal = calcSurfaceNormal(ray, hitObject);

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
        else if (!hitObject)
        {
            // no object hit fallback
            Vec3 hitColour = {0.08f, 0.08f, 0.08f};
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
// H prefix meaning host
/*MAYBE MAKE THIS MORE CLEAR*/

void initScene()
{
    Light Hlight = {
        {0.0f, 2.6f, -5.0f},   // position
        {0.20f, 0.20f, 0.20f}, // ambientIntensity
        {0.85f, 0.85f, 0.85f}, // diffuseIntensity
        {0.45f, 0.45f, 0.45f}, // specularIntensity
        1.20f                  // lightIntensity
    };

    Object Hobjects[40];
    int HobjectCount = 0;

    /*EXPLAIN THESE LATER TEST DATA*/
    Material whiteWall = {{0.90f, 0.90f, 0.90f}, 0.28f, 0.75f, 0.02f, 10.0f};
    Material redWall = {{0.85f, 0.15f, 0.15f}, 0.22f, 0.70f, 0.02f, 10.0f};
    Material greenWall = {{0.15f, 0.80f, 0.15f}, 0.22f, 0.70f, 0.02f, 10.0f};

    // back wall
    Hobjects[HobjectCount].v0 = {-3.0f, -3.0f, -8.0f};
    Hobjects[HobjectCount].v1 = {-3.0f, 3.0f, -8.0f};
    Hobjects[HobjectCount].v2 = {3.0f, -3.0f, -8.0f};
    Hobjects[HobjectCount].material = whiteWall;
    Hobjects[HobjectCount].type = triangleObject;
    HobjectCount++;

    Hobjects[HobjectCount].v0 = {3.0f, -3.0f, -8.0f};
    Hobjects[HobjectCount].v1 = {-3.0f, 3.0f, -8.0f};
    Hobjects[HobjectCount].v2 = {3.0f, 3.0f, -8.0f};
    Hobjects[HobjectCount].material = whiteWall;
    Hobjects[HobjectCount].type = triangleObject;
    HobjectCount++;

    // left wall red
    Hobjects[HobjectCount].v0 = {-3.0f, -3.0f, -8.0f};
    Hobjects[HobjectCount].v1 = {-3.0f, -3.0f, -2.0f};
    Hobjects[HobjectCount].v2 = {-3.0f, 3.0f, -8.0f};
    Hobjects[HobjectCount].material = redWall;
    Hobjects[HobjectCount].type = triangleObject;
    HobjectCount++;

    Hobjects[HobjectCount].v0 = {-3.0f, -3.0f, -2.0f};
    Hobjects[HobjectCount].v1 = {-3.0f, 3.0f, -2.0f};
    Hobjects[HobjectCount].v2 = {-3.0f, 3.0f, -8.0f};
    Hobjects[HobjectCount].material = redWall;
    Hobjects[HobjectCount].type = triangleObject;
    HobjectCount++;

    // right wall green
    Hobjects[HobjectCount].v0 = {3.0f, -3.0f, -2.0f};
    Hobjects[HobjectCount].v1 = {3.0f, -3.0f, -8.0f};
    Hobjects[HobjectCount].v2 = {3.0f, 3.0f, -8.0f};
    Hobjects[HobjectCount].material = greenWall;
    Hobjects[HobjectCount].type = triangleObject;
    HobjectCount++;

    Hobjects[HobjectCount].v0 = {3.0f, -3.0f, -2.0f};
    Hobjects[HobjectCount].v1 = {3.0f, 3.0f, -8.0f};
    Hobjects[HobjectCount].v2 = {3.0f, 3.0f, -2.0f};
    Hobjects[HobjectCount].material = greenWall;
    Hobjects[HobjectCount].type = triangleObject;
    HobjectCount++;

    // floor
    Hobjects[HobjectCount].v0 = {-3.0f, -3.0f, -8.0f};
    Hobjects[HobjectCount].v1 = {3.0f, -3.0f, -8.0f};
    Hobjects[HobjectCount].v2 = {-3.0f, -3.0f, -2.0f};
    Hobjects[HobjectCount].material = whiteWall;
    Hobjects[HobjectCount].type = triangleObject;
    HobjectCount++;

    Hobjects[HobjectCount].v0 = {-3.0f, -3.0f, -2.0f};
    Hobjects[HobjectCount].v1 = {3.0f, -3.0f, -8.0f};
    Hobjects[HobjectCount].v2 = {3.0f, -3.0f, -2.0f};
    Hobjects[HobjectCount].material = whiteWall;
    Hobjects[HobjectCount].type = triangleObject;
    HobjectCount++;

    // ceiling
    Hobjects[HobjectCount].v0 = {-3.0f, 3.0f, -8.0f};
    Hobjects[HobjectCount].v1 = {-3.0f, 3.0f, -2.0f};
    Hobjects[HobjectCount].v2 = {3.0f, 3.0f, -8.0f};
    Hobjects[HobjectCount].material = whiteWall;
    Hobjects[HobjectCount].type = triangleObject;
    HobjectCount++;

    Hobjects[HobjectCount].v0 = {-3.0f, 3.0f, -2.0f};
    Hobjects[HobjectCount].v1 = {3.0f, 3.0f, -2.0f};
    Hobjects[HobjectCount].v2 = {3.0f, 3.0f, -8.0f};
    Hobjects[HobjectCount].material = whiteWall;
    Hobjects[HobjectCount].type = triangleObject;
    HobjectCount++;

    cudaMemcpyToSymbol(light, &Hlight, sizeof(Light));
    cudaMemcpyToSymbol(objects, &Hobjects, sizeof(Object) * HobjectCount);
    cudaMemcpyToSymbol(objectCount, &HobjectCount, sizeof(int));
}

float launchRayTracer(void *hostPixels, int screenWidth, int screenHeight)
{
    // https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
    // as mentioned on the nvidia blog its better to use the inbuilt functions for timings in cuda instead of cpu itmings
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