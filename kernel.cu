#include <cuda_runtime.h>
#include <cmath>
#include <math_constants.h>
#include <curand_kernel.h>
#include "raytracer.h"
#include "structs.h"
#include "bvh.cuh"

const unsigned int screenWidth = 800;
const unsigned int screenHeight = 800;

// consts for gpu prevents recopying
// no more array constas too expensive
__constant__ Light light;
__constant__ int objectCount;
__constant__ int bvhRootIndex;
__constant__ int bvhNodeCount;

// device pixels declared frist point to empty memory additionressi in gpu
static uchar4 *devicePixels = nullptr;
// for rng values like angles rays bounce in
static curandState *deviceRngStates = nullptr;
// accum buffer
static Vec3 *deviceAccumulation = nullptr;
// using device pointers instead
// for bvh obvs
static BVHNode *deviceBvhNodes = nullptr;
static int *deviceBvhObjects = nullptr;
// for objs
static Object *deviceObjects = nullptr;
// cur frames
static int currentFrame = 0;

// from my reading on https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
// ray defoed as ray(t) = Origin + t * Direction
// Origin (ray.origin)
// Direction (ray.direction)
// t = distance from point of intersection to origin of the ray
// find value of t (if intersects) use it to find the point of intersect with spehere
__device__ bool rayIntersectSphere(const Ray &ray, const Sphere &sphere, float &distance)
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

    // L = ray.origin - sphere.position
    Vec3 sphereToOrigin = ray.origin - sphere.position;

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
    float a = ray.direction.dot(ray.direction);

    // b = L . D
    // dot sphereToOrigin with ray.direction
    // multiplypy b by 2 explained below
    float b = 2.0f * sphereToOrigin.dot(ray.direction);

    // c = L . L - r^2
    // dot sphereToOrigin with itself subtract sphere radius sphere to make sure its equal to zero explained below
    float c = sphereToOrigin.dot(sphereToOrigin) - (sphere.radius * sphere.radius);

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
        float inv2A = 1.0f / (2.0f * a);
        float closeIntersection = (-b - sqrtf(discriminant)) * inv2A;
        if (closeIntersection > 0.01f)
        {
            distance = closeIntersection;
            return true;
        }

        float farIntersection = (-b + sqrtf(discriminant)) * inv2A;
        if (farIntersection > 0.01f)
        {
            distance = farIntersection;
            return true;
        }
    }
    return false;
}

__device__ bool rayIntersectTriangle(const Ray &ray, const Triangle &triangle, float &distance)
{
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
    // both edges of the triangle
    Vec3 edge1 = triangle.v1 - triangle.v0;
    Vec3 edge2 = triangle.v2 - triangle.v0;

    Vec3 T = ray.origin - triangle.v0;
    /* COME UP WITH BETTER VAR NAMES LATER*/
    Vec3 P = ray.direction.cross(edge2);
    Vec3 Q = T.cross(edge1);

    float baseDet = P.dot(edge1);

    // just like ground if parallel no bother or near zero
    if (fabsf(baseDet) < 0.0001f)
    {
        return false;
    }

    // prevents dividing 3 trimes
    float invDet = 1.0f / baseDet;

    float t = Q.dot(edge2) * invDet;
    float u = P.dot(T) * invDet;
    float v = Q.dot(ray.direction) * invDet;

    // also coords u, v must be between 0 and 1
    // and also must add between 0 and 1 so we check
    if (u < 0.0f || u > 1.0f)
        return false; // false if outside range
    if (v < 0.0f || u + v > 1.0f)
        return false; // check for less than 0 and combined check for u + v

    if (t > 0.01f)
    {
        distance = t;
        return true;
    }
    return false;
}
// searchs through the BVH to find which object ray hits
__device__ bool rayCastBVH(const Ray &ray, float &distance, int &objectIndex, BVHNode *bvhNodes, int *bvhObjects, Object *objects)
{
    // we will use a depth first search to travese the tree
    // stack to store which nodes visted
    int stack[32];
    int stackPtr = 0;
    // pinc to store current node on stack
    stack[stackPtr++] = bvhRootIndex;

    // A
    float closest = INFINITY;
    int closestIdx = -1; // wich obj clostest

    // keep going till stack empty
    while (stackPtr > 0)
    {
        // pop node from stack pdec then get value
        int nodeIdx = stack[--stackPtr];

        // only continue if check bounds node not negative or past node count
        if (nodeIdx < 0 || nodeIdx >= bvhNodeCount)
            continue;

        // fetch node from array bvhNodes using current node index
        BVHNode node = bvhNodes[nodeIdx];

        // do ray test if no intersection then we can skip all objs inside
        if (!hitAABB(ray, node.box, closest))
            continue;

        // if more than 0 objs in node
        if (node.objectCount > 0)
        {
            // runs through all objs in leaf
            for (int i = 0; i < node.objectCount; i++)
            {
                // gets object index from list of all bvhObjects
                // the way it works
                // e.g. leaf node contains 3 objs so node.objectCount = 3
                // node.firstObject is the pos in bvhObjects where the objs within this specific node are stored
                // lets say  node.firstObject = 4
                // bcos there are 3 objs this leaf needs pos 4,5,6 for all 3 objs within the bvhObjects array
                // as it iterates through the object count in our case 0,1,2
                // node.firstObject is added with each of our i
                // giving the ids of 4,5,6 which is what we need
                int objIdx = bvhObjects[node.firstObject + i];

                //  then do normal ray interset with each obj within the box
                float dist = INFINITY;
                // split into two funcs one for tri and one for sphere easier to check for object type then within one equation
                bool objectRayIntersect = objects[objIdx].type == triangleObject
                                              ? rayIntersectTriangle(ray, objects[objIdx].triangle, dist)
                                              : rayIntersectSphere(ray, objects[objIdx].sphere, dist);

                // then check that bool instead
                if (objectRayIntersect)
                {
                    // but we are making sure we always keep the closest obj and its id
                    if (dist < closest)
                    {
                        closest = dist;
                        closestIdx = objIdx;
                    }
                }
            }
        }
        // push children
        else // not leaf node it has up to two children
        {
            // LIFO so do right child first pushed first popped last
            // if righ child push rightIndex onto stack
            if (node.rightIndex >= 0)
                stack[stackPtr++] = node.rightIndex;
            // if left child push leftIndex onto stack
            if (node.leftIndex >= 0)
                stack[stackPtr++] = node.leftIndex;
        }
    }

    // as long as we have found at least on hit
    if (closestIdx >= 0)
    {
        // copy results reporting sucess
        distance = closest;
        objectIndex = closestIdx;
        return true;
    }
    // ifnot no obj hit
    return false;
}

// og code for testing old brute force
__device__ bool rayCastObjects(const Ray &ray, float &distance, int &objectIndex, Object *objects)
{
    float closest = INFINITY;
    int closestIdx = -1;

    for (int i = 0; i < objectCount; i++)
    {
        float dist = INFINITY;
        bool objectRayIntersect = objects[i].type == triangleObject
                                      ? rayIntersectTriangle(ray, objects[i].triangle, dist)
                                      : rayIntersectSphere(ray, objects[i].sphere, dist);

        if (objectRayIntersect && dist < closest)
        {
            closest = dist;
            closestIdx = i;
        }
    }

    if (closestIdx >= 0)
    {
        distance = closest;
        objectIndex = closestIdx;
        return true;
    }

    return false;
}

// og code for testing old brute force shadow
__device__ bool rayCastShadowObjects(const Ray &ray, float &hitDistance, int &objectIndex, Object *objects, float maxDist)
{
    float closest = INFINITY;
    int closestIdx = -1;

    for (int i = 0; i < objectCount; i++)
    {
        float dist = INFINITY;
        bool hit = objects[i].type == triangleObject
                       ? rayIntersectTriangle(ray, objects[i].triangle, dist)
                       : rayIntersectSphere(ray, objects[i].sphere, dist);

        if (hit && dist > 0.001f && dist < maxDist && dist < closest)
        {
            closest = dist;
            closestIdx = i;
        }
    }

    if (closestIdx >= 0)
    {
        hitDistance = closest;
        objectIndex = closestIdx;
        return true;
    }

    return false;
}

__device__ bool rayCastShadowBVH(const Ray &ray, float &hitDistance, int &objectIndex, BVHNode *bvhNodes, int *bvhObjects, Object *objects, float maxDist)
{
    // we will use a depth first search to travese the tree
    // stack to store which nodes visted
    int stack[32];
    int stackPtr = 0;
    // pinc to store current node on stack
    stack[stackPtr++] = bvhRootIndex;

    // anything beyond light is irrev
    float closest = maxDist;
    int closestIdx = -1;

    // keep going till stack empty
    while (stackPtr > 0)
    {
        // pop node from stack pdec then get value
        int nodeIdx = stack[--stackPtr];

        // only continue if check bounds node not negative or past node count
        if (nodeIdx < 0 || nodeIdx >= bvhNodeCount)
            continue;

        // fetch node from array bvhNodes using current node index
        BVHNode node = bvhNodes[nodeIdx];

        // as shadow if any thing hit then must of hit object shadow ray
        if (!hitAABB(ray, node.box, closest))
            continue;

        // if more than 0 objs in node
        if (node.objectCount > 0)
        {
            // runs through all objs in leaf
            for (int i = 0; i < node.objectCount; i++)
            {
                // gets object index from list of all bvhObjects
                // the way it works
                // e.g. leaf node contains 3 objs so node.objectCount = 3
                // node.firstObject is the pos in bvhObjects where the objs within this specific node are stored
                // lets say  node.firstObject = 4
                // bcos there are 3 objs this leaf needs pos 4,5,6 for all 3 objs within the bvhObjects array
                // as it iterates through the object count in our case 0,1,2
                // node.firstObject is added with each of our i
                // giving the ids of 4,5,6 which is what we need
                int objIdx = bvhObjects[node.firstObject + i];

                //  then do normal ray interset with each obj within the box
                float dist = INFINITY;
                // split into two funcs one for tri and one for sphere easier to check for object type then within one equation
                bool hit = objects[objIdx].type == triangleObject
                               ? rayIntersectTriangle(ray, objects[objIdx].triangle, dist)
                               : rayIntersectSphere(ray, objects[objIdx].sphere, dist);

                // then check that bool instead
                if (hit && dist > 0.001f && dist < closest)
                {
                    closest = dist;
                    objectIndex = objIdx;
                    // early exit only if closet hit obj is opqauqe
                    // light is fully blocked no point in searching
                    if (objects[objIdx].material.transparency == 0.0f)
                        goto done; // goto best way to break nested loops
                }
            }
        }
        // push children
        else // not leaf node it has up to two children
        {
            // LIFO so do right child first pushed first popped last
            // if righ child push rightIndex onto stack
            if (node.rightIndex >= 0)
                stack[stackPtr++] = node.rightIndex;
            // if left child push leftIndex onto stack
            if (node.leftIndex >= 0)
                stack[stackPtr++] = node.leftIndex;
        }
    }
done:
    if (closestIdx >= 0)
    {
        hitDistance = closest;
        objectIndex = closestIdx;
        return true;
    }
    // ifnot no obj hit
    return false;
}

// surfaceNormal - normalised direction vector of a vector perpendicular to surface on that point
__device__ Vec3 calcSurfaceNormal(const Ray &ray, const Sphere &sphere)
{
    return (ray.hitPoint - sphere.position).normalise();
}

__device__ Vec3 calcSurfaceNormal(const Ray &, const Triangle &triangle)
{
    return triangle.normal;
}
// returns respective surface normal based on functions type inputed
__device__ Vec3 calcSurfaceNormal(const Ray &ray, const Object &object)
{
    return object.type == sphereObject ? calcSurfaceNormal(ray, object.sphere)
                                       : calcSurfaceNormal(ray, object.triangle);
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

__device__ void calcLighting(const Ray &ray, const Light &light, float distanceToObject, const Object &object, Vec3 surfaceNormal, Vec3 &diffuse, Vec3 &specular)
{
    // flips normal if inside
    surfaceNormal = ray.direction.dot(surfaceNormal) > 0.0f
                        ? surfaceNormal * -1.0f
                        : surfaceNormal;

    Vec3 lightToHit = light.position - ray.hitPoint;
    float lightDistance = lightToHit.magnitude();
    Vec3 lightDirection = lightToHit.normalise();

    float lightScaling = 1.0f / fmaxf(0.01f, (lightDistance * lightDistance));
    float lightDotNormal = __saturatef(lightDirection.dot(surfaceNormal));

    Vec3 reflectDir = (surfaceNormal * (2.0f * lightDotNormal)) - lightDirection;

    Vec3 originToHit = (ray.origin - ray.hitPoint).normalise();

    float reflectdotOriginToHit = reflectDir.dot(originToHit);
    reflectdotOriginToHit = __saturatef(reflectdotOriginToHit);

    diffuse = (light.diffuseIntensity * (light.lightIntensity * lightScaling)) * (object.material.diffuseReflectivity * lightDotNormal);
    specular = (light.specularIntensity * (light.lightIntensity * lightScaling)) * (object.material.specularReflectivity * __powf(reflectdotOriginToHit, object.material.shininess));
}

// refraction ray direction
__device__ Vec3 refractDir(const Ray &ray, const Vec3 &surfaceNormal, const float &refractionIndex)
{
    // using snells law in vectorised from
    // T = nI + (nc1 - sqrt(1-n^2(1-c1^2)))N
    // T refracted ray
    // I incident vector
    // N surface normal
    // n1, n2 refractive indices/  n being the ratio between n1/n2
    // c1 the negative cosine of the incident angle
    float refractionIndexSq = refractionIndex * refractionIndex;
    // cosTheta -incident angle cosine from incoming ray and surface normal is -I . N
    float cosThetaI = __saturatef(-ray.direction.dot(surfaceNormal)); // shouldnt be anyways but stops angles larger than 1
    // as we are sqrting 1-n^2(1-c1^2) we must make sure that that part
    // known as the transmitted angle term is not negative otherwise that total internal reflection
    // snells law referred to as
    // n1sin(thetaI) = n2sin(thetaT)
    // common rearangement
    // n1/n2sin(thetaI) = sin(thetaT)
    // nsin(thetaI) = sin(thetaT)
    // n^2 sin^2(thetaI) = sin^2(thetaT)
    // n^2 (1 - cos^2(thetaI)) = = sin^2(thetaT)
    // which is the same as n^2(1-c1^2) just missing the 1-
    // if we calc sin^2 theta we can know if we have to exit early
    float sin2ThetaT = refractionIndexSq * (1 - cosThetaI * cosThetaI);
    // as we are 1 - sin2Theta if sin2Theta > 1 then it would become negative
    if (sin2ThetaT > 1.0f)
        return (ray.direction - (surfaceNormal * (2.0f * ray.direction.dot(surfaceNormal)))).normalise();
    // else return full equation
    // T = nI + N(nc1 - sqrt(1-n^2(1-c1^2)))
    return ((ray.direction * refractionIndex) + (surfaceNormal * (refractionIndex * cosThetaI - sqrtf(fmaxf(0.0f, 1.0f - sin2ThetaT))))).normalise();
}

// reflected ray direction
__device__ Vec3 reflectDir(const Ray &ray, const Vec3 &surfaceNormal)
{
    // R = I - 2N(I . N)
    // R reflected ray
    // I incident ray
    // N surface normal
    // easier to R = I - N * (2 * (I . N)) in this case bcos of the way the vec3s r calced
    Vec3 reflectDir = ray.direction - (surfaceNormal * (2.0f * ray.direction.dot(surfaceNormal)));
    return reflectDir.normalise();
}

__device__ void applyBeerLambertAbsorption(Vec3 &strengthOfRay, const Object &hitObject, int objectIndex, int insideObjectIndex, const Vec3 &insideEntryPoint, const Ray &ray)
{
    if (insideObjectIndex == objectIndex)
    {
        // inside of passing in object distance we have to calculate the obj dist
        // between ray hit point and new inside entry point
        float objectDistance = (ray.hitPoint - insideEntryPoint).magnitude();
        if (objectDistance <= 0.0f)
            return;

        Vec3 absorption = hitObject.material.absorption;
        // expf returns e^x where x is ()
        strengthOfRay.x *= __expf(-absorption.x * objectDistance);
        strengthOfRay.y *= __expf(-absorption.y * objectDistance);
        strengthOfRay.z *= __expf(-absorption.z * objectDistance);
    }
}

__device__ void offsetRayOrigin(Ray &ray, const Vec3 &offsetDirection, float offsetAmount)
{
    ray.origin = ray.hitPoint + (offsetDirection * offsetAmount);
}

/*EXPLAIN THIS*/
__device__ float calcFresnel(const Ray &ray, const Vec3 &surfaceNormal, float refractionRatio)
{
    // schlicks approx fresnel factor specular reflection
    float R0 = (1.0f - refractionRatio) / (1.0f + refractionRatio);
    R0 = R0 * R0;
    float cosTheta = __saturatef(-ray.direction.dot(surfaceNormal));
    return __saturatef(R0 + (1.0f - R0) * __powf(1.0f - cosTheta, 5.0f));
}

__device__ __forceinline__ SurfaceInteraction computeSurfaceInteraction(const Ray &ray, const Vec3 &rawNormal)
{
    SurfaceInteraction interaction;
    // before was assuming that surface normal alaways points against rau direction
    interaction.inside = ray.direction.dot(rawNormal) > 0.0f;
    // now testing it against ray direction if true then inside

    // flips normal if inside
    interaction.normal = interaction.inside
                             ? rawNormal * -1.0f
                             : rawNormal;
    return interaction;
}

__device__ float processTransparentRay(Ray &ray, const Object &hitObject, int objectIndex, float objectDistance, int &insideObjectIndex, Vec3 &insideEntryPoint, Vec3 &strengthOfRay, curandState &rng, Vec3 &surfaceNormal, bool isShadowRay = false)
{
    SurfaceInteraction interaction = computeSurfaceInteraction(ray, surfaceNormal);
    surfaceNormal = interaction.normal;
    // also adjusts refraction ratio because its changes depending on exit / entrance
    float refractionRatio = interaction.inside
                                ? hitObject.material.refraction
                                : 1.0f / hitObject.material.refraction;

    // beer lambert law as light travel through medium each wavelength is absored exponentially with distance
    // defined as
    // I = I0 x (e^(-a x d))
    // I is the new strength after absoprtion
    // I0 inital strength
    // a is abosorption coeff
    // d is distance travel through medium
    applyBeerLambertAbsorption(strengthOfRay, hitObject, objectIndex, insideObjectIndex, insideEntryPoint, ray);

    // for traingles as they are open sufraces they can just pass throyg
    if (hitObject.type == triangleObject)
    {
        // make sure adv origin past the surface so no slef intersect
        offsetRayOrigin(ray, ray.direction, 0.001f);
        // direction unchanged
        return 1.0f;
    }

    float cosTheta = fminf(fabsf(ray.direction.dot(surfaceNormal)), 1.0f);
    float sin2ThetaT = refractionRatio * refractionRatio * (1.0f - cosTheta * cosTheta);
    bool totalInternalReflection = sin2ThetaT > 1.0f;

    float fresnelValue = calcFresnel(ray, surfaceNormal, refractionRatio);

    // for spheres
    // we need to decide either to refract or reflectusing  fresnel
    // on shadow ray always refract
    bool doReflect = totalInternalReflection || (!isShadowRay && (curand_uniform(&rng) < fresnelValue));

    if (doReflect)
    {
        // specular reflection branch primary ray only
        // reflect ray off the surface normal
        // clasic equagiton
        Vec3 refractDirection = reflectDir(ray, surfaceNormal);

        // again both uodate dir and origin
        offsetRayOrigin(ray, surfaceNormal, 0.001f);
        ray.direction = refractDirection;

        // inside tracking same reflection doesnt go in obvs
        return 1.0f;
    }
    else
    {
        // compute refr dir
        ray.direction = refractDir(ray, surfaceNormal, refractionRatio);
        // move origin slightly inside the surface in the refracted dir
        offsetRayOrigin(ray, ray.direction, 0.001f);

        // update the inside objectfor next beer
        // if entering sphere record if not dont bother
        if (!interaction.inside)
        {
            insideObjectIndex = objectIndex;
            insideEntryPoint = ray.hitPoint;
        }
        else
        {
            insideObjectIndex = -1;
        }

        return 1.0f;
    }
}

__device__ void processOpaqueRay(Ray &ray, Vec3 surfaceNormal, curandState &rng)
{
    // for opquae objects
    // standard pracise to use diffuse bounce cosine weighted hemisphere sampling
    // im going to use the example from https://www.pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions
    // adjust ray oirigin like before
    offsetRayOrigin(ray, surfaceNormal, 0.01f);

    // in real life diffuse light would scatter randomly
    // could just multiply the random two directions like theta = R1 * PI and phi = R2 * 2PI
    // however this would sample uniformly accross every directions not realistic

    // in real life diffuse surfaces reflect more light towards the surface normal and less strongly parallel to surface
    // lambertian reflectance

    // in order for this to work we need to
    // convert uniform random numbers into directions
    // this is where we use a direction probability density function
    // where we use spherical coords as opposed to cartesian
    // (r, θ, phi)
    // r - distance from orign
    // θ - polar angle from vert axis
    // phi - azimuth angle roattion around vert axis

    // cartesian conversion
    // x = rsin(θ)cos(phi)
    // y = rsin(θ)sin(phi)
    // z = rcos(θ)
    // as a unit direction vector r = 1 so we can remove that

    // converting back to sphereical given (x,y,z)
    // distance from origin is gonna be mag of xyz
    // r = sqrt(x^2 + y^2 + z^2)

    // z = rcos(θ)
    // cos(θ) = z/r

    // θ = arccos(z/r)

    // we dont need to worry about z coords

    // phi direction around normal
    // theta how far away from normal

    // we also need the differential solid angle
    // wont go into this
    // d(w) = sinθ * dθ * d(phi)

    // paper defines directional PDF as p(w) = cos(θ) / pi
    // to get a 1D PDF with theta multiply by d(w)
    // integrate also removing the dθ from d(w) as we are integrating by θ
    // integral p(w) * sinθ * d(phi)
    // sub in p(w)
    // integral (cosθ / pi)sinθ d(phi)
    // pull out constants
    // (cosθsinθ / pi) integral d(phi)
    // integral d(phi) = 2pi
    // (cosθsinθ / pi) * 2pi
    // 2sinθcosθ
    // int from 0  to  theta
    // F(θ) = int(2sin(t)cos(t))dt
    // which is the cumulative distribution func
    // F(θ) = sin^2(θ)

    // for inverse transform sampling we set u, random bumber equal to F
    // u = sin^2(theta)
    // sin(theta) = u

    // phi = 2piR1 (uniform around circle)
    // θ = arccos(sqrt(1-R2))   (weighted towards normal)
    // where R1 and R2 and random floats

    // generate two random numbers
    float randNum1 = curand_uniform(&rng);
    float randNum2 = curand_uniform(&rng);

    // phi = 2piR1
    float phi = 2.0f * CUDART_PI_F * randNum1;

    // θ = arccos(sqrt(1-R2))
    // instead of using arccos as it is very expensive
    // cos(θ) = sqrt(1-R2)
    // cos^2(θ) = 1-R2

    // classic equation
    // sin(θ) = sqrt(1-cos^2(θ))
    // sin(θ) = sqrt(1-(1-R2))

    // sin(θ) = sqrt(R2)
    // same as our above sin(theta) = u
    float sinTheta = sqrtf(randNum2);
    // cos^2(θ) = 1-R2
    // cos(θ) = sqrt(1-R2)
    float cosTheta = sqrtf(1.0f - randNum2);

    // in this to represent each axis we create a
    // tangent for x axis
    // bitangent for y axis
    // surfacenormal for z

    // we need a vector that is not parallel to the surface normal
    // if the abs of x of surfacenormal is than than 0.99
    // that must mean the normal is not pointing in the x direction
    // otherwise if it is greater than it then we cant use x bcos they would be parallel so we use Y axis
    // we must do this later bcos we need the tangent of the surface normal
    // cross producting to get the tangent to the surface normal with identical same direction would equal 0
    Vec3 up = fabsf(surfaceNormal.x) < 0.99f
                  ? Vec3{1.0f, 0.0f, 0.0f}  // use x axis
                  : Vec3{0.0f, 1.0f, 0.0f}; // if not use y

    // tangent to the surface normal
    Vec3 tangent = up.cross(surfaceNormal).normalise();

    Vec3 bitangent = surfaceNormal.cross(tangent);

    // they define the bounce direction as
    // d = T . cos(phi)sinθ + B . sin(phi)sin(θ) + N . cos(θ)
    // each plus representing each xyz direction
    // x = T . cos(phi)sinθ
    // y = B . sin(phi)sin(θ)
    // z = N . cos(θ)
    // exactly the same as our cartesian conversion except now we are multipling by the following vectors
    // T - tangent vector
    // B - bitangent
    // N - surface normal

    // putting that all in we get
    // dont have to normalise already unit vector
    ray.direction = (tangent * (cosf(phi) * sinTheta) + bitangent * (sinf(phi) * sinTheta) + surfaceNormal * cosTheta).normalise();
}

/*ADD SPECULAR LIGHT TRANSPARENT OBJ SPECULAR I THINK SURELY?*/
__device__ Vec3 postShadingColour(const Ray &ray, const Object &object, float objectDistance, int objectIndex, curandState *rng, Vec3 surfaceNormal, BVHNode *bvhNodes, int *bvhObjects, Object *objects, Vec3 strengthOfRay, bool useBVH)
{
    const int lightSamples = 1;

    Ray shadeRay = ray;
    shadeRay.hitPoint = shadeRay.origin + (shadeRay.direction * objectDistance);
    offsetRayOrigin(shadeRay, surfaceNormal, 0.01f);

    Vec3 accumulatedDiffuse = {0.0f, 0.0f, 0.0f};
    Vec3 accumulatedSpecular = {0.0f, 0.0f, 0.0f};

    // monte carlo method
    // sample multi points on surface
    for (int sample = 0; sample < lightSamples; sample++)
    {
        // we need a random angle as well as radius for the sample
        // full circle obvs 2pi
        float angle = curand_uniform(rng) * 2.0f * CUDART_PI_F;        // mapping angle in radians 0 to 2pi can shoot in any direction
        float radius = sqrtf(curand_uniform(rng)) * light.lightRadius; // sqrt to spread out evenly

        // then as well we need a position
        Vec3 sampleLightPosition = light.position;
        // direction given by angle
        // radius allows a pos to be determined allong the sample
        sampleLightPosition.x += cosf(angle) * radius; //
        sampleLightPosition.z += sinf(angle) * radius;

        // sample as before now we get new lightpos
        Vec3 shadowToLight = sampleLightPosition - shadeRay.origin;
        float shadowToLightDistance = shadowToLight.magnitude();
        Vec3 shadowDirection = shadowToLight * (1.0f / shadowToLightDistance);

        // introducing continually bouncing shadow rays werent a thing b4 basically the same tho
        Vec3 shadowTransmission = {1.0f, 1.0f, 1.0f}; // 1 ful shadow passthrough
        Vec3 currentShadowOrigin = shadeRay.origin;
        float distanceRemain = shadowToLightDistance;
        Vec3 shadowStrength = {1.0f, 1.0f, 1.0f};

        // vars for shadow bounces
        int shadowBounces = 0;
        const int maxShadowBounces = 3;
        bool blocked = false;

        int insideObjectIndex = -1; // -1 meansn air
        Vec3 insideEntryPoint = {0.0f, 0.0f, 0.0f};

        while (shadowBounces < maxShadowBounces)
        {
            Ray shadowRay = {currentShadowOrigin, shadowDirection, {0.0f, 0.0f, 0.0f}};
            float shadowHitDistance = INFINITY;
            int shadowHitObject = -1;

            bool hitSomething = useBVH
                                    ? rayCastShadowBVH(shadowRay, shadowHitDistance, shadowHitObject, bvhNodes, bvhObjects, objects, distanceRemain)
                                    : rayCastShadowObjects(shadowRay, shadowHitDistance, shadowHitObject, objects, distanceRemain);

            // check if we hit an object before reaching the light
            if (hitSomething && shadowHitDistance > 0.001f && shadowHitDistance < distanceRemain - 0.001f)
            {
                Object hitObj = objects[shadowHitObject];
                Vec3 surfaceNormal = calcSurfaceNormal(shadowRay, hitObj);

                if (hitObj.material.transparency > 0.0f)
                {
                    // set hit point ray before passing
                    shadowRay.hitPoint = currentShadowOrigin + (shadowDirection * shadowHitDistance);

                    SurfaceInteraction interaction = computeSurfaceInteraction(shadowRay, surfaceNormal);
                    surfaceNormal = interaction.normal;

                    // also adjusts refraction ratio because its changes depending on exit / entrance
                    float refractionRatio = interaction.inside
                                                ? hitObj.material.refraction
                                                : 1.0f / hitObj.material.refraction;

                    applyBeerLambertAbsorption(shadowStrength, hitObj, shadowHitObject, insideObjectIndex, insideEntryPoint, shadowRay);

                    shadowTransmission = shadowTransmission * shadowStrength;
                    if (!interaction.inside)
                    {
                        shadowTransmission = shadowTransmission * hitObj.material.transparency;
                    }
                    shadowStrength = {1.0f, 1.0f, 1.0f};

                    Vec3 refractDirection = refractDir(shadowRay, surfaceNormal, refractionRatio);
                    if (!interaction.inside)
                    {
                        insideObjectIndex = shadowHitObject;
                        insideEntryPoint = shadowRay.hitPoint;
                    }
                    else
                    {
                        insideObjectIndex = -1;
                    }

                    shadowRay.direction = refractDirection;
                    offsetRayOrigin(shadowRay, shadowRay.direction, 0.001f);

                    // continue shadow ray through transparent object
                    currentShadowOrigin = shadowRay.origin;
                    shadowDirection = shadowRay.direction;
                    distanceRemain -= shadowHitDistance;
                    shadowBounces++;
                }
                else
                {
                    // opaque obj then light is completely blocked.
                    blocked = true;
                    break;
                }
            }
            else
            {
                // ray reached light or no more hits
                break;
            }
        }

        // then basicaally the same except accum dif and spec are multiplied by shadow transmission factor
        if (!blocked)
        {
            Vec3 sampleDiffuse = {0.0f, 0.0f, 0.0f};
            Vec3 sampleSpecular = {0.0f, 0.0f, 0.0f};
            calcLighting(shadeRay, light, objectDistance, object, surfaceNormal, sampleDiffuse, sampleSpecular);

            accumulatedDiffuse = accumulatedDiffuse + (sampleDiffuse * shadowTransmission);
            accumulatedSpecular = accumulatedSpecular + (sampleSpecular * shadowTransmission);
        }
    }

    // Ip = ka * ia + Sum (of all light soruces)(kd * (Lm . N)* im,d + ks * (Rm . V)^a * im,s))
    float invLightSamples = (1.0f / (float)lightSamples);
    Vec3 finalDiffuse = accumulatedDiffuse * invLightSamples;
    Vec3 finalSpecular = accumulatedSpecular * invLightSamples;

    return (finalDiffuse * object.material.colour) + finalSpecular;
}

__global__ void initRNGKernel(curandState *states, int width, int height)
{
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelX >= width || pixelY >= height)
        return;

    int pixelIndex = (pixelY * width + pixelX);

    curand_init(1000, pixelIndex, 0, &states[pixelIndex]);
}

// ACES approximation https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
// common tone mapping curve to make it look better
__device__ __forceinline__ float acesToneMap(float x)
{
    float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return __saturatef((x * (a * x + b)) / (x * (c * x + d) + e));
}
__device__ Vec3 accumulateColour(Vec3 *accumulation, Vec3 processedColour, int pixelIndex, int frameIndex)
{
    // accumulates the colour from this frame and previous building up a multi frame average image
    accumulation[pixelIndex].x += processedColour.x;
    accumulation[pixelIndex].y += processedColour.y;
    accumulation[pixelIndex].z += processedColour.z;
    // computes averages colour across all of the frames so far
    // helps smoothen out noise
    float inverseFrame = 1.0f / (float)(frameIndex + 1);
    Vec3 accumulatedColour;

    accumulatedColour.x = accumulation[pixelIndex].x * inverseFrame;
    accumulatedColour.y = accumulation[pixelIndex].y * inverseFrame;
    accumulatedColour.z = accumulation[pixelIndex].z * inverseFrame;

    return accumulatedColour;
}

__device__ uchar4 toneMappedPixel(Vec3 *accumulation, int pixelIndex, int frameIndex, Vec3 processedColour)
{
    Vec3 accumulatedColour = accumulateColour(accumulation, processedColour, pixelIndex, frameIndex);
    // scales exposure
    const float exposure = 0.6f;
    float r = sqrtf(acesToneMap(accumulatedColour.x * exposure));
    float g = sqrtf(acesToneMap(accumulatedColour.y * exposure));
    float b = sqrtf(acesToneMap(accumulatedColour.z * exposure));

    unsigned char finalR = (unsigned char)(__saturatef(r) * 255.0f);
    unsigned char finalG = (unsigned char)(__saturatef(g) * 255.0f);
    unsigned char finalB = (unsigned char)(__saturatef(b) * 255.0f);

    return make_uchar4(finalR, finalG, finalB, 255);
}

__global__ void renderKernel(uchar4 *pixels, curandState *rngStates, Vec3 *accumulation, int frameIndex, int screenWidth, int screenHeight, BVHNode *bvhNodes, int *bvhObjects, Object *objects, bool useBVH)
{
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    float viewPortHeight = 2.0f;
    float viewPortWidth = ((float)screenWidth / (float)screenHeight) * viewPortHeight;

    if (pixelX >= screenWidth || pixelY >= screenHeight)
        return;

    Vec3 origin = {0.0f, 0.0f, 0.0f};

    int writeRow = (screenHeight - 1 - pixelY);
    int pixelIndex = (writeRow * screenWidth + pixelX);
    curandState rng = rngStates[pixelIndex];

    const int samplesPerPixel = 1;

    Vec3 postSampleColour = {0.0f, 0.0f, 0.0f};

    for (int s = 0; s < samplesPerPixel; s++)
    {
        float randomJitterX = curand_uniform(&rng) - 0.5f;
        float randomJitterY = curand_uniform(&rng) - 0.5f;

        // map pixel coords between -0.5/0.5 for x and 0.5/-0.5 Y
        float normalX = (((float)pixelX + 0.5f + randomJitterX) / (screenWidth - 1)) - 0.5f;
        float normalY = 0.5f - (((float)pixelY + 0.5f + randomJitterY) / (screenHeight - 1));

        // these are then scaled by view port to get correct aspect ratio then normalise it
        Vec3 direction = {normalX * viewPortWidth, normalY * viewPortHeight, -1.0f};
        direction = direction.normalise();

        Ray ray = {origin, direction, {0.0f, 0.0f, 0.0f}};

        // additioning reflection rays
        // every time ray bounces its strength is reduced
        Vec3 pixelColour = {0.0f, 0.0f, 0.0f};
        Vec3 strengthOfRay = {1.0f, 1.0f, 1.0f};
        const int maxBounce = 4;

        // beers law tracking
        int insideObjectIndex = -1; // -1 meansn air
        Vec3 insideEntryPoint = {0.0f, 0.0f, 0.0f};

        for (int i = 0; i < maxBounce; i++)
        {
            float objectDistance = INFINITY;
            int objectIndex = -1;

            if (useBVH)
                rayCastBVH(ray, objectDistance, objectIndex, bvhNodes, bvhObjects, objects);
            else
                rayCastObjects(ray, objectDistance, objectIndex, objects);

            // they both also update their value for their respective distances whenever they run and return true
            if (objectIndex != -1)
            {
                Object hitObject = objects[objectIndex];

                // check for light emmisve material
                if (hitObject.material.emission.x > 0.0f || hitObject.material.emission.y > 0.0f || hitObject.material.emission.z > 0.0f)
                {
                    pixelColour = pixelColour + (hitObject.material.emission * strengthOfRay);
                    break;
                }

                // okay now we have to update each var depending on the results of the ray
                // then we can calc the actual final value
                // update hitointl ray dir making sure it runs shadeSphere with its new values
                Vec3 hitPoint = ray.origin + (ray.direction * objectDistance);
                ray.hitPoint = hitPoint;

                Vec3 surfaceNormal = calcSurfaceNormal(ray, hitObject);

                // random branching for partial transparency
                bool treatAsTransparent = (hitObject.material.transparency > 0.0f) &&
                                          (curand_uniform(&rng) < hitObject.material.transparency);

                if (treatAsTransparent)
                {
                    // func mutates strengthOfRay in place with beer absorp and leaves
                    // the ray strength untouched so transmission stays visible
                    processTransparentRay(ray, hitObject, objectIndex, objectDistance, insideObjectIndex, insideEntryPoint, strengthOfRay, rng, surfaceNormal);
                }
                else // otherwise treat as opaque
                {
                    Vec3 hitColour = postShadingColour(ray, hitObject, objectDistance, objectIndex, &rng, surfaceNormal, bvhNodes, bvhObjects, objects, strengthOfRay, useBVH);
                    pixelColour = pixelColour + (hitColour * strengthOfRay);

                    // calcs via opaque ray function simples the code
                    processOpaqueRay(ray, surfaceNormal, rng);
                    strengthOfRay = strengthOfRay * hitObject.material.colour;
                }

                if ((strengthOfRay.x + strengthOfRay.y + strengthOfRay.z) < 0.01f) // more aggressive culling
                {
                    break;
                }
            }
        }
        postSampleColour = postSampleColour + pixelColour;
    }

    // change storing of pixel buffer to use the cuda uchar4
    // storing rgba values
    Vec3 processedColour = postSampleColour * (1.0f / float(samplesPerPixel));

    rngStates[pixelIndex] = rng;

    pixels[pixelIndex] = toneMappedPixel(accumulation, pixelIndex, frameIndex, processedColour);
}

// these functions now help avoid mem being allocated everyframe
// mem init at the start before launchraytracer loop is executed with main.cpp
// and cleared when it ends
void initDevicePixel(int w, int h)
{
    // cuda malloc takes additionress of devicepixels storing the size needed as W * H * 4 as RGBA of each pixel
    // 4 removed as changed to uchar4 storage need size of it tho
    cudaMalloc(&devicePixels, w * h * sizeof(uchar4));
    cudaMalloc(&deviceAccumulation, w * h * sizeof(Vec3));
    cudaMemset(deviceAccumulation, 0, w * h * sizeof(Vec3));
    currentFrame = 0;
}

void freeDevicePixels()
{
    // fres up the g
    cudaFree(devicePixels);
    cudaFree(deviceAccumulation);
}

inline void addSphere(Object *objects, int &objectCount, const Vec3 &pos, const Material &mat, const float &radius)
{
    objects[objectCount].sphere.position = pos;
    objects[objectCount].material = mat;
    objects[objectCount].type = sphereObject;
    objects[objectCount].sphere.radius = radius;
    objectCount++;
}

inline void addTriangle(Object *objects, int &objectCount, const Vec3 &v0, const Vec3 &v1, const Vec3 &v2, const Material &mat)
{
    objects[objectCount].triangle.v0 = v0;
    objects[objectCount].triangle.v1 = v1;
    objects[objectCount].triangle.v2 = v2;
    objects[objectCount].material = mat;
    objects[objectCount].type = triangleObject;

    // saves calculating later itll never change
    Vec3 edge1 = objects[objectCount].triangle.v1 - objects[objectCount].triangle.v0;
    Vec3 edge2 = objects[objectCount].triangle.v2 - objects[objectCount].triangle.v0;
    objects[objectCount].triangle.normal = edge2.cross(edge1).normalise();

    objectCount++;
}

// can use the triangles to make a quad by defining 4 corners making up a quad
// v0 bottom left
// v1 top left
// v2 top right
// v3 bottom right
inline void addQuadAsTwoTriangles(Object *objects, int &objectCount, const Vec3 &v0, const Vec3 &v1, const Vec3 &v2, const Vec3 &v3, const Material &mat)
{
    addTriangle(objects, objectCount, v0, v1, v2, mat);
    addTriangle(objects, objectCount, v2, v3, v0, mat);
}

// additioned init scene to prevent reloading the scene
// H prefix meaning host
/*MAYBE MAKE THIS MORE CLEAR*/
void initScene()
{
    Light Hlight = {
        {0.0f, 2.95f, -5.0f},  // position
        {0.20f, 0.18f, 0.14f}, // ambientIntensity
        {1.0f, 0.92f, 0.78f},  // diffuseIntensity
        {1.0f, 0.95f, 0.85f},  // specularIntensity
        14.0f,                 // lightIntensity
        0.9f                   // lightRadius
    };

    Object Hobjects[256];
    BuildObject buildObject[256];
    int HobjectCount = 0;

    /*EXPLAIN THESE LATER TEST DATA*/
    Material whiteWall = {{0.82f, 0.82f, 0.80f}, 0.30f, 0.78f, 0.02f, 5.0f, 0.0f, 1.0f, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
    Material redWall = {{0.65f, 0.07f, 0.07f}, 0.28f, 0.72f, 0.02f, 5.0f, 0.0f, 1.0f, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
    Material greenWall = {{0.10f, 0.48f, 0.10f}, 0.28f, 0.72f, 0.02f, 5.0f, 0.0f, 1.0f, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
    Material lightFixtureMaterial = {{1.0f, 1.0f, 1.0f}, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, {0.0f, 0.0f, 0.0f}, {18.0f, 16.0f, 13.0f}};
    // three showcase spheres
    // saturated diffuse
    Material sphereDiffuse = {{0.08f, 0.15f, 0.88f}, 0.35f, 0.85f, 0.05f, 24.0f, 0.0f, 1.0f, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
    // clear glass
    Material sphereGlass = {{0.98f, 0.98f, 0.98f}, 0.05f, 0.10f, 0.95f, 128.0f, 1.0f, 1.52f, {0.015f, 0.015f, 0.015f}, {0.0f, 0.0f, 0.0f}};
    // amber tinted glass
    Material sphereAmber = {{0.98f, 0.98f, 0.98f}, 0.05f, 0.10f, 0.95f, 128.0f, 1.0f, 1.45f, {0.08f, 0.40f, 0.95f}, {0.0f, 0.0f, 0.0f}};
    // lighting fixture

    //
    //
    //
    // PERFORMANCE TEST
    // 64 spheres
    for (int x = 0; x < 8; x++)
    {
        for (int z = 0; z < 8; z++)
        {
            Vec3 pos = {-3.0f + x * 0.85f, -2.5f, -3.0f - z * 0.85f};
            addSphere(Hobjects, HobjectCount, pos, sphereDiffuse, 0.35f);
        }
    }

    /*
    addQuadAsTwoTriangles(
        Hobjects, HobjectCount,
        {-0.8f, 2.999f, -5.8f}, {-0.8f, 2.999f, -4.2f},
        {0.8f, 2.999f, -4.2f}, {0.8f, 2.999f, -5.8f},
        lightFixtureMaterial);

    // back wall
    addQuadAsTwoTriangles(
        Hobjects, HobjectCount,
        {3.0f, -3.0f, -8.0f}, {-3.0f, -3.0f, -8.0f},
        {-3.0f, 3.0f, -8.0f}, {3.0f, 3.0f, -8.0f},
        whiteWall);

    // left wall red
    addQuadAsTwoTriangles(
        Hobjects, HobjectCount,
        {-3.0f, -3.0f, 1.0f}, {-3.0f, 3.0f, 1.0f},
        {-3.0f, 3.0f, -8.0f}, {-3.0f, -3.0f, -8.0f},
        redWall);

    // right wall green
    addQuadAsTwoTriangles(
        Hobjects, HobjectCount,
        {3.0f, -3.0f, 1.0f}, {3.0f, -3.0f, -8.0f},
        {3.0f, 3.0f, -8.0f}, {3.0f, 3.0f, 1.0f},
        greenWall);

    // floor
    addQuadAsTwoTriangles(
        Hobjects, HobjectCount,
        {-3.0f, -3.0f, 1.0f}, {-3.0f, -3.0f, -8.0f},
        {3.0f, -3.0f, -8.0f}, {3.0f, -3.0f, 1.0f},
        whiteWall);

    // ceiling
    addQuadAsTwoTriangles(
        Hobjects, HobjectCount,
        {-3.0f, 3.0f, 1.0f}, {3.0f, 3.0f, 1.0f},
        {3.0f, 3.0f, -8.0f}, {-3.0f, 3.0f, -8.0f},
        whiteWall);

    // front wall
    addQuadAsTwoTriangles(
        Hobjects, HobjectCount,
        {-3.0f, 3.0f, 1.0f}, {-3.0f, -3.0f, 1.0f},
        {3.0f, -3.0f, 1.0f}, {3.0f, 3.0f, 1.0f},
        whiteWall);

    addSphere(
        Hobjects, HobjectCount,
        {-1.25f, -2.0f, -6.2f}, sphereDiffuse, 1.0f);

    addSphere(
        Hobjects, HobjectCount,
        {1.30f, -1.20f, -5.10f}, sphereGlass, 0.8f);

    addSphere(
        Hobjects, HobjectCount,
        {-0.55f, -2.50f, -3.70f}, sphereAmber, 0.50f);

    */

    for (int i = 0; i < HobjectCount; i++)
    {
        // fill in buildobject struct
        buildObject[i].bounds = boundsOf(Hobjects[i]);
        buildObject[i].centroid = centroidAABB(buildObject[i].bounds);
        buildObject[i].objectIndex = i;
    }

    BVHNode HbvhNodes[256];
    int HbvhObjects[64];
    int HbvhNodeCount = 0;
    int HbvhRootIndex = buildBVH(buildObject, 0, HobjectCount, HbvhNodes, HbvhNodeCount);

    for (int i = 0; i < HobjectCount; i++)
    {
        HbvhObjects[i] = buildObject[i].objectIndex;
    }

    cudaMalloc(&deviceRngStates, screenWidth * screenHeight * sizeof(curandState));
    dim3 block(16, 16);
    dim3 grid(
        (screenWidth + block.x - 1) / block.x,
        (screenHeight + block.y - 1) / block.y);
    initRNGKernel<<<grid, block>>>(deviceRngStates, screenWidth, screenHeight);

    cudaMalloc(&deviceBvhNodes, sizeof(BVHNode) * HbvhNodeCount);
    cudaMalloc(&deviceBvhObjects, sizeof(int) * HobjectCount);
    cudaMalloc(&deviceObjects, sizeof(Object) * HobjectCount);

    cudaMemcpy(deviceBvhNodes, HbvhNodes, sizeof(BVHNode) * HbvhNodeCount, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBvhObjects, HbvhObjects, sizeof(int) * HobjectCount, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceObjects, &Hobjects, sizeof(Object) * HobjectCount, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(bvhRootIndex, &HbvhRootIndex, sizeof(int));
    cudaMemcpyToSymbol(bvhNodeCount, &HbvhNodeCount, sizeof(int));
    cudaMemcpyToSymbol(light, &Hlight, sizeof(Light));
    cudaMemcpyToSymbol(objectCount, &HobjectCount, sizeof(int));
}

float launchRayTracer(void *hostPixels, int screenWidth, int screenHeight, bool useBVH)
{
    // https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
    // as mentioned on the nvidia blog its better to use the inbuilt functions for timings in cuda instead of cpu itmings
    // the way on the blog is the best way to go about it
    static int frameIndex = 0;
    // change to stop destroying events every frame
    static cudaEvent_t start, stop;
    static bool eventReady = false;
    if (!eventReady)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        eventReady = true;
    }

    // 256 threads per block (16x16)
    dim3 blockSize(16, 16);
    dim3 gridSize((screenWidth + blockSize.x - 1) / blockSize.x, (screenHeight + blockSize.y - 1) / blockSize.y);

    // begins once the kernel is launch
    cudaEventRecord(start);
    renderKernel<<<gridSize, blockSize>>>(devicePixels, deviceRngStates, deviceAccumulation, frameIndex, screenWidth, screenHeight, deviceBvhNodes, deviceBvhObjects, deviceObjects, useBVH);
    cudaEventRecord(stop);

    frameIndex++;

    cudaEventSynchronize(stop);
    cudaMemcpy(hostPixels, devicePixels, screenWidth * screenHeight * 4, cudaMemcpyDeviceToHost);

    float ms = 0;
    // calcs the difference
    cudaEventElapsedTime(&ms, start, stop);

    return ms;
}
// reset frames when swithcing from bvh to brute force
void resetAccumulation()
{
    cudaMemset(deviceAccumulation, 0, screenWidth * screenHeight * sizeof(Vec3));
    currentFrame = 0;
}