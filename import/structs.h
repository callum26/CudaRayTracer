#ifndef STRUCTS_H
#define STRUCTS_H

#include <cuda_runtime.h>
#include <cmath>

// stores a 3D vector with coords xyz
// vital part of the whole program
struct Vec3
{
    float x, y, z;

    // calculations for commonly needed vector maths
    // otherVec3 passed in via reference to prevent not needed copies
    // const so it doesnt get changed while calcs r running
    // trailing const to not modifiy obj called
    __host__ __device__ Vec3 operator-(const Vec3 &otherVec3) const
    {
        return {x - otherVec3.x, y - otherVec3.y, z - otherVec3.z};
    }

    __host__ __device__ Vec3 operator+(const Vec3 &otherVec3) const
    {
        return {x + otherVec3.x, y + otherVec3.y, z + otherVec3.z};
    }

    __host__ __device__ Vec3 operator/(const Vec3 &otherVec3) const
    {
        return {x / otherVec3.x, y / otherVec3.y, z / otherVec3.z};
    }

    __host__ __device__ Vec3 operator*(const Vec3 &otherVec3) const
    {
        return {x * otherVec3.x, y * otherVec3.y, z * otherVec3.z};
    }

    // a(vector) * b(scalar) = a.x * b, a.y * b, a.z * b
    __host__ __device__ Vec3 operator*(float scalar) const
    {
        return {x * scalar, y * scalar, z * scalar};
    }

    // more complex vector maths

    // a dot b = a.x * b.x + a.y * b.y + a.z * b.z
    __host__ __device__ float dot(const Vec3 &otherVec3) const
    {
        return (x * otherVec3.x) + (y * otherVec3.y) + (z * otherVec3.z);
    }

    // normalise = u / magnitude of (u)

    // magnitude equation is given as = sqrt(x^2 + y^2 + z^2)
    // this takes all xyz components of vec3 passed in
    // dots them with themself return mag as result
    __host__ __device__ float magnitude() const
    {
        return sqrtf(dot(*this));
    }

    // normalinng making vector have a magnitude of 1 (unit vector) (for not 0 vectors) therefore its components r direction only
    // direction is needed a lot regardless of magnitude so its better to normalise them then use for calculations
    __host__ __device__ Vec3 normalise() const
    {
        // faster method to calculate normalise
        // rsqrtf reciproical of square root
        float inverseSquare = rsqrtf(x * x + y * y + z * z);
        // then multiply instead cheaper
        return {x * inverseSquare, y * inverseSquare, z * inverseSquare};
    }

    /*EXPLAIN THIS LATER*/
    __host__ __device__ Vec3 cross(const Vec3 &otherVec3) const
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

    // for glass maybe water etc
    float transparency;
    float refraction; // how much it bends light
};

struct Light
{
    Vec3 position;
    // ia, imd, ims
    Vec3 ambientIntensity;
    Vec3 diffuseIntensity;
    Vec3 specularIntensity;
    float lightIntensity;
    float lightRadius;
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
    Vec3 v0, v1, v2;
};

struct Ray
{
    Vec3 origin;
    Vec3 direction;
    Vec3 hitPoint;
};

// starting BVH to optimise calculations
// to start we need to create a bounding box struct for our object
// commonly used is a axis aligned bounding box or AABB
// a 3D box containing the object where edges are locked to axes and not roated
// cube so we only need min axes and max
struct AABB
{
    Vec3 boxMin, boxMax;
};

// as a BVH is a tree structure we need a tree structure :)
// node doesnt
struct BVHNode
{
    // contains our bounding box
    AABB box;
    // and its index within the tree (for our nodes)
    int leftIndex, rightIndex;
    // we also need to store where the object array starts in a seprate arry
    int firstObject;
    // and how many object there are within the leaf node
    int objectCount;
};

// also we need a seperate struct to store object index refferencing its bounds and centroid
// a centroid is basically the centre of mass of given object
// we need the centre of the bounding box quite simple
struct BuildObject
{
    AABB bounds;
    Vec3 centroid;
    int objectIndex;
};

struct SAHBucket
{
    int count;
    AABB bounds;
};

#endif
