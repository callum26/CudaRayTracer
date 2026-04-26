#ifndef BVH_CUH
#define BVH_CUH

#include <cuda_runtime.h>
#include "structs.h"

// com of bounding box quite simple
__host__ __device__ Vec3 centroidAABB(const AABB &box)
{
    return {
        0.5f * (box.boxMin.x + box.boxMax.x),
        0.5f * (box.boxMin.y + box.boxMax.y),
        0.5f * (box.boxMin.z + box.boxMax.z)};
}

// now we need to create bounds of each type of object
/* COULD DEFINE SPHERE OUT OF TRIANGLE WOULD BE VERY COMPLEX BUT WOULD REMOVE A LOT OF REDUNNAT CKDE BUT MAY BE OUT OF SCOPE OF THIS PROJECT  */
__host__ __device__ AABB boundsOf(const Object &object)
{
    AABB box;
    if (object.type == sphereObject)
    {
        // we can use radius of sphere to calc the bounding box
        Vec3 radius = {object.radius, object.radius, object.radius};
        // object.pos is centre of object so subtracting or adding of the radius is gonna give the edges
        // around the sphere, which is the bounding box coords
        box.boxMin = object.position - radius;
        box.boxMax = object.position + radius;
    }
    else
    {
        // for triangle
        // taking the minimum of each verticies for every coord
        box.boxMin = {
            fminf(object.v0.x, fminf(object.v1.x, object.v2.x)),
            fminf(object.v0.y, fminf(object.v1.y, object.v2.y)),
            fminf(object.v0.z, fminf(object.v1.z, object.v2.z))};
        // taking the max of each verticies for every coord
        box.boxMax = {
            fmaxf(object.v0.x, fmaxf(object.v1.x, object.v2.x)),
            fmaxf(object.v0.y, fmaxf(object.v1.y, object.v2.y)),
            fmaxf(object.v0.z, fmaxf(object.v1.z, object.v2.z))};
        // creating easy bounding box
    }
    return box;
}

// using new direct aray index
__host__ __device__ float getAxisComponent(const Vec3 &vector, int axisIndex)
{
    return vector.v[axisIndex];
}

// happens a lot so quite useful to have
// init an empty AABB
__host__ __device__ AABB emptyAABB()
{
    AABB box;
    box.boxMin = {INFINITY, INFINITY, INFINITY};
    box.boxMax = {-INFINITY, -INFINITY, -INFINITY};
    return box;
}

// union of two AABB is the smallest box that contains both of them
// basically encapuslating two boxes
__host__ __device__ AABB unionAABB(const AABB &leftBox, const AABB &rightBox)
{
    // finding min/max of each coord of boxes to find min max of union box
    AABB box;
    box.boxMin = {
        fminf(leftBox.boxMin.x, rightBox.boxMin.x),
        fminf(leftBox.boxMin.y, rightBox.boxMin.y),
        fminf(leftBox.boxMin.z, rightBox.boxMin.z)};
    box.boxMax = {
        fmaxf(leftBox.boxMax.x, rightBox.boxMax.x),
        fmaxf(leftBox.boxMax.y, rightBox.boxMax.y),
        fmaxf(leftBox.boxMax.z, rightBox.boxMax.z)};
    return box;
}

// to find surface area of an aabb obvs same as surface area of a cube
// however we dont have the lwd only min and max points
// but we can work it out from that
// sA = 2(wh + wd + hd)
// where  w = x(max) - x(min) h = y(max) - y(min) d = z(max) - z(min)
__host__ __device__ float surfaceAreaAABB(const AABB &box)
{
    // cant have negative area ensures extent >= 0
    // dont really know if thats possible tho but might as well
    Vec3 extent = {
        fmaxf(0.0f, box.boxMax.x - box.boxMin.x), // width
        fmaxf(0.0f, box.boxMax.y - box.boxMin.y), // height
        fmaxf(0.0f, box.boxMax.z - box.boxMin.z)  // depth
    };

    // 2(wh + wd + hd)
    return 2.0f * (extent.x * extent.y + extent.x * extent.z + extent.y * extent.z);
}

__device__ bool hitAABB(const Ray &ray, const AABB &box)
{
    // simialr to our ray intersection we need to find
    // the ray distance min and max to find where we should focus the intersection on
    // if ray is outside of these ranges it hasnt hit the box
    float rayMinDistance = 0.001f;
    float rayMaxDistance = INFINITY;

    // R = O + tD
    // same with other interesctions if we assume a point P to be our ray equation
    // P = O + tD
    // sub in the min and max point for our AABB we calc
    // P(min) = O + tD
    // P(max) = O + tD
    // but unlike sphere intersection test we know what points we are calculating
    // we need to rearrange to find for t(min) and t(max)
    // t(min) = (P - O) / D
    // t(max) = (P - O) / D
    // each xyz is calculated respective of one another so we repeat the equation 3 times
    // so need to loop through each axes

    for (int i = 0; i < 3; i++)
    {
        // get all components need for each
        float rayOrigin = getAxisComponent(ray.origin, i);
        float rayDirection = getAxisComponent(ray.direction, i);
        float boxMin = getAxisComponent(box.boxMin, i);
        float boxMax = getAxisComponent(box.boxMax, i);

        // classic bcos we r diving more than once store it
        float inverseDirection = 1.0f / rayDirection;
        float rayNearHit = (boxMin - rayOrigin) * inverseDirection;
        float rayFarHit = (boxMax - rayOrigin) * inverseDirection;

        // if direction is negative we need to swap the near and far as they will be inversed
        // ensuring near is always the smallest and far is always largest
        if (inverseDirection < 0.0f)
        {
            float temp = rayNearHit;
            rayNearHit = rayFarHit;
            rayFarHit = temp;
        }

        // clamp them to our bounds at the top of the func
        rayMinDistance = fmaxf(rayMinDistance, rayNearHit);
        rayMaxDistance = fminf(rayMaxDistance, rayFarHit);
        if (rayMaxDistance < rayMinDistance)
        {
            return false; // if min exceeds max then it must no longe be in box
        }
    }
    return true;
}

// happened a lot so created a helper to set up a leaf node with the given range of objects
__host__ void createLeaf(BVHNode &node, int start, int count)
{
    node.firstObject = start; // index into bvhObjects
    node.objectCount = count; // amount of objects starting from the index
    // no children as its a leaf node
    node.leftIndex = -1;
    node.rightIndex = -1;
}

// only runs on cpu at scene load traversal on gpu
// we go through every object building the tree top down recursively
__host__ int buildBVH(BuildObject *objs, int start, int end, BVHNode *nodes, int &nodeCount)
{
    // if we test at every possible poistion to find where to split it taxes O(n^2) per axis
    // to solve this we split each axis into buckets which are split ranges of the total axis range
    // e.g. if our x axis is 0-4.0 and we split into 12 buckets that gives us ranges of 0.33
    // bucket0 0-0.33 bucket1 0.33-0.66 etc
    // then each of the objects within that range are stored in their resprctive bucket
    static const int sahBucketAmount = 12;
    // cost of an aabb test its relative to cIntereset
    static const float cTraverse = 1.0f;
    // cost of an obj intersection test for now 1.5f
    /* CURRENTLY NOT ACCURATE RATIO COULD MEASURE ACTUALLY HOW LONG IT TAKES TO GENERATE A PROPER RATIO BETWEEN INTERSECTION*/
    static const float cIntersect = 1.5f;
    // hard cap on objs per leaf regardless of SAH decision
    static const int maxLeafSize = 4;

    // post inc nodeCount passing ref so we can fill node with data
    int nodeIdx = nodeCount++;
    BVHNode &node = nodes[nodeIdx];

    // init a new box infinite bounds
    node.box = emptyAABB();
    // union all object in this range expanding this node box
    // parent surface area comes from this and is used in the SAH probability ratio later
    for (int i = start; i < end; i++)
    {
        node.box = unionAABB(node.box, objs[i].bounds);
    }
    // ending with a box containg all the range objects

    // the amount of objects in current range
    int count = end - start;

    // small enough to be a leaf no point splitting further
    if (count <= maxLeafSize)
    {
        createLeaf(node, start, count);
        return nodeIdx;
    }

    // centroid bounds r separate from object bounds
    // splitting on where object centers are distributed gives better bucket placement than using full extents
    // ensures balanced distro of objects across split
    AABB centroidBounds = emptyAABB();

    // accumulate bounding box of all object centroids
    for (int i = start; i < end; i++)
    {
        // each centroid is represented as a degenerate box with 0x0x0 dimensions
        AABB centroidAsBox;
        // as box min and box max are same coords
        centroidAsBox.boxMin = objs[i].centroid;
        centroidAsBox.boxMax = objs[i].centroid;
        // we can then expan to include all centroids
        // basically comparing the coords of each centroid for every ovject
        // creating smallest box possible
        centroidBounds = unionAABB(centroidBounds, centroidAsBox);
    }

    // calculate the surface area of this current nodes bounding box
    float parentArea = surfaceAreaAABB(node.box);

    // if area is 0 objects are coplanar and we cannot divide by area in sah calc anyways
    // force it to be a leaf
    if (parentArea <= 0.0f)
    {
        createLeaf(node, start, count);
        return nodeIdx;
    }

    // now finding best split
    // init extreme values
    float bestCost = INFINITY;
    int bestAxis = -1;
    int bestSplit = -1;

    // try all 3 axes picking the one with the cheapest split
    for (int axis = 0; axis < 3; axis++)
    {
        float centroidMin = getAxisComponent(centroidBounds.boxMin, axis);
        float centroidMax = getAxisComponent(centroidBounds.boxMax, axis);
        // range of centroid on current axis
        float extent = centroidMax - centroidMin;

        // if range is tiny skip cant split
        if (extent <= 1e-6f)
        {
            continue;
        }

        // init (currently 12) buckets each one tracks count and union bounds of objs within it
        // and this will of course happen for every axis
        SAHBucket buckets[sahBucketAmount];

        // for each bucket init an empty box and set count
        // we will increase this later once we know have many objects within it
        for (int i = 0; i < sahBucketAmount; i++)
        {
            buckets[i].count = 0;
            buckets[i].bounds = emptyAABB();
        }

        // map each objs centroid to a bucket along the current axis
        // formula spreads centroid range evenly across sahBucket
        for (int i = start; i < end; i++)
        {
            // get centroid position of current object along current axis
            float centroidPos = getAxisComponent(objs[i].centroid, axis);

            // map centroid to a bucket index between 0 and bucketAmount -1
            // normalize the centroid position to range 0 to 1
            // (centroidPos - centroidMin) / extent
            // then scale to bucket count
            // cast int get bucket index
            int bucketIndex = int(sahBucketAmount * ((centroidPos - centroidMin) / extent));

            // clamp to valid bucket range to handle floating point edge cases as we are casting int
            if (bucketIndex < 0)
                bucketIndex = 0;
            if (bucketIndex >= sahBucketAmount)
                bucketIndex = sahBucketAmount - 1;

            // increase current bucket obj count and add its bounds for each object along each axis
            buckets[bucketIndex].count++;
            buckets[bucketIndex].bounds = unionAABB(buckets[bucketIndex].bounds, objs[i].bounds);
        }

        // need accumulated bounds and counts for left right sides of every split
        // init boxs for left right split
        AABB leftBounds[sahBucketAmount - 1];
        AABB rightBounds[sahBucketAmount - 1];
        // and respective object counts for each box
        int leftCounts[sahBucketAmount - 1];
        int rightCounts[sahBucketAmount - 1];

        // we create a running count for each left right so we can determine how many objs in total
        // init empty left right box and respective count
        AABB runningLeftBounds = emptyAABB();
        int runningLeftCount = 0;
        AABB runningRightBounds = emptyAABB();
        int runningRightCount = 0;

        // for all buckets going forward
        for (int i = 0; i < sahBucketAmount - 1; i++)
        {
            // include all counts of objects within specifc bucket index
            runningLeftCount += buckets[i].count;
            // and its repsective bounds
            runningLeftBounds = unionAABB(runningLeftBounds, buckets[i].bounds);
            // and then can store them in the array object count and its bounds
            leftCounts[i] = runningLeftCount;
            leftBounds[i] = runningLeftBounds;
        }

        // same for the right but starting at the last bucket and moving backwards
        // stores results at i -1 to align with left for when splitting
        for (int i = sahBucketAmount - 1; i >= 1; i--)
        {
            runningRightCount += buckets[i].count;
            runningRightBounds = unionAABB(runningRightBounds, buckets[i].bounds);
            rightCounts[i - 1] = runningRightCount;
            rightBounds[i - 1] = runningRightBounds;
        }

        // eval SAH cost for each of the splits on this axis

        // again for each ucket
        for (int i = 0; i < sahBucketAmount - 1; i++)
        {
            // if either side empty cant split here pointless
            if (leftCounts[i] == 0 || rightCounts[i] == 0)
            {
                continue;
            }

            // compute sA of each splits bounding box
            float leftArea = surfaceAreaAABB(leftBounds[i]);
            float rightArea = surfaceAreaAABB(rightBounds[i]);
            // then we can calc the split cost using the formila
            //// Cost = cT + (SA(L)/SA(P)) * NL * cI + (SA(R)/SA(P)) * NR * cI
            // N being the counts of the respective objects
            float splitCost = cTraverse + (leftArea / parentArea) * leftCounts[i] * cIntersect + (rightArea / parentArea) * rightCounts[i] * cIntersect;

            // then store cheapest split found across all axes and boundaries
            if (splitCost < bestCost)
            {
                bestCost = splitCost;
                bestAxis = axis;
                bestSplit = i;
            }
        }
    }

    // compare best split against cost of just keeping this node as a leaf
    float leafCost = count * cIntersect;
    // if its more beneficial aka if bestcost if greater than leafcost then we might as well leaf it
    if (bestAxis < 0 || bestCost >= leafCost)
    {
        createLeaf(node, start, count);
        return nodeIdx;
    }

    // now we r out of the loop we know the best axis and the best split
    // get boundingbox of our best axis and its range
    float centroidMin = getAxisComponent(centroidBounds.boxMin, bestAxis);
    float centroidMax = getAxisComponent(centroidBounds.boxMax, bestAxis);
    float extent = centroidMax - centroidMin;

    // partition objs using same bucket mapping as the sweep above
    // anything in bucket <= bestSplit goes left rest go right
    int mid = start;
    for (int i = start; i < end; i++)
    {
        // centroid pos of best axis
        float centroidPos = getAxisComponent(objs[i].centroid, bestAxis);
        // same maths to get bucket index
        int bucketIndex = int(sahBucketAmount * ((centroidPos - centroidMin) / extent));
        if (bucketIndex < 0)
            bucketIndex = 0;
        if (bucketIndex >= sahBucketAmount)
            bucketIndex = sahBucketAmount - 1;

        // move obj based on best split
        // e.g. if  bestSplit = 5
        // buckets 0-5 go left
        // buckets 6-11 right
        // mid to track boundary between L and R
        if (bucketIndex <= bestSplit)
        {
            BuildObject temp = objs[mid];
            objs[mid] = objs[i];
            objs[i] = temp;
            mid++;
        }
    }

    // fallback if partition is rassed split half and half instead
    if (mid == start || mid == end)
    {
        mid = start + count / 2;
    }

    // internal node so no object list here, recurse into children
    node.firstObject = -1;
    node.objectCount = 0;
    node.leftIndex = buildBVH(objs, start, mid, nodes, nodeCount);
    node.rightIndex = buildBVH(objs, mid, end, nodes, nodeCount);
    return nodeIdx;
}

#endif