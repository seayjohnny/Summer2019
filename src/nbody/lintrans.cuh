#pragma once
#include "vecmath.cuh"
#include "walls.h"

__host__ __device__ void normalizeArray(float* array, int n, float normFactor = 1.0)
{
    float max = 0.0f;
    float min = 1e12f;    
    int i;
    for(i = 0; i < n; i++)
    {
        if(array[i] > max) max = array[i];
        if(array[i] < min) min = array[i];
    }

    if(max-min != 0){
        for(i = 0; i < n; i++)
        {
            array[i] = normFactor*(array[i] - min)/(max - min);
        }
    }
}

__host__ __device__ void shiftArray(float* array, int n, float shift) { for(int i = 0; i < n; i++) {array[i] += shift; }}

__host__ __device__ float normalizePointsF2(float2* points, int n, float normFactor = 1.0) 
{
    float dist;
    float max = 0;
    float factor;

    int i;
    for(i = 0; i < n; i++)
    {
        dist = mag(points[i]);
        if(dist > max) max = dist;
    }

    for(i = 0; i < n; i++) 
    {
        factor = normFactor/max;
        points[i] *= factor;
    }

    return factor;
}

__host__ __device__ void shiftPointsF2(float2* points, int n, float2 shift){ for(int i = 0; i < n; i++){ points[i] += shift;} }
__host__ __device__ void scalePointsF2(float2* points, int n, float scale){ for(int i = 0; i < n; i++){ points[i] *= scale;} }
__host__ __device__ float2 getGeometricCenterF2(float2* points, int n)
{
    float2 center = {0.0f, 0.0f};
    for(int i = 0; i < n; i++)  center += points[i];
    center /= n;
    return center;
}

__host__ __device__ float normalizeRadiiF2(float* radii, int n, float normFactor = 1.0)
{
    float max = 0;
    float factor;
    float dist;
    
    int i;
    for(i = 0; i < n; i++)
    {
        dist = radii[i];
        if(dist > max) max = dist;
    }

    for(i = 0; i < n; i++) 
    {
        factor = normFactor/max;
        radii[i] *= factor;
    }

    return factor;
}

float normalizeWalls(CircleWall* wall, int n, float normFactor = 1.0)
{
    float max = 0;
    float factor;
    float dist;
    
    int i;
    for(i = 0; i < n; i++)
    {
        dist = wall[i].radius;
        if(dist > max) max = dist;
    }

    for(i = 0; i < n; i++) 
    {
        factor = normFactor/max;
        wall[i].radius *= factor;
    }

    return factor;
}