#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <unistd.h>


#include "NBodySim.cuh"
#include "vecmath.cuh"
#include "render.h"
#include "curandfuncs.cuh"
#include "ansi.h"

// #include <cuda.h>
// #include <cuda_runtime.h>

using namespace std;

__global__ void bodyToBodyForces(Node* nodes, long N, float p, float q, float m) {
    
    // shPos holds both current positions and initial positions.
    // The first half is current, the second half is intiail.
    extern __shared__ float2 shPos[];

    int idx = threadIdx.x;
    float2 pos = shPos[idx] = nodes[idx].position;
    float2 initPos = shPos[idx + N] = nodes[idx].initPosition;

    float2 force = {0.0f, 0.0f};
    float forceMag;
    float d, natLength;
    float h, c;

    __syncthreads();
    for( int i = 0; i < N; i++ ) {
        if(i != idx) {
            d = dist(shPos[i], pos) + 0.01;
            natLength = dist(shPos[i + N], initPos);

            h = m*(powf(powf(q/p, 1/(q-p))*natLength, p))/(1 - p/q);
            c = powf(natLength/d, q-p);
            forceMag = (c-1)*h/powf(d, p);

            force += make_float2(forceMag*(shPos[i].x - pos.x)/d, forceMag*(shPos[i].y - pos.y)/d);
            // if(idx==0) printf("Thread %i force: <%f, %f>\n", idx, force.x, force.y);
        }
    }
    
    nodes[idx].currentForce += force;
}

__global__ void bodyToWallForces(Node* nodes, CircleWall* walls, double wallStrength)
{
    int idx = threadIdx.x;
    float forceMag;
    float2 force = {0.0f, 0.0f};
    float2 pos = nodes[idx].position;    
    float radius = mag(pos);

    if(radius < walls[1].radius) {
        forceMag = wallStrength*(walls[1].radius - radius);
        force += make_float2(forceMag*pos.x/radius, forceMag*pos.y/radius);
        atomicAdd(&walls[1].pressure, abs(forceMag)/(2.0*PI*walls[1].radius));
    }
    else if (radius > walls[0].radius) {
        forceMag = wallStrength*(walls[0].radius - radius);
        force += make_float2(forceMag*pos.x/radius, forceMag*pos.y/radius);
        atomicAdd(&walls[0].pressure, abs(forceMag)/(2.0*PI*walls[0].radius));
    }

    nodes[idx].currentForce += force;
}

__global__ void recordForces(Node* nodes, float* history, int historyLen)
{
    int idx = threadIdx.x;
    int i = idx*historyLen;
    for(int j = historyLen-2; j >= 0; j--)
    {
        history[i+j+1] = history[i+j];
    }
    history[i] = nodes[idx].currentForce.x;
}

__global__ void recordPressure(CircleWall* walls, float* history, int historyLen)
{
    int idx = threadIdx.x;
    int i = idx*historyLen;
    for(int j = historyLen-2; j >= 0; j--)
    {
        history[i+j+1] = history[i+j];
    }
    history[i] = walls[idx].pressure;
}

__global__ void applyForces(Node* nodes, float dt, float damp, float force_cutoff)
{
    int idx = threadIdx.x;
    Node node = nodes[idx];
    node.currentForce += -damp*node.velocity;
    if(abs(node.currentForce.x) > force_cutoff) node.currentForce.x = 0;
    if(abs(node.currentForce.y) > force_cutoff) node.currentForce.y = 0;
    node.acceleration = node.currentForce/node.mass;


    node.velocity += node.acceleration*dt;
    node.position += node.velocity*dt;

    node.currentForce = make_float2(0.0f, 0.0f);
    nodes[idx] = node;
    
}

__global__ void moveWalls(CircleWall* walls, float dr, int lowerLimit, int upperLimit)
{
    CircleWall outerWall = walls[0];
    CircleWall innerWall = walls[1];
    if(outerWall.pressure > upperLimit)
    {
        outerWall.direction = 1.0 + 100*(outerWall.pressure - upperLimit)/upperLimit;
        innerWall.direction = 0;
    }
    else if(outerWall.pressure < lowerLimit)
    {
        outerWall.direction = -1.0 + (outerWall.pressure-lowerLimit)/lowerLimit;
        innerWall.direction = 0;
    }
    else
    {
        outerWall.direction = 0;
        innerWall.direction = 1;
    }

    outerWall.pressure = 0;

    outerWall.radius += outerWall.direction*dr;
    innerWall.radius += innerWall.direction*dr;

    walls[0] = outerWall;
    walls[1] = innerWall;
}

__global__ void perturbate(curandState* state, Node* nodes, float amount)
{
    int idx = threadIdx.x;
    nodes[idx].position.x += amount*rand_uni_range(state, -1.0, 1.0);
    nodes[idx].position.y += amount*rand_uni_range(state, -1.0, 1.0);
}

__global__ void getAngles(Node* nodes, float* angles)
{
    int idx = threadIdx.x;
    float theta = dangle(nodes[idx].position, make_float2(1.0f, 0.0f));
    if(nodes[idx].position.y < 0) theta = 360 - theta;
    angles[idx] = theta;
}

__global__ void getOrder(Node* nodes, float* angles, int* order, int n)
{
    extern __shared__ float shAngles[];
    
    int idx = threadIdx.x;
    float mAngle = shAngles[idx] = angles[idx];

    int num_of_smaller = 0;
    for(int i = 0; i < n; i++) if(mAngle > shAngles[i]) num_of_smaller++;

    nodes[idx].order = num_of_smaller;
    order[num_of_smaller] = idx;
}

void printDynamicString(string s)
{
    cout << hide_cursor + cha(1) << s + cha(s.length() + 5) << reset << flush;

}

void NBodySim::run()
{
    dim3 grid(1);
    dim3 block(N);

    curandState *d_state;
    cudaMalloc(&d_state, sizeof(curandState)*N);
    init_rand_state<<<grid, block>>>(d_state);

    if(draw)
    {
        cudaMemcpy(this->h_nodes, this->d_nodes, N*sizeof(Node), cudaMemcpyDeviceToHost);
        nBodyRenderInit(this);
        while(waitForMouseClick());
    }

    float t = 0.0;
    float wallTimer = 0.0;
    float recordTimer = 0.0;
    int drawTimer = 0;
    cout << "Starting run..." << endl;
    while( h_walls[1].radius < h_walls[0].radius - 2*dr) {
        // cout << hide_cursor + cha(1) + rgb(0, 255, 0) << floor(100 * h_walls[1] / (h_walls[0] - dr)) << "%    " + cha(10) << reset << flush;
        bodyToBodyForces<<<grid, block, 2*N*sizeof(float2)>>>(d_nodes, N, p, q, m);
        bodyToWallForces<<<grid, block>>>(d_nodes, d_walls, wallStrength);

        if(recordTimer > 0.1) {
            recordForces<<<grid, block>>>(d_nodes, d_history, historyLen);
            // recordPressure<<<grid, 2>>>(d_walls, d_history, historyLen);
            recordTimeline(t);
            cudaMemcpy(this->h_history, this->d_history, historyLen*sizeof(float), cudaMemcpyDeviceToHost);
            graphRenderUpdate(this);
            recordTimer = 0;
        }

        applyForces<<<grid, block>>>(d_nodes, dt, damp, force_cutoff);

        if(wallTimer > dt) {
            moveWalls<<<grid, 1>>>(d_walls, dr, lowerPressureLimit, upperPressureLimit);
            cudaMemcpy(this->h_walls, this->d_walls, 2*sizeof(CircleWall), cudaMemcpyDeviceToHost);
            wallTimer = 0;
        }

        if(draw && getWindowState() && drawTimer == 100) {
            cudaMemcpy(this->h_nodes, this->d_nodes, N*sizeof(Node), cudaMemcpyDeviceToHost);
            nBodyRenderUpdate(this);
            drawTimer = 0;
        }

        if(draw) drawTimer++;
        wallTimer += dt;
        recordTimer += dt;
        t += dt;
    }

    getAngles<<<grid, block>>>(d_nodes, d_angles);
    getOrder<<<grid, block, N*sizeof(float)>>>(d_nodes, d_angles, d_order, N);

    cudaMemcpy(this->h_nodes, this->d_nodes, N*sizeof(Node), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->h_angles, this->d_angles, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->h_order, this->d_order, N*sizeof(float), cudaMemcpyDeviceToHost);

    getNBodyPathCost();
    getPercentError();
    cout << "Cost: " << nBodyCost << endl;
    cout << "Percent Error: " << percentError << "%" << endl;
    if(draw) while(getWindowState()) nBodyPathRender(this);
}

int main()
{
    NBodySim sim("zi929");
    return 0;
}
