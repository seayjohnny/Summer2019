#pragma once
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <unistd.h>

#include "vecmath.cuh"
#include "walls.h"
#include "lintrans.cuh"


using namespace std;



class Node
{
public:
    float2 initPosition;
    float2 position;
    float2 velocity;
    float2 acceleration;
    float mass;
    float2 currentForce;
    int order;

    Node()
    {
        position = make_float2(0.0f, 0.0f);
        velocity = make_float2(0.0f, 0.0f);
        acceleration = make_float2(0.0f, 0.0f);
        mass = 80.0f;
        currentForce = make_float2(0.0f, 0.0f);
        order = 0;
    };
    
};

class NBodySim
{
public:
    long N;
    Node* h_nodes;
    Node* d_nodes;
    CircleWall* h_walls;
    CircleWall* d_walls;

    float* h_angles;
    float* d_angles;

    int* h_order;
    int* d_order;

    float dt, dr_factor;
    double dr;
    double wallStrength;
    float damp, force_cutoff;
    int lowerPressureLimit, upperPressureLimit;

    float p, q, m;

    bool draw;
    bool renderInit;
    float2* nodePositions;

    float* timeline;
    int historyLen;
    float* h_history;
    float* d_history;
    
    double nBodyCost;
    double optimalCost;
    bool hasOptimalCost;
    float percentError;

    bool deconstructing;

    void loadParametersFromConfigFile(string configFilename)
    {
        ifstream cfg_input( configFilename.c_str() );
        if( cfg_input.is_open() )
        {
            string line;
            while( getline(cfg_input, line) )
            {
                line.erase(remove_if(line.begin(), line.end(), ::isspace), line.end());
                if( line[0] == '#' || line.empty() ) continue;
                int delimiterPos = line.find("=");
                string name = line.substr(0, delimiterPos);
                string value = line.substr(delimiterPos+1);
                cout << name << " " << value << endl;

                if(name == "P") p = atof(value.c_str());
                if(name == "Q") q = atof(value.c_str());
                if(name == "M") m = atof(value.c_str());

                if(name == "DT") dt = atof(value.c_str());
                if(name == "DR_FACTOR") dr_factor = atof(value.c_str());
                if(name == "WALL_STRENGTH") wallStrength = atof(value.c_str());
                if(name == "DAMP") damp = atof(value.c_str());
                if(name == "FORCE_CUTOFF") force_cutoff = atof(value.c_str());

                if(name == "LOWER_PRESSURE_LIMIT") lowerPressureLimit = atoi(value.c_str());
                if(name == "UPPER_PRESSURE_LIMIT") upperPressureLimit = atoi(value.c_str());

                if(name == "HISTORY_LENGTH") historyLen = atoi(value.c_str());
            }
        }
        else { cerr << "Couldn't open config file for reading." << endl;  }
    };

    void loadDataFromDatasetFiles(string datasetName)
    {
        // Construct filenames used to load dataset                
        string co_filename = "./datasets/" + datasetName + ".co";
        string cfg_filename = "./datasets/" + datasetName + ".cfg";
        
        ifstream cfg_input( cfg_filename.c_str() );
        if( cfg_input.is_open() )
        {
            string line;
            while( getline(cfg_input, line) )
            {
                line.erase(remove_if(line.begin(), line.end(), ::isspace), line.end());
                if( line[0] == '#' || line.empty() ) continue;
                int delimiterPos = line.find("=");
                string name = line.substr(0, delimiterPos);
                string value = line.substr(delimiterPos+1);

                if(name == "N") N = atoi(value.c_str());
                if(name == "OPTIMAL_COST"){
                    optimalCost = atof(value.c_str());
                    hasOptimalCost = true;
                } 
            }
        }
        if(optimalCost == 0) hasOptimalCost = false;
        
        // Initialize node array and read in positions to host arrat
        cout << "Reading in node coordinates from \"" << datasetName + ".co\"..." << endl;
        h_nodes = (Node*)malloc(N*sizeof(Node));
        nodePositions = (float2*)malloc(N*sizeof(float2));
        ifstream co_input( co_filename.c_str() );
        float x, y;
        for (int i = 0; i < N; i++) {
            co_input >> x >> y;
            nodePositions[i] = make_float2(x, y);
        }
        cout << "Successfully loaded in coordinates." << endl;

        cout << "Adjusting geometric center to (0, 0)..." << endl;
        shiftPointsF2(nodePositions, N, -1.0*getGeometricCenterF2(nodePositions, N));
        for (int i = 0; i < N; i++) {
            h_nodes[i] = Node();
            h_nodes[i].position = h_nodes[i].initPosition = nodePositions[i];
        }

        // Allocate node array on device and copy data into it
        cout << "Allocating node array on device and copying host array to device array..." << endl;
        cudaMalloc((void**)&d_nodes, N*sizeof(Node));
        cudaMemcpy((void*)d_nodes, h_nodes, N*sizeof(Node), cudaMemcpyHostToDevice);
        cout << "Host to device copy complete. (node array)" << endl;
    };

    NBodySim(){}; 

    NBodySim(string datasetName) : draw(true), renderInit(true)
    {
        
        
        loadParametersFromConfigFile("./configs/default.cfg");
        loadDataFromDatasetFiles(datasetName);
        initForceHistory();
        initWalls();
        initAngles();
        initOrder();
 
        deconstructing = false;
        run();
    };

    ~NBodySim()
    {
        if(!deconstructing) {
            deconstructing = true;
            cout << "Deconstructing simulation..." << endl;
            // free(h_nodes); free(h_walls); free(h_history);
            cudaFree((void*)d_nodes); cudaFree(d_walls); cudaFree(d_history);
        }
    };

    void initWalls()
    {
        h_walls = (CircleWall*)malloc(2*sizeof(CircleWall));

        // Outer wall
        h_walls[0].radius = getFurthestNodeDist(); 
        h_walls[0].direction = 0;
        h_walls[0].pressure = 0.0f;

        // Inner wall
        h_walls[1].radius = 0.0f; 
        h_walls[1].direction = 1;
        h_walls[1].pressure = 0.0f;

        dr = dr_factor*h_walls[0].radius;

        cudaMalloc((void**)&d_walls, 2*sizeof(CircleWall));
        cudaMemcpy((void*)d_walls, h_walls, 2*sizeof(CircleWall), cudaMemcpyHostToDevice);
    };

    void initForceHistory()
    {
        timeline = (float*)malloc(historyLen*sizeof(float));
        for(int i = 0; i < historyLen; i++) timeline[i] = 0.0f;
        h_history = (float*)malloc(N*historyLen*sizeof(float));
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < historyLen; j++)
            {
                h_history[j + i*historyLen] = 0.0f;
            }
        }
        cudaMalloc((void**)&d_history, N*historyLen*sizeof(float));
        cudaMemcpy((void*)d_history, h_history, N*historyLen*sizeof(float), cudaMemcpyHostToDevice);
    };

    void initAngles()
    {
        h_angles = (float*)malloc(N*sizeof(float));
        for(int i = 0; i < N; i++) h_angles[i] = 0.0f;

        cudaMalloc((void**)&d_angles, N*sizeof(float));
        cudaMemcpy((void*)d_angles, h_angles, N*sizeof(float), cudaMemcpyHostToDevice);

    };

    void initOrder()
    {
        h_order = (int*)malloc(N*sizeof(int));
        for(int i = 0; i < N; i++) h_order[i] = 0;

        cudaMalloc((int**)&d_order, N*sizeof(int));
        cudaMemcpy((int*)d_order, h_order, N*sizeof(int), cudaMemcpyHostToDevice);

    };

    float2* getNodePositions()
    {
        for(int i = 0; i < N; i++) nodePositions[i] = h_nodes[i].position;
        return nodePositions;
    };

    float2* getNodeInitPositions()
    {
        for(int i = 0; i < N; i++) nodePositions[i] = h_nodes[i].initPosition;
        return nodePositions;
    };

    float2* getNodeInitPositionsOrdered()
    {
        for(int i = 0; i < N; i++) nodePositions[h_nodes[i].order] = h_nodes[i].initPosition;
        return nodePositions;
    };

    float getFurthestNodeDist()
    {
        float dist;
        float max = 0.0;
        for(int i = 0; i < N; i++)
        {
            dist = mag(nodePositions[i]);
            if(dist > max) max = dist;
        }

        return max;
    };

    void recordTimeline(float t)
    {
        for(int i = historyLen - 2; i >=0; i--)
        {
            timeline[i + 1] = timeline[i];
        }
        timeline[0] = t;
    };

    double getNBodyPathCost()
    {
        nBodyCost = 0.0;
        for(int i = 0; i <= N; i++)
        {
            nBodyCost += dist(h_nodes[h_order[i%N]].initPosition, h_nodes[h_order[(i+1)%N]].initPosition);
        }

        return nBodyCost;
    };

    float getPercentError()
    {
        if(hasOptimalCost)
        {
            percentError = 100.0*(nBodyCost - optimalCost)/optimalCost;
            return percentError;
        }
        return -100.0;
    }

private:
    void run();
};
