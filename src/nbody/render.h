#pragma once
#include <GL/freeglut.h>
#include <GL/gl.h>

#include "NBodySim.cuh"
#include "walls.h"
#include "lintrans.cuh"
#include "draw.cuh"

#define DIM 768

class RenderConfig {
public:
    float SimScale, GraphScale;
    int SimWindowWidth, SimWindowHeight;
    int GraphWindowWidth, GraphWindowHeight;

    RenderConfig() {
        loadParametersFromConfigFile("./configs/render.cfg");
    };

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

                if(name == "SCALE") SimScale = GraphScale = atof(value.c_str());
                if(name == "SIM_WINDOW_WIDTH") SimWindowWidth = atoi(value.c_str());
                if(name == "SIM_WINDOW_HEIGHT") SimWindowHeight = atoi(value.c_str());
                if(name == "SIM_SCALE") SimScale = atof(value.c_str());
                if(name == "GRAPH_SCALE") GraphScale = atof(value.c_str());
                if(name == "GRAPH_WINDOW_WIDTH") GraphWindowWidth = atoi(value.c_str());
                if(name == "GRAPH_WINDOW_HEIGHT") GraphWindowHeight = atoi(value.c_str());

                if(name == "SIM_WINDOW_DIM")
                {
                    float temp = atoi(value.c_str());
                    SimWindowHeight = temp;
                    SimWindowWidth = temp;
                }
                
            }
        }
        else { cerr << "Couldn't open config file for reading." << endl;  }
    };
};

RenderConfig renderConfig;
NBodySim sim;
float2* points;
CircleWall* walls;

float* forces;
float* xrange;

int closed;
int clicked;

GLint SimWindow, GraphWindow, ForceWindow;

float Aged[3] = {1.0, 253.0/255.0, 240.0/255.0};
float LightGray0[3] = {220.0/255.0, 220.0/255.0, 220.0/255.0};
float LightGray1[3] = {206.0/255.0, 206.0/255.0, 206.0/255.0};
float LightGray2[3] = {188.0/255.0, 188.0/255.0, 188.0/255.0};
float DarkGray2[3] = {50.0/255.0, 50.0/255.0, 50.0/255.0};
float DarkGray3[3] = {89.0/255.0, 89.0/255.0, 89.0/255.0};
float Green[3] = {66.0/255.0, 244.0/255.0, 143.0/255.0};
float Red[3] = {244.0/255.0, 65.0/255.0, 65.0/255.0};
float Blue[3] = {109.0/255.0, 158.0/255.0, 235.0/255.0};
float White[3] = {1.0, 1.0, 1.0};
float Black[3] = {0.0, 0.0, 0.0};

void copyPoints(float2* dst, float2* src, int n){ for(int i = 0; i < n; i++){ dst[i] = src[i]; } }
void copyWalls(CircleWall* dst, CircleWall* src){ for(int i = 0; i < 2; i++){ dst[i] = src[i]; } }
void copyForces(float* dst, float* src, int offset)
{
    for(int i = 0; i < sim.historyLen; i++)
    {
        dst[i] = src[i + offset*sim.historyLen];
    }
}
void copyTimeline(float* dst, float* src){ for(int i = 0; i < sim.historyLen; i++){ dst[i] = src[i]; } }

void simRender()
{
    glutSetWindow(SimWindow);
    glClear(GL_COLOR_BUFFER_BIT);
    copyPoints(points, sim.getNodePositions(), sim.N);

    // float normFactor = normalizePointsF2(points, sim.N, 1.0);
    scalePointsF2(points, sim.N, renderConfig.SimScale);
    drawPointsF2(points, sim.N, 10.0, Red);

    copyWalls(walls, sim.h_walls);
    // normalizeWalls(walls, 2, 1.0);
    drawCircleF2(make_float2(0.0f, 0.0f), walls[1].radius*renderConfig.SimScale, 100, 3.0, Black);
    drawCircleF2(make_float2(0.0f, 0.0f), walls[0].radius*renderConfig.SimScale, 100, 3.0, Black);
    glutSwapBuffers();

}

void pathRender()
{
    glutSetWindow(SimWindow);
    glClear(GL_COLOR_BUFFER_BIT);
    copyPoints(points, sim.getNodeInitPositionsOrdered(), sim.N);

    float normFactor = normalizePointsF2(points, sim.N, 1.0);
    scalePointsF2(points, sim.N, renderConfig.SimScale);
    drawPointsF2(points, sim.N, 10.0, Red);

    glBegin(GL_LINE_LOOP);
    for(int i = 0; i <= sim.N; i++)
    {
        glVertex2f(points[i%sim.N].x, points[i%sim.N].y);
        glVertex2f(points[(i+1)%sim.N].x, points[(i+1)%sim.N].y);
    }
    glEnd();
    glutSwapBuffers();
}

void graphRender()
{
    glutSetWindow(GraphWindow);
    glClear(GL_COLOR_BUFFER_BIT);
    copyForces(forces, sim.h_history, 0);
    copyTimeline(xrange, sim.timeline);

    drawAxes(renderConfig.GraphScale, 0.1f, 0.1f);

    normalizeArray(forces, sim.historyLen, 2*renderConfig.GraphScale);
    shiftArray(forces, sim.historyLen, -renderConfig.GraphScale);
    
    normalizeArray(xrange, sim.historyLen, 2*renderConfig.GraphScale);
    shiftArray(xrange, sim.historyLen, -renderConfig.GraphScale);

    drawPlot(xrange, forces, sim.historyLen);

    glutSwapBuffers();
}

void close()
{
    closed = 1;
}

void mouse(int button, int state, int x, int y)
{
    if(button==0)
    {
        clicked = 1;
    }
}

void nBodyRenderInit(NBodySim* nbodysim)
{
    sim = *nbodysim;
    if(sim.renderInit) {
        renderConfig = RenderConfig();
        points = (float2*)malloc(sim.N*sizeof(float2));
        walls = (CircleWall*)malloc(2*sizeof(CircleWall));
        forces = (float*)malloc(sim.historyLen*sizeof(float));
        xrange = (float*)malloc(sim.historyLen*sizeof(float));

        int argc = 0;
        char** argv = (char**)malloc(sizeof(char*));
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_MULTISAMPLE);

        glEnable(GL_MULTISAMPLE_ARB);
        glEnable(GL_POINT_SMOOTH);
        glEnable(GL_LINE_SMOOTH);
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
        glEnable(GL_BLEND);
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
            GLUT_ACTION_GLUTMAINLOOP_RETURNS);

        glutInitWindowSize(renderConfig.SimWindowWidth, renderConfig.SimWindowHeight);
        glutInitWindowPosition(0, 0);
        SimWindow = glutCreateWindow("TSP N-Body");
        glutDisplayFunc(simRender);
        glutCloseFunc(close);
        glutMouseFunc(mouse);
        glClearColor(1.0,1.0,1.0,1.0);

        glutInitWindowSize(renderConfig.GraphWindowWidth, renderConfig.GraphWindowHeight);
        glutInitWindowPosition(renderConfig.SimWindowWidth, 0);
        GraphWindow = glutCreateWindow("Force Graph");
        glutDisplayFunc(graphRender);
        glClearColor(1.0,1.0,1.0,1.0);

        sim.renderInit = false;
        glutMainLoopEvent();
    }
}

void nBodyRenderUpdate(NBodySim* nbodysim)
{
    sim = *nbodysim;
    simRender();
}

void graphRenderUpdate(NBodySim* nbodysim)
{
    sim = *nbodysim;
    graphRender();
}

void nBodyPathRender(NBodySim* nbodysim)
{
    sim = *nbodysim;
    pathRender();
}


bool waitForMouseClick()
{
    glutMainLoopEvent();
    return !clicked;
}

int getWindowState(){ return closed ? 0:1;}
int hasMouseClicked(){ return clicked;}