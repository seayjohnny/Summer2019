#pragma once
#include <GL/freeglut.h>
#include <GL/gl.h>

#include "vecmath.cuh"

void drawPointsF2(float2* points, int n, float pointSize, float color[3])
{
    glPointSize(pointSize);
    // (color == NULL) ? glColor3f(0.0, 0.0, 0.0):glColor3f(color[0], color[1], color[2]);
    
    for(int i = 0; i < n; i++)
    {
        if(i==1) glColor3f(0.0f, 0.0f, 1.0f);
        else glColor3f(color[0], color[1], color[2]);
        
        glBegin(GL_POINTS);
        glVertex2f(points[i].x, points[i].y);
        glEnd();
    }
}

void drawCircleF2(float2 center, float radius, int segments, float thickness, float color[3])
{
    float theta;
    (color == NULL) ? glColor3f(0.0, 0.0, 0.0):glColor3f(color[0], color[1], color[2]);
    glLineWidth(thickness);
    glBegin(GL_LINE_LOOP);
    for(int i = 0; i <= segments; i++)
    {
        theta = i*2.0*PI/segments;
        glVertex2f(radius*cos(theta)+center.x, radius*sin(theta)+center.y);
    }
    glEnd();
}

void drawAxes(float scale, float dx, float dy)
{
    glLineWidth(0.5);
    glColor3f(0.4, 0.4, 0.4);
    glBegin(GL_LINES);
    for(float x = -scale; x <= scale+0.01; x+=dx)
    {
        glVertex2f(x, scale);
        glVertex2f(x, -scale);
    }

    for(float y = -scale; y <= scale+0.01; y+=dy)
    {
        glVertex2f(-scale, y);
        glVertex2f(scale, y);
    }
    glEnd();

    glLineWidth(2.0);
    glColor3f(0.0, 0.0, 0.0);
    glBegin(GL_LINES);
    glVertex2f(-scale, -scale);
    glVertex2f(scale, -scale);

    glVertex2f(-scale, scale);
    glVertex2f(-scale, -scale);

    glEnd();
}

void drawPlot(float* xData, float* yData, int size)
{

    glColor3f(1.0, 0.0, 0.0);
    glBegin(GL_LINE_STRIP);
    for(int i = 0; i < size; i++)
    {
        glVertex2f(xData[i], yData[i]);
    }
    glEnd();
}