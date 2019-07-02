#pragma once
#include <string>
#include <sstream>

using namespace std;

//          foreground background
// black        30         40
// red          31         41
// green        32         42
// yellow       33         43
// blue         34         44
// magenta      35         45
// cyan         36         46
// white        37         47

// reset             0  (everything back to normal)
// bold/bright       1  (often a brighter shade of the same colour)
// underline         4
// inverse           7  (swap foreground and background colours)
// bold/bright off  21
// underline off    24
// inverse off      27


string esc = "\033[";
string hide_cursor = esc + "25l";

// Change Horizontal Absolute
string cha(int column)
{
    ostringstream stringStream;
    stringStream << esc << column << "G";
    return stringStream.str();
}

// Change text color to RGB
string rgb(int r, int g, int b)
{
    ostringstream stringStream;
    stringStream << esc << "38;2;" << r << ";" << g << ";" << b << "m";
    return stringStream.str();
}

string reset = esc + "0m";
string red = esc + "31m";

// 79 45 127
string purp = esc + "38;2;79;45;127m";

