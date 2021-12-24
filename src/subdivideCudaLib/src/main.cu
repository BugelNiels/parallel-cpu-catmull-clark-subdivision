#include <stdlib.h>
#include <stdio.h>

#include "mesh/objFile.cuh"
#include "mesh/meshInitialization.cuh"

int main(int argc, char *argv[]) {
    ObjFile objFile = readObjFromFile("../models/OpenCube.obj");
    Mesh mesh = meshFromObjFile(objFile);
}