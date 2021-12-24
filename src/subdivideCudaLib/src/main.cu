#include <stdlib.h>
#include <stdio.h>

#include "mesh/objFile.cuh"
#include "mesh/meshInitialization.cuh"
#include "cudaSubdivision.cuh"

int main(int argc, char *argv[]) {
    ObjFile objFile = readObjFromFile("../models/OpenCube.obj");
    Mesh baseMesh = meshFromObjFile(objFile);
	cudaSubdivide(&baseMesh, 3);
}