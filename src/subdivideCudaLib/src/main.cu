#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "mesh/objFile.cuh"
#include "mesh/meshInitialization.cuh"
#include "cudaSubdivision.cuh"

int main(int argc, char *argv[]) {

    if(argc < 3) {
        printf("Please provide a subdivision level and filename\n");
        return 0;
    }
    int subdivisionLevel = atoi(argv[1]);
    char* fileName = argv[2];
    char filePath[80] = "../models/";
    strcat(filePath, fileName);
    strcat(filePath, ".obj");

    ObjFile objFile = readObjFromFile(filePath);
    Mesh mesh = meshFromObjFile(objFile);
    freeObjFile(objFile);

	Mesh result = cudaSubdivide(&mesh, subdivisionLevel);
    toObjFile(&result);
    freeMesh(&mesh);
    freeMesh(&result);
    return 0;
}