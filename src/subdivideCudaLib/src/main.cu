#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "mesh/objFile.cuh"
#include "mesh/meshInitialization.cuh"
#include "cudaSubdivision.cuh"

#define BUFFER_SIZE 80

char* createObjFilePath(char const* dir, char const* name) {
    char* filePath = (char*)malloc(BUFFER_SIZE * sizeof(char));
    strcpy(filePath, dir);
    strcat(filePath, name);
    strcat(filePath, ".obj");
    return filePath;
}

int main(int argc, char *argv[]) {

    if(argc < 3) {
        printf("Please provide a subdivision level and filename\n");
        return 0;
    }
    int subdivisionLevel = atoi(argv[1]);
    char* filePath = createObjFilePath("../models/", argv[2]);
    char* resultPath = NULL;
    if(argc == 4) {
        resultPath = createObjFilePath("/", argv[3]);
    }

    ObjFile objFile = readObjFromFile(filePath);
    free(filePath);
    Mesh mesh = meshFromObjFile(objFile);
    freeObjFile(objFile);

	Mesh result = cudaSubdivide(&mesh, subdivisionLevel);
    if(resultPath != NULL) {
        // TODO: use path
        toObjFile(&result);
    }
    freeMesh(&mesh);
    freeMesh(&result);
    return 0;
}