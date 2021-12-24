#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "objFile.cuh"

void printObjFile(ObjFile obj) {
    for(int i = 0; i < obj.numVerts; i++) {
        printf("v %lf %lf %lf\n", obj.xCoords[i], obj.yCoords[i], obj.zCoords[i]);
    }

    printf("\n");
    for(int i = 0; i < obj.numFaces; i++) {
        printf("f");
        for(int j = 0; j < obj.faceValencies[i]; j++) {
            printf(" %d", obj.faceIndices[i][j]);
        }
        printf("\n");
    }
}

ObjFile readObjFromFile(char const* objFileName) {
    FILE *objFile = fopen(objFileName, "r");
    if (objFile == NULL)
    {
        printf("Error opening .obj file!\n");
        exit(1);
    }
    char * line = NULL;
    size_t len = 0;

    ObjFile obj;
    obj.numVerts = 0;
    obj.numFaces = 0;
    char *token;

    // TODO: only works if there are no leading whitespaces
    while (getline(&line, &len, objFile) != -1) {
        if(line[0] == 'v') {
            obj.numVerts++;
        } else if(line[0] == 'f') {
            obj.numFaces++;        }
        // printf("%s -- %d\n", line, len);
    }
    rewind(objFile);
    obj.xCoords = (float*)malloc(obj.numVerts * sizeof(float));
    obj.yCoords = (float*)malloc(obj.numVerts * sizeof(float));
    obj.zCoords = (float*)malloc(obj.numVerts * sizeof(float));

    obj.faceIndices = (int**)malloc(obj.numFaces * sizeof(int*));
    obj.faceValencies = (int*)malloc(obj.numFaces * sizeof(int));

    int v = 0;
    int f = 0;
    while (getline(&line, &len, objFile) != -1) {
        if(line[0] == 'v') {
            // remove the v
            strsep(&line, " ");
            obj.xCoords[v] = atof(strsep(&line, " "));
            obj.yCoords[v] = atof(strsep(&line, " "));
            obj.zCoords[v] = atof(strsep(&line, " "));
            v++;
            // vertex
        } else if(line[0] == 'f') {
            // remove the f
            strsep(&line, " ");
            int currentSize = 4;
            int* indices = (int*)malloc(currentSize * sizeof(int));
            int i = 0;
            while((token = strsep(&line, " "))) {
                
                if(i >= currentSize) {
                    currentSize *= 2;
                    indices = (int*)realloc(indices, currentSize);
                }
                indices[i] = atoi(token);
                i++;
            }
            obj.faceIndices[f] = indices;
            obj.faceValencies[f] = i;
            f++;
        }
    }
    fclose(objFile);
    printObjFile(obj);
    return obj;
}