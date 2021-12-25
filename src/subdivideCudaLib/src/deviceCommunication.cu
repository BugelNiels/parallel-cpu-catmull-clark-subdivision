#include <stdlib.h>
#include <stdio.h>

#include "deviceCommunication.cuh"
#include "util/util.cuh"


// m = number of vertices in vD; n = number of half edges in vD
void allocateDeviceMemory(DeviceMesh* deviceMesh, int m, int n, int n0, int isQuad) {
	cudaError_t cuda_ret;
	Timer timer;
	printf("Allocating device variables...\n"); fflush(stdout);
    startTime(&timer);
    printf("    Allocating size %d: %d bytes\n", m, m * sizeof(float));
    cuda_ret = cudaMalloc((void**)&deviceMesh->xCoords, m * sizeof(float));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for X coordinates");
    cuda_ret = cudaMalloc((void**)&deviceMesh->yCoords, m * sizeof(float));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for Y coordinates");
	cuda_ret = cudaMalloc((void**)&deviceMesh->zCoords, m * sizeof(float));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for Z coordinates");


    printf("    Allocating size %d: %d bytes\n", n, n * sizeof(int));
	cuda_ret = cudaMalloc((void**)&deviceMesh->twins, n * sizeof(int));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for twin array");
	cuda_ret = cudaMalloc((void**)&deviceMesh->verts, n * sizeof(int));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for vert array");
	cuda_ret = cudaMalloc((void**)&deviceMesh->edges, n * sizeof(int));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for edge array");

    if(isQuad == 0) {
        //only allocate enough for the very first mesh
        cuda_ret = cudaMalloc((void**)&deviceMesh->nexts, n0 * sizeof(int));
        cudaErrCheck(cuda_ret, "Unable to allocate device memory for next array");
        cuda_ret = cudaMalloc((void**)&deviceMesh->prevs, n0 * sizeof(int));
        cudaErrCheck(cuda_ret, "Unable to allocate device memory for prev array");
        cuda_ret = cudaMalloc((void**)&deviceMesh->faces, n0 * sizeof(int));
        cudaErrCheck(cuda_ret, "Unable to allocate device memory for face array");
    }

    stopTime(&timer); printf("Allocation took: %f s\n\n", elapsedTime(timer));
}

int getDeviceVal(int** deviceLoc) {
	cudaError_t cuda_ret;
    int val = 0;
    cuda_ret = cudaMemcpy(&val, *deviceLoc, sizeof(int), cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Unable to copy val to device pointer");
    return val;
}

void reallocHostMemory(Mesh* hostMesh, DeviceMesh* deviceMesh) {

    
    hostMesh->numEdges = getDeviceVal(&deviceMesh->numEdges);
    hostMesh->numFaces = getDeviceVal(&deviceMesh->numFaces);
    hostMesh->numHalfEdges = getDeviceVal(&deviceMesh->numHalfEdges);
    hostMesh->numVerts = getDeviceVal(&deviceMesh->numVerts);

    hostMesh->xCoords = (float*)realloc(hostMesh->xCoords, hostMesh->numVerts);
    hostMesh->yCoords = (float*)realloc(hostMesh->yCoords, hostMesh->numVerts);
    hostMesh->zCoords = (float*)realloc(hostMesh->zCoords, hostMesh->numVerts);

    hostMesh->twins = (int*)realloc(hostMesh->twins, hostMesh->numHalfEdges);
    hostMesh->verts = (int*)realloc(hostMesh->verts, hostMesh->numHalfEdges);
    hostMesh->edges = (int*)realloc(hostMesh->edges, hostMesh->numHalfEdges);
}

void copyHostToDeviceMesh(Mesh* from, DeviceMesh* to, int isQuad) {
	Timer timer;

	printf("Copying mesh from host to device...\n"); fflush(stdout);
    startTime(&timer);

    cudaError_t cuda_ret;

    int m = from->numVerts;
    if(m == 0) {
        printf("Source mesh coords are empty"); 
        return;
    }
    printf("    Copying %d: %d bytes\n", m, m * sizeof(float));
	cuda_ret = cudaMemcpy(to->xCoords, from->xCoords, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy x-coordinates to the device");
	cuda_ret = cudaMemcpy(to->yCoords, from->yCoords, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy y-coordinates to the device")
	cuda_ret = cudaMemcpy(to->zCoords, from->zCoords, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy z-coordinates to the device");

	int n = from->numHalfEdges;
    if(n == 0) {
        printf("Source mesh properties are empty"); 
        return;
    }
    printf("    Copying %d: %d bytes\n", n, n * sizeof(float));
	cuda_ret = cudaMemcpy(to->twins, from->twins, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy twins to the device");
	cuda_ret = cudaMemcpy(to->verts, from->verts, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy verts to the device");
	cuda_ret = cudaMemcpy(to->edges, from->edges, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy edges to the device");
    
    if(isQuad == 0) {
        cuda_ret = cudaMemcpy(to->nexts, from->nexts, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaErrCheck(cuda_ret, "Unable to copy nexts to the device");
        cuda_ret = cudaMemcpy(to->prevs, from->prevs, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaErrCheck(cuda_ret, "Unable to copy prevs to the device"); 
        cuda_ret = cudaMemcpy(to->faces, from->faces, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaErrCheck(cuda_ret, "Unable to copy faces to the device");
    }

	stopTime(&timer); printf("Copy to device took %f s\n\n", elapsedTime(timer));
}

void copyDeviceMeshToHostMesh(Mesh* to, DeviceMesh* from) {
	Timer timer;

	printf("Copying mesh from device back to host...\n"); fflush(stdout);
    startTime(&timer);

    cudaError_t cuda_ret;

    // to already has the correct values for num..
    int m = to->numVerts;
    if(m == 0) {
        printf("Source mesh coords are empty"); 
        return;
    }
    printf("    Copying %d: %d bytes\n", m, m * sizeof(float));
	cuda_ret = cudaMemcpy(to->xCoords, from->xCoords, m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Unable to copy x-coordinates from the device");
	cuda_ret = cudaMemcpy(to->yCoords, from->yCoords, m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Unable to copy y-coordinates from the device")
	cuda_ret = cudaMemcpy(to->zCoords, from->zCoords, m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Unable to copy z-coordinates from the device");

	int n = to->numHalfEdges;
    if(n == 0) {
        printf("Source mesh properties are empty"); 
        return;
    }
    printf("    Copying %d: %d bytes\n", n, n * sizeof(float));
	cuda_ret = cudaMemcpy(to->twins, from->twins, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Unable to copy twins from the device");
	cuda_ret = cudaMemcpy(to->verts, from->verts, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Unable to copy verts from the device");
	cuda_ret = cudaMemcpy(to->edges, from->edges, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Unable to copy edges from the device");
 

	stopTime(&timer); printf("Copy to host took: %f s\n\n", elapsedTime(timer));
}