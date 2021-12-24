#include <stdlib.h>
#include <stdio.h>

#include "deviceCommunication.cuh"
#include "util/util.cuh"

// m = number of vertices in vD; n = number of half edges in vD
void allocateDeviceMemory(Mesh* deviceMesh, int m, int n, int n0, int isQuad) {
	cudaError_t cuda_ret;
	Timer timer;
	printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMalloc((void**)&deviceMesh->xCoords, m * sizeof(float));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for X coordinates");
    cuda_ret = cudaMalloc((void**)&deviceMesh->yCoords, m * sizeof(float));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for Y coordinates");
	cuda_ret = cudaMalloc((void**)&deviceMesh->zCoords, m * sizeof(float));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for Z coordinates");

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

    cudaDeviceSynchronize();
    stopTime(&timer); printf(" %f s\n", elapsedTime(timer));
}

void reallocHostMemory(Mesh* hostMesh, Mesh* deviceMesh) {
    hostMesh->xCoords = (float*)realloc(hostMesh->xCoords, deviceMesh->numVerts);
    hostMesh->yCoords = (float*)realloc(hostMesh->yCoords, deviceMesh->numVerts);
    hostMesh->zCoords = (float*)realloc(hostMesh->zCoords, deviceMesh->numVerts);

    hostMesh->twins = (int*)realloc(hostMesh->twins, deviceMesh->numHalfEdges);
    hostMesh->verts = (int*)realloc(hostMesh->verts, deviceMesh->numHalfEdges);
    hostMesh->edges = (int*)realloc(hostMesh->edges, deviceMesh->numHalfEdges);
}

void copyHostDevice(Mesh* from, Mesh* to, cudaMemcpyKind direction, int isQuad) {
	cudaError_t cuda_ret;

    int m = from->numVerts;
    if(m == 0) {
        printf("Source mesh coords are empty"); 
        return;
    }
	cuda_ret = cudaMemcpy(to->xCoords, from->xCoords, m * sizeof(float), direction);
    cudaErrCheck(cuda_ret, "Unable to copy x-coordinates to the device");
	cuda_ret = cudaMemcpy(to->yCoords, from->yCoords, m * sizeof(float), direction);
    cudaErrCheck(cuda_ret, "Unable to copy y-coordinates to the device")
	cuda_ret = cudaMemcpy(to->zCoords, from->zCoords, m * sizeof(float), direction);
    cudaErrCheck(cuda_ret, "Unable to copy z-coordinates to the device");

	int n = from->numHalfEdges;
    if(n == 0) {
        printf("Source mesh properties are empty"); 
        return;
    }
	cuda_ret = cudaMemcpy(to->twins, from->twins, n * sizeof(int), direction);
    cudaErrCheck(cuda_ret, "Unable to copy twins to the device");
	cuda_ret = cudaMemcpy(to->verts, from->verts, n * sizeof(int), direction);
    cudaErrCheck(cuda_ret, "Unable to copy verts to the device");
	cuda_ret = cudaMemcpy(to->edges, from->edges, n * sizeof(int), direction);
    cudaErrCheck(cuda_ret, "Unable to copy edges to the device");
    
    if(isQuad == 0) {
        cuda_ret = cudaMemcpy(to->nexts, from->nexts, n * sizeof(int), direction);
        cudaErrCheck(cuda_ret, "Unable to copy nexts to the device");
        cuda_ret = cudaMemcpy(to->prevs, from->prevs, n * sizeof(int), direction);
        cudaErrCheck(cuda_ret, "Unable to copy prevs to the device"); 
        cuda_ret = cudaMemcpy(to->faces, from->faces, n * sizeof(int), direction);
        cudaErrCheck(cuda_ret, "Unable to copy faces to the device");
    }
	
	
}

void copyHostToDeviceMesh(Mesh* hostMesh, Mesh* deviceMesh, int isQuad) {
	Timer timer;

	printf("Copying mesh from host to device..."); fflush(stdout);
    startTime(&timer);

	copyHostDevice(hostMesh, deviceMesh, cudaMemcpyHostToDevice, isQuad);

	stopTime(&timer); printf(" %f s\n", elapsedTime(timer));
}

void copyDeviceMeshToHostMesh(Mesh* hostMesh, Mesh* deviceMesh) {
	Timer timer;

	printf("Copying mesh from device back to host..."); fflush(stdout);
    startTime(&timer);

	// copy all data from gpu to host mesh
    copyHostDevice(deviceMesh, hostMesh, cudaMemcpyDeviceToHost, 1);
    hostMesh->numEdges = deviceMesh->numEdges;
    hostMesh->numFaces = deviceMesh->numFaces;
    hostMesh->numHalfEdges = deviceMesh->numHalfEdges;
    hostMesh->numVerts = deviceMesh->numVerts;

	stopTime(&timer); printf(" %f s\n", elapsedTime(timer));
}