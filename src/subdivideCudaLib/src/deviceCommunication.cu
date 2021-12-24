#include <stdlib.h>
#include <stdio.h>

#include "deviceCommunication.cuh"
#include "util/util.cuh"

// m = number of vertices in vD; n = number of half edges in vD
void allocateDeviceMemory(Mesh* deviceMesh, int m, int n) {
	cudaError_t cuda_ret;
	Timer timer;
	printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMalloc((void**)&deviceMesh->xCoords, m * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for X coordinates");
    cuda_ret = cudaMalloc((void**)&deviceMesh->yCoords, m * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for Y coordinates");
	cuda_ret = cudaMalloc((void**)&deviceMesh->zCoords, m * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for Z coordinates");

	cuda_ret = cudaMalloc((void**)&deviceMesh->twins, n * sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for twin array");
	cuda_ret = cudaMalloc((void**)&deviceMesh->nexts, n * sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for next array");
	cuda_ret = cudaMalloc((void**)&deviceMesh->prevs, n * sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for prev array");
	cuda_ret = cudaMalloc((void**)&deviceMesh->verts, n * sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for vert array");
	cuda_ret = cudaMalloc((void**)&deviceMesh->edges, n * sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for edge array");
	cuda_ret = cudaMalloc((void**)&deviceMesh->faces, n * sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for face array");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
}

void reallocHostMemory(Mesh* hostMesh, Mesh* deviceMesh) {
    hostMesh->xCoords = (float*)realloc(hostMesh->xCoords, deviceMesh->numVerts);
    hostMesh->yCoords = (float*)realloc(hostMesh->yCoords, deviceMesh->numVerts);
    hostMesh->zCoords = (float*)realloc(hostMesh->zCoords, deviceMesh->numVerts);

    hostMesh->twins = (int*)realloc(hostMesh->twins, deviceMesh->numHalfEdges);
    hostMesh->nexts = (int*)realloc(hostMesh->nexts, deviceMesh->numHalfEdges);
    hostMesh->prevs = (int*)realloc(hostMesh->prevs, deviceMesh->numHalfEdges);
    hostMesh->verts = (int*)realloc(hostMesh->verts, deviceMesh->numHalfEdges);
    hostMesh->edges = (int*)realloc(hostMesh->edges, deviceMesh->numHalfEdges);
    hostMesh->faces = (int*)realloc(hostMesh->faces, deviceMesh->numHalfEdges);
}

void copyHostDevice(Mesh* from, Mesh* to, cudaMemcpyKind direction) {
	cudaError_t cuda_ret;

    int m = from->numVerts;
	cuda_ret = cudaMemcpy(&to->xCoords, &from->xCoords, m * sizeof(float),
        direction);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
	cuda_ret = cudaMemcpy(&to->yCoords, &from->yCoords, m * sizeof(float),
        direction);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
	cuda_ret = cudaMemcpy(&to->zCoords, &from->zCoords, m * sizeof(float),
        direction);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

	int n = from->numHalfEdges;
	cuda_ret = cudaMemcpy(&to->twins, &from->twins, n * sizeof(int),
        direction);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
	cuda_ret = cudaMemcpy(&to->nexts, &from->nexts, n * sizeof(int),
        direction);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
	cuda_ret = cudaMemcpy(&to->prevs, &from->prevs, n * sizeof(int),
        direction);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
	cuda_ret = cudaMemcpy(&to->verts, &from->verts, n * sizeof(int),
        direction);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
	cuda_ret = cudaMemcpy(&to->edges, &from->edges, n * sizeof(int),
        direction);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
	cuda_ret = cudaMemcpy(&to->faces, &from->faces, n * sizeof(int),
        direction);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
}

void copyHostToDeviceMesh(Mesh* hostMesh, Mesh* deviceMesh) {
	Timer timer;

	printf("Copying mesh from host to device..."); fflush(stdout);
    startTime(&timer);

	copyHostDevice(hostMesh, deviceMesh, cudaMemcpyHostToDevice);

	stopTime(&timer); printf("%f s\n", elapsedTime(timer));
}

void copyDeviceMeshToHostMesh(Mesh* hostMesh, Mesh* deviceMesh) {
	Timer timer;

	printf("Copying mesh from device back to host..."); fflush(stdout);
    startTime(&timer);

	// copy all data from gpu to host mesh
    copyHostDevice(deviceMesh, hostMesh, cudaMemcpyDeviceToHost );

	stopTime(&timer); printf("%f s\n", elapsedTime(timer));
}