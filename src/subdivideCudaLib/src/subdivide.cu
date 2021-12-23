#include "../libheaders/subdivide.h"

#include "mesh.h"
#include "stdio.h"
#include "util.h"
#include "math.h"
#include "kernel.h"

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

void copyHostToDeviceMesh(Mesh* hostMesh, Mesh* deviceMesh) {
	cudaError_t cuda_ret;
	Timer timer;

	printf("Copying mesh from host to device..."); fflush(stdout);
    startTime(&timer);

	int m = hostMesh->numVerts;
	cuda_ret = cudaMemcpy(&deviceMesh->xCoords, &hostMesh->xCoords, m * sizeof(float),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
	cuda_ret = cudaMemcpy(&deviceMesh->yCoords, &hostMesh->yCoords, m * sizeof(float),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
	cuda_ret = cudaMemcpy(&deviceMesh->zCoords, &hostMesh->zCoords, m * sizeof(float),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

	int n = hostMesh->numHalfEdges;
	cuda_ret = cudaMemcpy(&deviceMesh->twins, &hostMesh->twins, n * sizeof(float),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
	cuda_ret = cudaMemcpy(&deviceMesh->nexts, &hostMesh->nexts, n * sizeof(float),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
	cuda_ret = cudaMemcpy(&deviceMesh->prevs, &hostMesh->prevs, n * sizeof(float),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
	cuda_ret = cudaMemcpy(&deviceMesh->verts, &hostMesh->verts, n * sizeof(float),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
	cuda_ret = cudaMemcpy(&deviceMesh->edges, &hostMesh->edges, n * sizeof(float),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
	cuda_ret = cudaMemcpy(&deviceMesh->faces, &hostMesh->faces, n * sizeof(float),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

	stopTime(&timer); printf("%f s\n", elapsedTime(timer));
}

void copyDeviceMeshToHostMesh(Mesh* hostMesh, Mesh* deviceMesh) {
	// resize host mesh
	// copy all data from gpu to device mesh
	cudaError_t cuda_ret;
	Timer timer;

	printf("Copying mesh from device back to host..."); fflush(stdout);
    startTime(&timer);

	stopTime(&timer); printf("%f s\n", elapsedTime(timer));
}

void subdivide(Mesh* mesh, int subdivisionLevel) {
	printf("Starting Subdvision\n");
	
	// use double buffering; calculate final number of half edges and numVerts and allocat out and in arrays
	// switch each subdivision level
	Mesh in = makeEmptyCopy(mesh);
	Mesh out = makeEmptyCopy(mesh);

	int finalNumberOfHalfEdges = pow(4, subdivisionLevel) * mesh->numHalfEdges;
	// assumes quad mesh
	int v1 = mesh->numVerts + mesh->numFaces + mesh->numEdges;
	int e1 = 2 * mesh->numEdges + mesh->numHalfEdges;
	int f1 = mesh->numHalfEdges;
	int finalNumberOfVerts = v1 + pow(2, subdivisionLevel - 1) * (e1 + (pow(2, subdivisionLevel) -1) * f1);

	allocateDeviceMemory(&in, finalNumberOfVerts, finalNumberOfHalfEdges);
	allocateDeviceMemory(&out, finalNumberOfVerts, finalNumberOfHalfEdges);

	cudaDeviceSynchronize();

	copyHostToDeviceMesh(mesh, &in);
	
	cudaDeviceSynchronize();

	performSubdivision(in, out, subdivisionLevel);

	copyDeviceMeshToHostMesh(mesh, &out);

	freeDeviceMesh(&in);
	freeDeviceMesh(&out);

	printf("Subdivision Complete!\n");
}

// returns the number of milsecs the subdivision took
double timedSubdivision(float* xCoords, float* yCoords, float* zCoords, int numVerts, int numHalfEdges, int numFaces, int numEdges, int* twins, int* nexts, int* prevs, int* verts, int* edges, int* faces, int subdivisionLevel) {
	printf("Setting up mesh\n");
	Mesh baseMesh = initMesh(numVerts, numHalfEdges, numFaces, numEdges);

	baseMesh.xCoords = xCoords;
	baseMesh.yCoords = yCoords;
	baseMesh.zCoords = zCoords;

	baseMesh.twins = twins;
	baseMesh.nexts = nexts;
	baseMesh.prevs = prevs;
	baseMesh.verts = verts;
	baseMesh.edges = edges;
	baseMesh.faces = faces;
	subdivide(&baseMesh, subdivisionLevel);

	return 0.0;
}
