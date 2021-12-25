#include "cudaSubdivision.cuh"

#include <stdlib.h>
#include <stdio.h>
#include "math.h"
#include <assert.h>

#include "util/util.cuh"
#include "kernelInvoker.cuh"
#include "deviceCommunication.cuh"

void verifyMesh(Mesh* mesh) {
	assert(mesh->xCoords != NULL);
	assert(mesh->yCoords != NULL);
	assert(mesh->zCoords != NULL);

	assert(mesh->twins != NULL);
	assert(mesh->edges != NULL);
	assert(mesh->verts != NULL);
	printf("Mesh verified\n");
}

void cudaSubdivide(Mesh* mesh, int subdivisionLevel) {
	cudaError_t cuda_ret;
	verifyMesh(mesh);
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
	printf("final he: %d -- final v: %d\n", finalNumberOfHalfEdges, finalNumberOfVerts);


    int isQuad = mesh->nexts == NULL || mesh->prevs == NULL || mesh->faces == NULL;
	// TODO: in mesh does not need as much memory only D-1
	allocateDeviceMemory(&in, finalNumberOfVerts, finalNumberOfHalfEdges, mesh->numHalfEdges, isQuad);
	allocateDeviceMemory(&out, finalNumberOfVerts, finalNumberOfHalfEdges, 0, 0);

	cuda_ret = cudaDeviceSynchronize();
	cudaErrCheck(cuda_ret, "Unable to sync");

	copyHostToDeviceMesh(mesh, &in, isQuad);
	
	cuda_ret = cudaDeviceSynchronize();
	cudaErrCheck(cuda_ret, "Unable to sync");

	Mesh result = performSubdivision(in, out, subdivisionLevel, mesh->numHalfEdges);
	// device is synced after this call
	// result is in out
	reallocHostMemory(mesh, &result);	

	cuda_ret = cudaDeviceSynchronize();
	cudaErrCheck(cuda_ret, "Unable to sync");
	
	copyDeviceMeshToHostMesh(mesh, &result);
	
	cuda_ret = cudaDeviceSynchronize();
	cudaErrCheck(cuda_ret, "Unable to sync");
	
	// toObjFile(mesh);

	freeDeviceMesh(&in);
	freeDeviceMesh(&out);

	printf("Subdivision Complete!\n");
}