#include "cudaSubdivision.cuh"

#include <stdlib.h>
#include <stdio.h>
#include "math.h"

#include "util/util.cuh"
#include "kernelInvoker.cuh"
#include "deviceCommunication.cuh"

void cudaSubdivide(Mesh* mesh, int subdivisionLevel) {
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
	
    int isQuad = mesh->nexts == NULL || mesh->prevs == NULL || mesh->faces == NULL ? 1 : 0;

	// TODO: in mesh does not need as much memory only D-1
	allocateDeviceMemory(&in, finalNumberOfVerts, finalNumberOfHalfEdges, mesh->numHalfEdges, isQuad);
	allocateDeviceMemory(&out, finalNumberOfVerts, finalNumberOfHalfEdges, 0, 0);

	cudaDeviceSynchronize();

	copyHostToDeviceMesh(mesh, &in, isQuad);
	
	cudaDeviceSynchronize();

	performSubdivision(in, out, subdivisionLevel);
	// device is synced after this call
	// result is in out
	reallocHostMemory(mesh, &out);	
	
	copyDeviceMeshToHostMesh(mesh, &out);
	
	cudaDeviceSynchronize();
	
	toObjFile(mesh);

	freeDeviceMesh(&in);
	freeDeviceMesh(&out);

	printf("Subdivision Complete!\n");
}