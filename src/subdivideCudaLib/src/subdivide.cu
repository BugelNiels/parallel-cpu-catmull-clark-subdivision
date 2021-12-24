#include "../libheaders/subdivide.h"

#include <stdlib.h>
#include <stdio.h>
#include "math.h"

#include "mesh.cuh"
#include "util/util.cuh"
#include "kernelInvoker.cuh"
#include "deviceCommunication.cuh"


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
	// device is synced after this call
	// result is in out
	reallocHostMemory(mesh, &out);	
	copyHostToDeviceMesh(mesh, &out);
	
	cudaDeviceSynchronize();

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

	toObjFile(&baseMesh);

	subdivide(&baseMesh, subdivisionLevel);

	return 0.0;
}
