#include "../libheaders/subdivide.h"

#include "mesh/mesh.cuh"
#include "util/util.cuh"
#include "cudaSubdivision.cuh"


// returns the number of milsecs the subdivision took
void meshSubdivision(float* xCoords, float* yCoords, float* zCoords, int numVerts, int numHalfEdges, int numFaces, int numEdges, int* twins, int* nexts, int* prevs, int* verts, int* edges, int* faces, int subdivisionLevel) {
	printf("Setting up regular mesh\n");
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

	cudaSubdivide(&baseMesh, subdivisionLevel);
}

void quadMeshSubdivision(float* xCoords, float* yCoords, float* zCoords, int numVerts, int numHalfEdges, int numFaces, int numEdges, int* twins, int* verts, int* edges, int subdivisionLevel) {
	printFloatArr(xCoords, numVerts);
	printFloatArr(yCoords, numVerts);
	printFloatArr(zCoords, numVerts);
	printIntArr(twins, numHalfEdges);
	printIntArr(verts, numHalfEdges);
	printIntArr(edges, numHalfEdges);

	printf("\n\n%d, %d, %d, %d\n\n", numVerts, numHalfEdges, numFaces, numEdges);

	printf("Setting up quad mesh\n");
	Mesh baseMesh = initMesh(numVerts, numHalfEdges, numFaces, numEdges);

	baseMesh.xCoords = xCoords;
	baseMesh.yCoords = yCoords;
	baseMesh.zCoords = zCoords;

	baseMesh.twins = twins;
	baseMesh.verts = verts;
	baseMesh.edges = edges;

	cudaSubdivide(&baseMesh, subdivisionLevel);
}