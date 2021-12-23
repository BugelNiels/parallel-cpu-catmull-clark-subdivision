#include "../libheaders/subdivide.h"

#include "mesh.h"
#include "stdio.h"

Mesh initMesh(int numVerts, int numHalfEdges, int numFaces, int numEdges) {
	Mesh mesh;
	mesh.numVerts = numVerts;
	mesh.numHalfEdges = numHalfEdges;
	mesh.numEdges = numEdges;
	mesh.numFaces = numFaces;
	return mesh;
} 

void subdivide(Mesh mesh, int subdivisionLevel) {
	printf("ok");
}

// returns the number of milsecs the subdivision took
double timedSubdivision(float* xCoords, float* yCoords, float* zCoords, int numVerts, int numHalfEdges, int numFaces, int numEdges, int* twins, int* nexts, int* prevs, int* verts, int* edges, int* faces, int subdivisionLevel) {
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
	subdivide(baseMesh, subdivisionLevel);

	return 0.0;
}
