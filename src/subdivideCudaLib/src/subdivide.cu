#include "../libheaders/subdivide.h"

#include "mesh.h"
#include "stdio.h"

Mesh initMesh(int numVerts, int numHalfEdges, int numFaces, int numEdges) {
	Mesh mesh;
	mesh.numVerts = numVerts;
	mesh.numHalfEdges = numHalfEdges;
	mesh.numEdges = numEdges;
	mesh.numFaces = numFaces;

	mesh.xCoords = (float*)malloc(numVerts * sizeof(float));
	mesh.yCoords = (float*)malloc(numVerts * sizeof(float));
	mesh.zCoords = (float*)malloc(numVerts * sizeof(float));
	return mesh;
} 

void subdivide(Mesh mesh, int subdivisionLevel) {
	printf("ok");
}

// returns the number of milsecs the subdivision took
double timedSubdivision(float** vertCoords, int numVerts, int numHalfEdges, int numFaces, int numEdges, int* twins, int* nexts, int* prevs, int* verts, int* edges, int* faces, int subdivisionLevel) {
	Mesh baseMesh = initMesh(numVerts, numHalfEdges, numFaces, numEdges);
	for(int i = 0; i < numVerts; ++i) {
		baseMesh.xCoords[i] = vertCoords[i][0];
		baseMesh.yCoords[i] = vertCoords[i][1];
		baseMesh.zCoords[i] = vertCoords[i][2];
	}
	baseMesh.twins = twins;
	baseMesh.nexts = nexts;
	baseMesh.prevs = prevs;
	baseMesh.verts = verts;
	baseMesh.edges = edges;
	baseMesh.faces = faces;
	subdivide(baseMesh, subdivisionLevel);

	return 0.0;
}
