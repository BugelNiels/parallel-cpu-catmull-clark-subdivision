#include "quadmesh.h"

uint QuadMesh::next(uint h) { return h % 4 == 3 ? h - 3 : h + 1; }

uint QuadMesh::prev(uint h) { return h % 4 == 0 ? h + 3 : h - 1; }

uint QuadMesh::face(uint h) { return h / 4; }

uint QuadMesh::cycleLength(uint h) { return 4; }

uint QuadMesh::getNumberOfFaces() { return numHalfEdges / 4; }
