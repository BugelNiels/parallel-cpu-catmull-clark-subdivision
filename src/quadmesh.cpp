#include "quadmesh.h"

int QuadMesh::next(int h) {
  if (h < 0) {
    return -1;
  }
  return h % 4 == 3 ? h - 3 : h + 1;
}

int QuadMesh::prev(int h) { return h % 4 == 0 ? h + 3 : h - 1; }

int QuadMesh::face(int h) { return h / 4; }

int QuadMesh::cycleLength(int h) { return 4; }

int QuadMesh::getNumberOfFaces() { return numHalfEdges / 4; }
