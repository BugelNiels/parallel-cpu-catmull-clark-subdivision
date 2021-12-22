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

void QuadMesh::insertFacePoints(QuadMesh& mesh) {
#pragma omp for
  for (int h = 0; h < numHalfEdges; h += 4) {
    // everything is a quad and these are stored contiguously in memory.
    // avoids the need for critical sections
      float m = cycleLength(h);
      int i = numVerts + face(h);
      QVector3D c = QVector3D(0,0,0);
      for(int i; i < m; i++){
          int v = vert(h);
          c += vertexCoords[v];
      }
      mesh.vertexCoords[i] = c / m;
  }
}

void QuadMesh::insertEdgePoints(QuadMesh& mesh) {
#pragma omp for
  for (int h = 0; h < numHalfEdges; h++) {
    if (twin(h) < 0) {
      boundaryEdgePoint(mesh, h, numVerts, numFaces);
    } else if (twin(h) > h) {
      smoothEdgePoint(mesh, h, numVerts, numFaces);
      smoothEdgePoint(mesh, twin(h), numVerts, numFaces);
    }
  }
}

void QuadMesh::facePoint(QuadMesh& mesh, int h, int vd) {
  float m = cycleLength(h);
  int v = vert(h);
  int i = vd + face(h);
  QVector3D c = vertexCoords[v] / m;
  mesh.vertexCoords[i] += c;
}

void QuadMesh::smoothEdgePoint(QuadMesh& mesh, int h, int vd, int fd) {
  int v = vert(h);
  int i = vd + face(h);
  int j = vd + fd + edge(h);
  QVector3D c = (vertexCoords[v] + mesh.vertexCoords[i]) / 4.0f;
  mesh.vertexCoords[j] += c;
}
