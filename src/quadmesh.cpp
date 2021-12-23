#include "quadmesh.h"

int QuadMesh::next(int h) { return h % 4 == 3 ? h - 3 : h + 1; }

int QuadMesh::prev(int h) { return h % 4 == 0 ? h + 3 : h - 1; }

int QuadMesh::face(int h) { return h / 4; }

void QuadMesh::subdivideCatmullClark(QuadMesh& mesh) {
  recalculateSizes(mesh);
  mesh.resizeBuffers();

#pragma omp parallel
  {
// Half Edge Refinement Rules
#pragma omp for nowait
    for (int h = 0; h < numHalfEdges; ++h) {
      edgeRefinement(mesh, h, numVerts, numFaces, numEdges);
    }
#pragma omp for
    for (int h = 0; h < numHalfEdges; h += 4) {
      // everything is a quad and these are stored contiguously in memory.
      // avoids the need for critical sections
      QVector3D c;
      for (int j = 0; j < 4; j++) {
        int v = vert(h + j);
        c += vertexCoords.at(v);
      }
      int i = numVerts + face(h);
      mesh.vertexCoords[i] = c / 4.0f;
    }

#pragma omp for
    for (int h = 0; h < numHalfEdges; ++h) {
      if (twin(h) < 0) {
        boundaryEdgePoint(mesh, h, numVerts, numFaces);
      } else if (twin(h) > h) {
        smoothEdgePoint(mesh, h, numVerts, numFaces);
        smoothEdgePoint(mesh, twin(h), numVerts, numFaces);
      }
    }

#pragma omp for
    for (int h = 0; h < numHalfEdges; ++h) {
      // val = -1 if boundary vertex
      float val = valence(h);
      if (val < 0) {
        boundaryVertexPoint(mesh, h);
      } else {
        smoothVertexPoint(mesh, h, numVerts, numFaces, val);
      }
    }
  }
}
