#include "quadmesh.h"

/**
 * @brief QuadMesh::QuadMesh Creates an empty quad mesh with buffers of the
 * correct size
 * @param maxHalfEdges Maximum number of half-edges this mesh will hold
 * @param maxVertices Maximum number of vertices this mesh will hold
 */
QuadMesh::QuadMesh(int maxHalfEdges, int maxVertices) {
  twins.resize(maxHalfEdges);
  verts.resize(maxHalfEdges);
  edges.resize(maxHalfEdges);
  vertexCoords.resize(maxVertices);
}

/**
 * @brief QuadMesh::resetMesh Resets the vertex coordinates of this mesh
 */
void QuadMesh::resetMesh() { vertexCoords.fill({0, 0, 0}); }

/**
 * @brief QuadMesh::next Calculates the index of the next half-edge for a quad
 * mesh.
 * @param h Half-edge index to calculate the next index of
 * @return Next half-edge index of h
 */
int QuadMesh::next(int h) { return h % 4 == 3 ? h - 3 : h + 1; }

/**
 * @brief QuadMesh::prev Calculates the index of the previous half-edge for a
 * quad mesh.
 * @param h Half-edge index to calculate the previous index of
 * @return Previous half-edge index of h
 */
int QuadMesh::prev(int h) { return h % 4 == 0 ? h + 3 : h - 1; }

/**
 * @brief QuadMesh::face Calculates the index of the face this half-edge belongs
 * to in a quad mesh.
 * @param h Half-edge index to calculate the face index of
 * @return Faceindex of h
 */
int QuadMesh::face(int h) { return h / 4; }

/**
 * @brief QuadMesh::subdivideCatmullClark Performs a single Catmull-Clark
 * subdivision step and saves the result in the provided argument mesh.
 * @param mesh The mesh to store the result of this subdivision step in.
 */
void QuadMesh::subdivideCatmullClark(QuadMesh& mesh) {
  recalculateSizes(mesh);

#pragma omp parallel
  {
// Half Edge Refinement Rules
#pragma omp for nowait
    for (int h = 0; h < numHalfEdges; ++h) {
      edgeRefinement(mesh, h);
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
        boundaryEdgePoint(mesh, h);
      } else if (twin(h) > h) {
        interiorEdgePoint(mesh, h);
        interiorEdgePoint(mesh, twin(h));
      }
    }

#pragma omp for
    for (int h = 0; h < numHalfEdges; ++h) {
      // val = -1 if boundary vertex
      float n = valence(h);
      if (n > 0) {
        interiorVertexPoint(mesh, h, n);
      } else if (twin(h) < 0) {
        boundaryVertexPoint(mesh, h);
      }
    }
  }
}
