#include "mesh.h"
#include "omp.h"
#include "quadmesh.h"
#include "util.h"

/**
 * @brief Mesh::subdivideCatmullClark Performs a single Catmull-Clark
 * subdivision step and saves the result in the provided argument mesh.
 * @param mesh The mesh to store the result of this subdivision step in.
 */
void Mesh::subdivideCatmullClark(QuadMesh& mesh) {
  recalculateSizes(mesh);

#pragma omp parallel
  {
// Half Edge Refinement Rules
#pragma omp for nowait
    for (int h = 0; h < numHalfEdges; ++h) {
      edgeRefinement(mesh, h);
    }
// Face points
#pragma omp for
    for (int h = 0; h < numHalfEdges; ++h) {
      facePoint(mesh, h);
    }

// Edge points
#pragma omp for
    for (int h = 0; h < numHalfEdges; ++h) {
      if (twin(h) < 0) {
        boundaryEdgePoint(mesh, h);
      } else if (twin(h) > h) {
        // By doing both of these here, we do not need atomic adds.
        interiorEdgePoint(mesh, h);
        interiorEdgePoint(mesh, twin(h));
      }
    }

// Vertex points
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

/**
 * @brief Mesh::cycleLength Determines the number of edges a face has.
 * @param h A half-edge index in the face of which to determine the cycle length
 * @return Cycle length of face(h)
 */
int Mesh::cycleLength(int h) {
  int n = 1;
  int hp = next(h);
  while (hp != h) {
    hp = next(hp);
    n++;
  }
  return n;
}

/**
 * @brief Mesh::recalculateSizes Calculate the sizes at step d+1 based on the
 * mesh at d
 * @param mesh The for which to recalculate the sizes
 */
void Mesh::recalculateSizes(QuadMesh& mesh) {
  mesh.numEdges = 2 * numEdges + numHalfEdges;
  mesh.numFaces = numHalfEdges;
  mesh.numHalfEdges = numHalfEdges * 4;
  mesh.numVerts = numVerts + numFaces + numEdges;
}

/**
 * @brief Mesh::edgeRefinement Topology refinement of a single half-edge.
 * Generates 4 new half-edges.
 * @param mesh Mesh in which to save the topology changes
 * @param h Half-edge index
 */
void Mesh::edgeRefinement(QuadMesh& mesh, int h) {
  int hp = prev(h);

  int ht = twin(h);
  mesh.twins[4 * h] = ht < 0 ? -1 : 4 * next(ht) + 3;
  mesh.twins[4 * h + 1] = 4 * next(h) + 2;
  mesh.twins[4 * h + 2] = 4 * hp + 1;
  mesh.twins[4 * h + 3] = 4 * twin(hp);

  mesh.verts[4 * h] = vert(h);
  mesh.verts[4 * h + 1] = numVerts + numFaces + edge(h);
  mesh.verts[4 * h + 2] = numVerts + face(h);
  mesh.verts[4 * h + 3] = numVerts + numFaces + edge(hp);

  mesh.edges[4 * h] = h > ht ? 2 * edge(h) : 2 * edge(h) + 1;
  mesh.edges[4 * h + 1] = 2 * numEdges + h;
  mesh.edges[4 * h + 2] = 2 * numEdges + hp;
  mesh.edges[4 * h + 3] = hp > twin(hp) ? 2 * edge(hp) + 1 : 2 * edge(hp);
}

/**
 * @brief Mesh::facePoint Calculates the contribution of the half-edge to its
 * face point.
 * @param mesh Mesh the face point is in
 * @param h Half-edge index
 */
void Mesh::facePoint(QuadMesh& mesh, int h) {
  float m = cycleLength(h);
  int v = vert(h);
  int i = numVerts + face(h);
  QVector3D c = vertexCoords.at(v) / m;
  atomicAdd(mesh.vertexCoords[i], c);
}

/**
 * @brief Mesh::interiorEdgePoint Calculates the contribution of this half-edge
 * to its edge point. Edge point should not lie on boundary
 * @param mesh Mesh the edge point is in
 * @param h Half-edge index
 */
void Mesh::interiorEdgePoint(QuadMesh& mesh, int h) {
  int v = vert(h);
  int i = numVerts + face(h);
  int j = numVerts + numFaces + edge(h);
  QVector3D c = (vertexCoords.at(v) + mesh.vertexCoords.at(i)) / 4.0f;
  mesh.vertexCoords[j] += c;
}

/**
 * @brief Mesh::boundaryEdgePoint Calculates the contribution of this half-edge
 * to its edge point. Edge point should lie on boundary
 * @param mesh Mesh the edge point is in
 * @param h Half-edge index
 */
void Mesh::boundaryEdgePoint(QuadMesh& mesh, int h) {
  int v = vert(h);
  int vnext = vert(next(h));
  int j = numVerts + numFaces + edge(h);
  mesh.vertexCoords[j] = (vertexCoords.at(v) + vertexCoords.at(vnext)) / 2.0f;
}

/**
 * @brief Mesh::interiorVertexPoint Calculates the contribution of this
 * half-edge to its vertex point. Vertex point should not lie on boundary
 * @param mesh Mesh the vertex point is in
 * @param h Half-edge index
 * @param n Valence of the interior vertex
 */
void Mesh::interiorVertexPoint(QuadMesh& mesh, int h, float n) {
  int v = vert(h);
  int i = numVerts + face(h);
  int j = numVerts + numFaces + edge(h);
  QVector3D c = (4 * mesh.vertexCoords.at(j) - mesh.vertexCoords.at(i) +
                 (n - 3) * vertexCoords.at(v)) /
                (n * n);
  atomicAdd(mesh.vertexCoords[v], c);
}

/**
 * @brief Mesh::boundaryVertexPoint Calculates the contribution of this
 * half-edge to its vertex point. Vertex point should lie boundary
 * @param mesh Mesh the vertex point is in
 * @param h Half-edge index
 */
void Mesh::boundaryVertexPoint(QuadMesh& mesh, int h) {
  int v = vert(h);
  int j = numVerts + numFaces + edge(h);
  QVector3D edgePoint = mesh.vertexCoords.at(j);
  QVector3D c = (edgePoint + vertexCoords.at(v)) / 4.0f;
  atomicAdd(mesh.vertexCoords[v], c);

  int vnext = vert(next(h));
  QVector3D c2 = (edgePoint + vertexCoords.at(vnext)) / 4.0f;
  atomicAdd(mesh.vertexCoords[vnext], c2);
}
