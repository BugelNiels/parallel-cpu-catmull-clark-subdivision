#include "mesh.h"
#include "omp.h"
#include "quadmesh.h"

void Mesh::subdivideCatmullClark(QuadMesh& mesh) {
  recalculateSizes(mesh);
  resizeBuffers(mesh);

// Half Edge Refinement Rules
#pragma omp parallel for
  for (int h = 0; h < numHalfEdges; ++h) {
    edgeRefinement(mesh, h, numVerts, numFaces, numEdges);
  }

  insertFacePoints(mesh);
  insertEdgePoints(mesh);
  insertVertexPoints(mesh);
}

void Mesh::insertFacePoints(QuadMesh& mesh) {
#pragma omp parallel for
  for (int h = 0; h < numHalfEdges; ++h) {
    facePoint(mesh, h, numVerts);
  }
}

void Mesh::insertEdgePoints(QuadMesh& mesh) {
#pragma omp parallel for
  for (int h = 0; h < numHalfEdges; ++h) {
    if (twin(h) < 0) {
      boundaryEdgePoint(mesh, h, numVerts, numFaces);
    } else {
      smoothEdgePoint(mesh, h, numVerts, numFaces);
    }
  }
}

void Mesh::insertVertexPoints(QuadMesh& mesh) {
#pragma omp parallel for
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

int Mesh::cycleLength(int h) {
  int n = 1;
  int hp = next(h);
  while (hp != h) {
    hp = next(hp);
    n++;
  }
  return n;
}

void Mesh::recalculateSizes(QuadMesh& mesh) {
  mesh.numEdges = 2 * numEdges + numHalfEdges;
  mesh.numFaces = numHalfEdges;
  mesh.numHalfEdges = numHalfEdges * 4;
  mesh.numVerts = numVerts + numFaces + numEdges;
}

void Mesh::resizeBuffers(QuadMesh& mesh) {
  // TODO: might see improvements by doing this in parallel
  mesh.twins.resize(mesh.numHalfEdges);
  mesh.edges.resize(mesh.numHalfEdges);
  mesh.verts.resize(mesh.numHalfEdges);
  mesh.vertexCoords.resize(mesh.numVerts);
}

void Mesh::edgeRefinement(QuadMesh& mesh, int h, int vd, int fd, int ed) {
  int hp = prev(h);

  // For boundaries
  mesh.twins[4 * h] = 4 * next(twin(h)) + 3;
  mesh.twins[4 * h + 1] = 4 * next(h) + 2;
  mesh.twins[4 * h + 2] = 4 * prev(h) + 1;
  mesh.twins[4 * h + 3] = 4 * twin(hp);

  mesh.verts[4 * h] = vert(h);
  mesh.verts[4 * h + 1] = vd + fd + edge(h);
  mesh.verts[4 * h + 2] = vd + face(h);
  mesh.verts[4 * h + 3] = vd + fd + edge(hp);

  mesh.edges[4 * h] = h > twin(h) ? 2 * edge(h) : 2 * edge(h) + 1;
  mesh.edges[4 * h + 1] = 2 * ed + h;
  mesh.edges[4 * h + 2] = 2 * ed + hp;
  mesh.edges[4 * h + 3] = hp > twin(hp) ? 2 * edge(hp) + 1 : 2 * edge(hp);
}

void Mesh::facePoint(QuadMesh& mesh, int h, int vd) {
  float m = cycleLength(h);
  int v = vert(h);
  int i = vd + face(h);
  QVector3D c = vertexCoords[v] / m;
#pragma omp critical
  mesh.vertexCoords[i] += c;
}

inline void atomicAdd(QVector3D& vecA, const QVector3D& vecB) {
  for (int k = 0; k < 3; ++k) {
    float& a = vecA[k];
    const float b = vecB[k];
#pragma omp atomic
    a += b;
  }
}

void Mesh::smoothEdgePoint(QuadMesh& mesh, int h, int vd, int fd) {
  int v = vert(h);
  int i = vd + face(h);
  int j = vd + fd + edge(h);
  QVector3D c = (vertexCoords[v] + mesh.vertexCoords[i]) / 4.0f;
  atomicAdd(mesh.vertexCoords[j], c);
}

void Mesh::boundaryEdgePoint(QuadMesh& mesh, int h, int vd, int fd) {
  int v = vert(h);
  int vnext = vert(next(h));
  int j = vd + fd + edge(h);
  mesh.vertexCoords[j] = (vertexCoords[v] + vertexCoords[vnext]) / 2.0f;
}

void Mesh::smoothVertexPoint(QuadMesh& mesh, int h, int vd, int fd, float n) {
  int v = vert(h);
  int i = vd + face(h);
  int j = vd + fd + edge(h);
  QVector3D c = (4 * mesh.vertexCoords[j] - mesh.vertexCoords[i] +
                 (n - 3) * vertexCoords[v]) /
                (n * n);
  atomicAdd(mesh.vertexCoords[v], c);
}

void Mesh::boundaryVertexPoint(QuadMesh& mesh, int h) {
  int v = vert(h);
  mesh.vertexCoords[v] = vertexCoords[v];
}
