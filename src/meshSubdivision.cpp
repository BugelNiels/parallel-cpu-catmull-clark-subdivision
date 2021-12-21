#include "mesh.h"
#include "omp.h"
#include "quadmesh.h"

int Mesh::cycleLength(int h) {
  int n = 1;
  int hp = next(h);
  while (hp != h) {
    hp = next(hp);
    n++;
  }
  return n;
}

int Mesh::getNumberOfEdges() { return numHalfEdges / 2; }
int Mesh::getNumberOfFaces() { return face(faces.size() - 1); }

void Mesh::subdivideCatmullClark(QuadMesh& mesh) {
  qDebug() << "subdividing regular";
  // Allocate Buffers
  int newSize = numHalfEdges * 4;
  mesh.numHalfEdges = newSize;
  mesh.twins.resize(newSize);
  mesh.edges.resize(newSize);
  mesh.verts.resize(newSize);
  mesh.vertexCoords.resize(newSize);

  int vd = verts.size();
  int fd = getNumberOfFaces();
  // TODO does not work for boundaries
  int ed = getNumberOfEdges();

  // Half Edge Refinement Rules
#pragma omp parallel for
  for (int h = 0; h < numHalfEdges; ++h) {
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

// Face Points
#pragma omp parallel for
  for (int h = 0; h < numHalfEdges; ++h) {
    float m = cycleLength(h);
    int v = vert(h);
    int i = vd + face(h);
    QVector3D c = vertexCoords[v] / m;
#pragma omp critical
    mesh.vertexCoords[i] += c;
  }

// Edge Points
#pragma omp parallel for
  for (int h = 0; h < numHalfEdges; ++h) {
    int v = vert(h);
    int i = vd + face(h);
    int j = vd + fd + edge(h);
    QVector3D c = (vertexCoords[v] + mesh.vertexCoords[i]) / 4.0f;
#pragma omp critical
    mesh.vertexCoords[j] += c;
  }

// Vertex Points
#pragma omp parallel for
  for (int h = 0; h < numHalfEdges; ++h) {
    float n = valence(h);
    int v = vert(h);
    int i = vd + face(h);
    int j = vd + fd + edge(h);
    QVector3D c = (4 * mesh.vertexCoords[j] - mesh.vertexCoords[i] +
                   (n - 3) * vertexCoords[v]) /
                  (n * n);
#pragma omp critical
    mesh.vertexCoords[v] += c;
  }
}
