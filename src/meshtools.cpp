#include "mesh.h"

uint Mesh::valence(uint h) {
  uint n = 1;
  uint hp = next(twin(h));
  while (hp != h) {
    hp = next(twin(hp));
    n++;
  }
  return n;
}

uint Mesh::cycleLength(uint h) {
  uint n = 1;
  uint hp = next(h);
  while (hp != h) {
    hp = next(hp);
    n++;
  }
  return n;
}

void Mesh::subdivideCatmullClark(Mesh& mesh) {
  // Allocate Buffers
  uint newSize = numHalfEdges * 4;
  mesh.numHalfEdges = newSize;
  mesh.twins.resize(newSize);
  mesh.edges.resize(newSize);
  mesh.verts.resize(newSize);
  mesh.vertexCoords.resize(newSize);

  uint vd = verts.size();
  uint fd = isQuadMesh ? numHalfEdges / 4 : face(faces.size() - 1);
  // TODO does not work for boundaries
  uint ed = numHalfEdges / 2;

  // Half Edge Refinement Rules
  for (uint h = 0; h < numHalfEdges; ++h) {
    uint hp = prev(h);

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
  for (uint h = 0; h < numHalfEdges; ++h) {
    float m = cycleLength(h);
    uint v = vert(h);
    uint i = vd + face(h);
    mesh.vertexCoords[i] += vertexCoords[v] / m;
  }

  // Edge Points
  for (uint h = 0; h < numHalfEdges; ++h) {
    uint v = vert(h);
    uint i = vd + face(h);
    uint j = vd + fd + edge(h);
    mesh.vertexCoords[j] += (vertexCoords[v] + mesh.vertexCoords[i]) / 4.0f;
  }

  // Vertex Points
  for (uint h = 0; h < numHalfEdges; ++h) {
    float n = valence(h);
    uint v = vert(h);
    uint i = vd + face(h);
    uint j = vd + fd + edge(h);
    mesh.vertexCoords[v] += (4 * mesh.vertexCoords[j] - mesh.vertexCoords[i] +
                             (n - 3) * vertexCoords[v]) /
                            (n * n);
  }

  mesh.isQuadMesh = true;
}
