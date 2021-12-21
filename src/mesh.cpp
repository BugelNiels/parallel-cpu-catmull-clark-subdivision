#include "mesh.h"

#include "math.h"

/**
 * @brief Mesh::Mesh Creates an empty mesh
 */
Mesh::Mesh() {}

Mesh::Mesh(QVector<QVector3D> vertCoords, QVector<uint> twins,
           QVector<uint> nexts, QVector<uint> prevs, QVector<uint> verts,
           QVector<uint> edges, QVector<uint> faces, bool isQuad) {
  vertexCoords = vertCoords;
  this->twins = twins;
  this->nexts = nexts;
  this->prevs = prevs;
  this->verts = verts;
  this->edges = edges;
  this->faces = faces;
  isQuadMesh = isQuad;
  numHalfEdges = verts.size();
}

/**
 * @brief Mesh::~Mesh Destructor
 */
Mesh::~Mesh() {}

uint Mesh::twin(uint h) { return twins[h]; }

uint Mesh::vert(uint h) { return verts[h]; }

uint Mesh::edge(uint h) { return edges[h]; }

uint Mesh::next(uint h) {
  if (isQuadMesh) {
    return h % 4 == 3 ? h - 3 : h + 1;
  }
  return nexts[h];
}

uint Mesh::prev(uint h) {
  if (isQuadMesh) {
    return h % 4 == 0 ? h + 3 : h - 1;
  }
  return prevs[h];
}

uint Mesh::face(uint h) {
  if (isQuadMesh) {
    return h / 4;
  }
  return faces[h];
}

void Mesh::extractAttributes() {
  polyIndices.clear();
  // TODO: reserve at some point

  // half edges that belong to a face are stored contiguously
  for (int h = 0; h < numHalfEdges; ++h) {
    if (h > 0 && face(h) != face(h - 1)) {
      polyIndices.append(INT_MAX);
    }
    polyIndices.append(vert(h));
  }
  polyIndices.append(INT_MAX);

  // calculate vector of face normals
  QVector<QVector3D> faceNormals;
  int faceIdx = -1;
  for (uint h = 0; h < numHalfEdges; ++h) {
    if (face(h) != faceIdx) {
      faceIdx = face(h);
      faceNormals.append({0, 0, 0});
    }
    QVector3D pPrev = vertexCoords[vert(prev(h))];
    QVector3D pCur = vertexCoords[vert(h)];
    QVector3D pNext = vertexCoords[vert(next(h))];

    QVector3D edgeA = (pPrev - pCur).normalized();
    QVector3D edgeB = (pNext - pCur).normalized();

    faceNormals[faceIdx] += QVector3D::crossProduct(edgeB, edgeA);
  }

  for (int f = 0; f < faceNormals.size(); f++) {
    faceNormals[f].normalize();
  }

  vertexNormals.clear();
  vertexNormals.reserve(verts.size());
  vertexNormals.fill({0, 0, 0}, verts.size());

  // normal computation
  for (uint h = 0; h < numHalfEdges; ++h) {
    QVector3D pPrev = vertexCoords[vert(prev(h))];
    QVector3D pCur = vertexCoords[vert(h)];
    QVector3D pNext = vertexCoords[vert(next(h))];

    QVector3D edgeA = (pPrev - pCur).normalized();
    QVector3D edgeB = (pNext - pCur).normalized();
    // calculate angle between edges A and B
    float faceAngle = acos(fmax(-1.0f, QVector3D::dotProduct(edgeA, edgeB)));

    vertexNormals[vert(h)] += faceAngle * faceNormals[face(h)];
  }
}
