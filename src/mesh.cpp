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
}

/**
 * @brief Mesh::~Mesh Destructor
 */
Mesh::~Mesh() {}

void Mesh::extractAttributes() {
  polyIndices.clear();
  // TODO: reserve at some point

  // half edges that belong to a face are stored contiguously
  for (int k = 0; k < verts.size(); ++k) {
    if (k > 0 && faces[k] != faces[k - 1]) {
      polyIndices.append(INT_MAX);
    }
    polyIndices.append(verts[k]);
  }
  polyIndices.append(INT_MAX);

  // calculate vector of face normals
  QVector<QVector3D> faceNormals;
  int faceIdx = -1;
  for (uint h = 0; h < verts.size(); ++h) {
    if (faces[h] != faceIdx) {
      faceIdx = faces[h];
      faceNormals.append({0, 0, 0});
    }
    QVector3D pPrev = vertexCoords[verts[prevs[h]]];
    QVector3D pCur = vertexCoords[verts[h]];
    QVector3D pNext = vertexCoords[verts[nexts[h]]];

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
  for (uint h = 0; h < verts.size(); ++h) {
    QVector3D pPrev = vertexCoords[verts[prevs[h]]];
    QVector3D pCur = vertexCoords[verts[h]];
    QVector3D pNext = vertexCoords[verts[nexts[h]]];

    QVector3D edgeA = (pPrev - pCur).normalized();
    QVector3D edgeB = (pNext - pCur).normalized();
    // calculate angle between edges A and B
    float faceAngle = acos(fmax(-1.0f, QVector3D::dotProduct(edgeA, edgeB)));

    vertexNormals[verts[h]] += faceAngle * faceNormals[faces[h]];
  }
}
