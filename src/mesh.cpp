#include "mesh.h"

#include "math.h"

/**
 * @brief Mesh::Mesh Creates an empty mesh
 */
Mesh::Mesh() {}

Mesh::Mesh(QVector<QVector3D> vertCoords, QVector<int> twins,
           QVector<int> nexts, QVector<int> prevs, QVector<int> verts,
           QVector<int> edges, QVector<int> faces, int nEdges) {
  vertexCoords = vertCoords;
  this->twins = twins;
  this->nexts = nexts;
  this->prevs = prevs;
  this->verts = verts;
  this->edges = edges;
  this->faces = faces;

  numHalfEdges = verts.size();
  numEdges = nEdges;
  numFaces = face(faces.size() - 1) + 1;
  numVerts = vertCoords.size();
}

/**
 * @brief Mesh::~Mesh Destructor
 */
Mesh::~Mesh() {}

int Mesh::twin(int h) { return twins.at(h); }

int Mesh::vert(int h) { return verts.at(h); }

int Mesh::edge(int h) { return edges.at(h); }

int Mesh::next(int h) { return nexts.at(h); }

int Mesh::prev(int h) { return prevs.at(h); }

int Mesh::face(int h) { return faces.at(h); }

// returns -1 if it is a boundary vertex
int Mesh::valence(int h) {
  int ht = twin(h);
  if (ht < 0) {
    return -1;
  }
  int n = 1;
  int hp = next(ht);
  while (hp != h) {
    if (hp < 0) {
      return -1;
    }
    ht = twin(hp);
    if (ht < 0) {
      return -1;
    }
    hp = next(ht);
    n++;
  }
  return n;
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
  for (int h = 0; h < numHalfEdges; ++h) {
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
  for (int h = 0; h < numHalfEdges; ++h) {
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
