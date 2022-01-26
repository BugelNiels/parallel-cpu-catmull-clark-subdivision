#include "mesh.h"

#include "math.h"

/**
 * @brief Mesh::Mesh Creates an empty mesh
 */
Mesh::Mesh() {}

/**
 * @brief Mesh::Mesh Creates a pointerless half-edge mesh with the provided
 * information
 * @param vertCoords Vertex coordinates
 * @param twins Array of twin half-edge indices. Size: number of half edges
 * @param nexts Array of next half-edge indices. Size: number of half edges
 * @param prevs Array of previous half-edge indices. Size: number of half edges
 * @param verts Array of vertex indices. Size: number of half edges
 * @param edges Array of edge indices. Size: number of half edges
 * @param faces Array of face indices. Size: number of half edges
 * @param nEdges Number of edges in the mesh
 */
Mesh::Mesh(QVector<QVector3D>& vertCoords, QVector<int>& twins,
           QVector<int>& nexts, QVector<int>& prevs, QVector<int>& verts,
           QVector<int>& edges, QVector<int>& faces, int nEdges) {
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

/**
 * @brief Mesh::twin Retrieves the index of the twin of the provided half-edge
 * index for a non-quad mesh.
 * @param h Half-edge index to retrieve the twin index of
 * @return Twin half-edge index of h
 */
int Mesh::twin(int h) { return twins.at(h); }

/**
 * @brief Mesh::vert Retrieves the index of the vertex that the provided
 * half-edge index originates from for a non-quad mesh.
 * @param h Half-edge index to retrieve the vertex index of
 * @return Vertex index of h
 */
int Mesh::vert(int h) { return verts.at(h); }

/**
 * @brief Mesh::edge Retrieves the index of the edge that the provided half-edge
 * index belongs to for a non-quad mesh.
 * @param h Half-edge index to find the edge index of
 * @return Edge index of h
 */
int Mesh::edge(int h) { return edges.at(h); }

/**
 * @brief Mesh::next Retrieves the index of the next half-edge for a non-quad
 * mesh.
 * @param h Half-edge index to find the next index of
 * @return Next half-edge index of h
 */
int Mesh::next(int h) { return nexts.at(h); }

/**
 * @brief Mesh::prev Retrieves the index of the previous half-edge for a
 * non-quad mesh.
 * @param h Half-edge index to find the previous index of
 * @return Previous half-edge index of h
 */
int Mesh::prev(int h) { return prevs.at(h); }

/**
 * @brief Mesh::face Retrieves the index of the face the provided half-edge
 * index belongs to in a non-quad mesh.
 * @param h Half-edge index to find the face index of
 * @return Face index of h
 */
int Mesh::face(int h) { return faces.at(h); }

/**
 * @brief Mesh::valence Determines the valence of VERT(h). Returns -1 if VERT(h)
 * is a boundary vertex.
 * @param h A half-edge that originates from VERT(h)
 * @return The valence of VERT(h). -1 if boundary vertex
 */
int Mesh::valence(int h) {
  int ht = twin(h);
  if (ht < 0) {
    return -1;
  }
  int n = 1;
  int hp = next(ht);
  while (hp != h) {
    ht = twin(hp);
    if (ht < 0) {
      return -1;
    }
    hp = next(ht);
    n++;
  }
  return n;
}

/**
 * @brief Mesh::extractAttributes Extracts attributes so that these can be used
 * for drawing.
 */
void Mesh::extractAttributes() {
  polyIndices.clear();
  int fd = numHalfEdges / 4;
  polyIndices.reserve(fd + fd / 4);
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
  faceNormals.reserve(numFaces);
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
