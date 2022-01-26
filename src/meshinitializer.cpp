#include "meshinitializer.h"

#include <QSet>

#include "quadmesh.h"

/**
 * @brief MeshInitializer::MeshInitializer Initialises a pointerless half-edge
 * mesh from an OBJFile
 * @param loadedOBJFile OBJFile to create a mesh from
 */
MeshInitializer::MeshInitializer(OBJFile* loadedOBJFile) {
  loadedObj = loadedOBJFile;
}

/**
 * @brief createUndirectedEdge Creates an undirected edge from two vertex
 * indices.
 * @param v1 First vertex index
 * @param v2 Second vertex index
 * @return A pair of v1-v2. Two indices always produce the same pair, regardless
 * of their ordering.
 */
QPair<int, int> createUndirectedEdge(int v1, int v2) {
  // to ensure that edges are consistent, always put the lower index first
  if (v1 > v2) {
    return QPair<int, int>(v2, v1);
  }
  return QPair<int, int>(v1, v2);
}

/**
 * @brief MeshInitializer::constructHalfEdgeMesh Constructs a half-edge mesh
 * @return A pointerless half-edge mesh.
 */
Mesh* MeshInitializer::constructHalfEdgeMesh() {
  int h = 0;
  // loop over every face
  for (int faceIdx = 0; faceIdx < loadedObj->faceCoordInd.size(); ++faceIdx) {
    QVector<int> faceIndices = loadedObj->faceCoordInd[faceIdx];
    // each face ends up with a number of half edges equal to its number of
    // vertices
    for (int i = 0; i < faceIndices.size(); ++i) {
      addHalfEdge(h, faceIdx, faceIndices, i);
      h++;
    }
  }
  numHalfEdges = h;

  if (loadedObj->isQuad) {
    qDebug() << "Creating a quad mesh";
    return new QuadMesh(loadedObj->vertexCoords, twins, nexts, prevs, verts,
                        edges, faces, edgeList.size());
  }
  return new Mesh(loadedObj->vertexCoords, twins, nexts, prevs, verts, edges,
                  faces, edgeList.size());
}

/**
 * @brief MeshInitializer::addHalfEdge Adds half-edge properties to the mesh
 * @param h The half-edge index
 * @param faceIdx Index of the face the half-edge belongs to
 * @param vertIndices The vertex indices of this face
 * @param i Index in vertIndices this half-edge originates from
 */
void MeshInitializer::addHalfEdge(int h, int faceIdx, QVector<int> vertIndices,
                                  int i) {
  int faceValency = vertIndices.size();
  int vertIdx = vertIndices[i];
  verts.append(vertIdx);

  // prev and next
  int prev = h - 1;
  int next = h + 1;
  if (i == 0) {
    // prev = h - 1 + faceValency
    prev += faceValency;
  } else if (i == faceValency - 1) {
    // next = h + 1 - faceValency
    next -= faceValency;
  }
  prevs.append(prev);
  nexts.append(next);

  // twin
  twins.append(-1);
  int nextVertIdx = vertIndices[(i + 1) % faceValency];
  setEdgeAndTwins(h, nextVertIdx);

  // face
  faces.append(faceIdx);
}

/**
 * @brief MeshInitializer::setEdgeAndTwins Sets the edge and twin properties of
 * the half-edge
 * @param h Half-edge index
 * @param vertIdx2 Index of the vertex h points to
 */
void MeshInitializer::setEdgeAndTwins(int h, int vertIdx2) {
  int vertIdx1 = verts[h];
  QPair<int, int> currentEdge = createUndirectedEdge(vertIdx1, vertIdx2);

  int edgeIdx = edgeList.indexOf(currentEdge);
  // edge does not exist yet
  if (edgeIdx == -1) {
    // same as doing this after appending and adding -1
    edges.append(edgeList.size());
    edgeList.append(currentEdge);
  } else {
    edges.append(edgeIdx);
    // edge already existed, meaning there is a twin somewhere earlier in the
    // list of edges
    int twin = edges.indexOf(edgeIdx);
    twins[h] = twin;
    twins[twin] = h;
  }
}
