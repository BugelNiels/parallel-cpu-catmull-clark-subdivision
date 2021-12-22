#include "meshinitializer.h"

#include <QSet>

#include "quadmesh.h"

MeshInitializer::MeshInitializer(OBJFile* loadedOBJFile) {
  loadedObj = loadedOBJFile;
}

QPair<int, int> createUndirectedEdge(int h1, int h2) {
  // to ensure that edges are consistent, always put the lower index first
  if (h1 > h2) {
    return QPair<int, int>(h2, h1);
  }
  return QPair<int, int>(h1, h2);
}

Mesh* MeshInitializer::constructHalfEdgeMesh() {
  // half edge index
  int h = 0;
  // loop over every face
  for (int faceIdx = 0; faceIdx < loadedObj->faceCoordInd.size(); ++faceIdx) {
    QVector<int> faceIndices = loadedObj->faceCoordInd[faceIdx];
    // each face will end up with a number of half edges equal to its number of
    // faces
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

void MeshInitializer::addHalfEdge(int h, int faceIdx, QVector<int> faceIndices,
                                  int i) {
  int faceValency = faceIndices.size();
  // vert
  int vertIdx = faceIndices[i];
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
  int nextVertIdx = faceIndices[(i + 1) % faceValency];
  setEdgeAndTwins(h, nextVertIdx);

  // face
  faces.append(faceIdx);
}

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
