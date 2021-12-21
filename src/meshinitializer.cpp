#include "meshinitializer.h"

#include <QSet>

#include "quadmesh.h"

MeshInitializer::MeshInitializer(OBJFile* loadedOBJFile) {
  loadedObj = loadedOBJFile;
}

QPair<uint, uint> createUndirectedEdge(uint h1, uint h2) {
  // to ensure that edges are consistent, always put the lower index first
  if (h1 > h2) {
    return QPair<uint, uint>(h2, h1);
  }
  return QPair<uint, uint>(h1, h2);
}

Mesh* MeshInitializer::constructHalfEdgeMesh() {
  // half edge index
  uint h = 0;
  // loop over every face
  for (uint faceIdx = 0; faceIdx < loadedObj->faceCoordInd.size(); ++faceIdx) {
    QVector<uint> faceIndices = loadedObj->faceCoordInd[faceIdx];
    // each face will end up with a number of half edges equal to its number of
    // faces
    for (uint i = 0; i < faceIndices.size(); ++i) {
      addHalfEdge(h, faceIdx, faceIndices, i);
      h++;
    }
  }
  numHalfEdges = h;

  if (loadedObj->isQuad) {
    qDebug() << "Creating a quad mesh";
    return new QuadMesh(loadedObj->vertexCoords, twins, nexts, prevs, verts,
                        edges, faces);
  }
  return new Mesh(loadedObj->vertexCoords, twins, nexts, prevs, verts, edges,
                  faces);
}

void MeshInitializer::addHalfEdge(uint h, uint faceIdx,
                                  QVector<uint> faceIndices, uint i) {
  uint faceValency = faceIndices.size();
  // vert
  uint vertIdx = faceIndices[i];
  verts.append(vertIdx);

  // prev and next
  uint prev = h - 1;
  uint next = h + 1;
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
  uint nextVertIdx = faceIndices[(i + 1) % faceValency];
  setEdgeAndTwins(h, nextVertIdx);

  // face
  faces.append(faceIdx);
}

void MeshInitializer::setEdgeAndTwins(uint h, uint vertIdx2) {
  uint vertIdx1 = verts[h];
  QPair<uint, uint> currentEdge = createUndirectedEdge(vertIdx1, vertIdx2);

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
    uint twin = edges.indexOf(edgeIdx);
    twins[h] = twin;
    twins[twin] = h;
  }
}
