#ifndef MESHINITIALIZER_H
#define MESHINITIALIZER_H

#include <QDebug>
#include <QVector>

#include "mesh.h"
#include "objfile.h"

class MeshInitializer {
 public:
  MeshInitializer(OBJFile* loadedOBJFile);
  Mesh* constructHalfEdgeMesh();

 private:
  OBJFile* loadedObj;

  QList<QPair<uint, uint>> edgeList;

  uint numHalfEdges;
  QVector<uint> twins;
  QVector<uint> nexts;
  QVector<uint> prevs;
  QVector<uint> verts;
  QVector<uint> edges;
  QVector<uint> faces;

  void addHalfEdge(uint h, uint faceIdx, QVector<uint> faceIndices, uint i);
  void setEdgeAndTwins(uint h0, uint vertIdx2);
};

#endif  // MESHINITIALIZER_H
