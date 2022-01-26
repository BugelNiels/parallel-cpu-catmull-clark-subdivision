#ifndef MESHINITIALIZER_H
#define MESHINITIALIZER_H

#include <QDebug>
#include <QVector>

#include "mesh.h"
#include "objfile.h"

/**
 * @brief The MeshInitializer class initializes meshes from OBJFiles
 */
class MeshInitializer {
 public:
  MeshInitializer(OBJFile* loadedOBJFile);
  Mesh* constructHalfEdgeMesh();

 private:
  OBJFile* loadedObj;

  QList<QPair<int, int>> edgeList;

  int numHalfEdges;
  QVector<int> twins;
  QVector<int> nexts;
  QVector<int> prevs;
  QVector<int> verts;
  QVector<int> edges;
  QVector<int> faces;

  void addHalfEdge(int h, int faceIdx, QVector<int> faceIndices, int i);
  void setEdgeAndTwins(int h0, int vertIdx2);
};

#endif  // MESHINITIALIZER_H
