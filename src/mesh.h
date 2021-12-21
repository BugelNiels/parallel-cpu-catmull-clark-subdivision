#ifndef MESH_H
#define MESH_H

#include <QVector>

#include "objfile.h"

/**
 * @brief The Mesh class contains all the information for a mesh
 */
class Mesh {
 public:
  Mesh();
  // TODO: refactor
  Mesh(QVector<QVector3D> vertCoords, QVector<uint> twins, QVector<uint> nexts,
       QVector<uint> prevs, QVector<uint> verts, QVector<uint> edges,
       QVector<uint> faces, bool isQuad);
  ~Mesh();

  inline QVector<QVector3D>& getVertexCoords() { return vertexCoords; }
  inline QVector<QVector3D>& getVertexNorms() { return vertexNormals; }
  inline QVector<uint>& getPolyIndices() { return polyIndices; }

  void subdivideCatmullClark(Mesh& mesh);
  void extractAttributes();

 private:
  QVector<uint> twins;
  QVector<uint> nexts;
  QVector<uint> prevs;
  QVector<uint> verts;
  QVector<uint> edges;
  QVector<uint> faces;

  bool isQuadMesh;

  QVector<QVector3D> vertexCoords;
  QVector<QVector3D> vertexNormals;
  QVector<uint> polyIndices;
};

#endif  // MESH_H
