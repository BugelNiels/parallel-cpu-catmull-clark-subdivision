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
  Mesh(OBJFile* loadedOBJFile);
  ~Mesh();

  inline QVector<QVector3D>& getVertexCoords() { return vertexCoords; }
  inline QVector<QVector3D>& getVertexNorms() { return vertexNormals; }
  inline QVector<unsigned int>& getPolyIndices() { return polyIndices; }
  inline QVector<unsigned int>& getQuadIndices() { return quadIndices; }

  void extractAttributes();
  void subdivideCatmullClark(Mesh& mesh);

 private:
  QVector<QVector3D> vertexCoords;
  QVector<QVector3D> vertexNormals;
  QVector<unsigned int> polyIndices;

  // for quad tessellation
  QVector<unsigned int> quadIndices;
};

#endif  // MESH_H
