#ifndef MESH_H
#define MESH_H

#include <QDebug>
#include <QVector>

#include "objfile.h"

class QuadMesh;

/**
 * @brief The Mesh class contains all the information for a mesh
 */
class Mesh {
 public:
  Mesh();
  // TODO: refactor
  Mesh(QVector<QVector3D> vertCoords, QVector<int> twins, QVector<int> nexts,
       QVector<int> prevs, QVector<int> verts, QVector<int> edges,
       QVector<int> faces);
  virtual ~Mesh();

  inline QVector<QVector3D>& getVertexCoords() { return vertexCoords; }
  inline QVector<QVector3D>& getVertexNorms() { return vertexNormals; }
  inline QVector<int>& getPolyIndices() { return polyIndices; }

  void extractAttributes();

  void subdivideCatmullClark(QuadMesh& mesh);

 protected:
  QVector<int> twins;
  QVector<int> nexts;
  QVector<int> prevs;
  QVector<int> verts;
  QVector<int> edges;
  QVector<int> faces;

  int numHalfEdges;

  QVector<QVector3D> vertexCoords;
  QVector<QVector3D> vertexNormals;
  QVector<int> polyIndices;

  virtual int next(int h);
  virtual int prev(int h);
  virtual int face(int h);
  int twin(int h);
  int vert(int h);
  int edge(int h);

  virtual int cycleLength(int h);
  int valence(int h);

  int getNumberOfEdges();
  virtual int getNumberOfFaces();
};

#endif  // MESH_H
