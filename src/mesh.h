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
  Mesh(QVector<QVector3D> vertCoords, QVector<uint> twins, QVector<uint> nexts,
       QVector<uint> prevs, QVector<uint> verts, QVector<uint> edges,
       QVector<uint> faces);
  virtual ~Mesh();

  inline QVector<QVector3D>& getVertexCoords() { return vertexCoords; }
  inline QVector<QVector3D>& getVertexNorms() { return vertexNormals; }
  inline QVector<uint>& getPolyIndices() { return polyIndices; }

  void extractAttributes();

  void subdivideCatmullClark(QuadMesh& mesh);

 protected:
  QVector<uint> twins;
  QVector<uint> nexts;
  QVector<uint> prevs;
  QVector<uint> verts;
  QVector<uint> edges;
  QVector<uint> faces;

  uint numHalfEdges;

  QVector<QVector3D> vertexCoords;
  QVector<QVector3D> vertexNormals;
  QVector<uint> polyIndices;

  virtual uint next(uint h);
  virtual uint prev(uint h);
  virtual uint face(uint h);
  uint twin(uint h);
  uint vert(uint h);
  uint edge(uint h);

  virtual uint cycleLength(uint h);
  uint valence(uint h);

  uint getNumberOfEdges();
  virtual uint getNumberOfFaces();
};

#endif  // MESH_H
