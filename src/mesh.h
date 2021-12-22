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
       QVector<int> faces, int nEdges);
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
  int numEdges;
  int numFaces;
  int numVerts;

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

  void recalculateSizes(QuadMesh& mesh);
  void resizeBuffers(QuadMesh& mesh);

  virtual void insertFacePoints(QuadMesh& mesh);
  virtual void insertEdgePoints(QuadMesh& mesh);
  void insertVertexPoints(QuadMesh& mesh);

  void edgeRefinement(QuadMesh& mesh, int h, int vd, int fd, int ed);
  virtual void facePoint(QuadMesh& mesh, int h, int vd);
  void smoothEdgePoint(QuadMesh& mesh, int h, int vd, int fd);
  void boundaryEdgePoint(QuadMesh& mesh, int h, int vd, int fd);
  void smoothVertexPoint(QuadMesh& mesh, int h, int vd, int fd, float n);
  void boundaryVertexPoint(QuadMesh& mesh, int h);
};

#endif  // MESH_H
