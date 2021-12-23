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

  // Getters
  inline QVector<QVector3D>& getVertexCoords() { return vertexCoords; }
  inline QVector<QVector3D>& getVertexNorms() { return vertexNormals; }
  inline QVector<int>& getPolyIndices() { return polyIndices; }

  inline QVector<int> getTwins() { return twins; }
  inline QVector<int> getNexts() { return nexts; }
  inline QVector<int> getPrevs() { return prevs; }
  inline QVector<int> getVerts() { return verts; }
  inline QVector<int> getEdges() { return edges; }
  inline QVector<int> getFaces() { return faces; }

  inline int getNumHalfEdges() { return numHalfEdges; }
  inline int getNumEdges() { return numEdges; }
  inline int getNumFaces() { return numFaces; }
  inline int getNumVerts() { return numVerts; }

  void extractAttributes();

  virtual void subdivideCatmullClark(QuadMesh& mesh);

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

  int valence(int h);

  void recalculateSizes(QuadMesh& mesh);
  void resizeBuffers();

  void edgeRefinement(QuadMesh& mesh, int h, int vd, int fd, int ed);
  virtual void facePoint(QuadMesh& mesh, int h, int vd);
  void smoothEdgePoint(QuadMesh& mesh, int h, int vd, int fd);
  void boundaryEdgePoint(QuadMesh& mesh, int h, int vd, int fd);
  void smoothVertexPoint(QuadMesh& mesh, int h, int vd, int fd, float n);
  void boundaryVertexPoint(QuadMesh& mesh, int h);

 private:
  int cycleLength(int h);
};

#endif  // MESH_H
