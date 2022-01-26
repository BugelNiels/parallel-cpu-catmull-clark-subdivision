#ifndef QUADMESH_H
#define QUADMESH_H

#include <QDebug>

#include "mesh.h"

/**
 * @brief The QuadMesh class extends a mesh and contains optimized method
 * specific to quad meshes
 */
class QuadMesh : public Mesh {
 public:
  using Mesh::Mesh;
  QuadMesh(int maxHalfEdges, int maxVertices);
  void resetMesh();
  void subdivideCatmullClark(QuadMesh& mesh) override;

 protected:
  int next(int h) override;
  int prev(int h) override;
  int face(int h) override;
};

#endif  // QUADMESH_H
