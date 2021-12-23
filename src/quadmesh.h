#ifndef QUADMESH_H
#define QUADMESH_H

#include <QDebug>

#include "mesh.h"

class QuadMesh : public Mesh {
 public:
  using Mesh::Mesh;
  void subdivideCatmullClark(QuadMesh& mesh) override;

 protected:
  int next(int h) override;
  int prev(int h) override;
  int face(int h) override;
};

#endif  // QUADMESH_H
