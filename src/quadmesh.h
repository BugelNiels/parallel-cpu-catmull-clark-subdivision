#ifndef QUADMESH_H
#define QUADMESH_H

#include <QDebug>

#include "mesh.h"

class QuadMesh : public Mesh {
 public:
  using Mesh::Mesh;

 protected:
  int next(int h) override;
  int prev(int h) override;
  int face(int h) override;
  int cycleLength(int h) override;
  int getNumberOfFaces() override;
};

#endif  // QUADMESH_H
