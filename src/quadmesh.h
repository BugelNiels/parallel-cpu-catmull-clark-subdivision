#ifndef QUADMESH_H
#define QUADMESH_H

#include <QDebug>

#include "mesh.h"

class QuadMesh : public Mesh {
 public:
  using Mesh::Mesh;

 protected:
  uint next(uint h) override;
  uint prev(uint h) override;
  uint face(uint h) override;
  uint cycleLength(uint h) override;
  uint getNumberOfFaces() override;
};

#endif  // QUADMESH_H
