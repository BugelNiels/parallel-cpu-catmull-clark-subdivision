#ifndef SUBDIVIDER_H
#define SUBDIVIDER_H

#include "mesh.h"
#include "quadmesh.h"

class Subdivider {
 public:
  Subdivider(Mesh* mesh);
  double subdivide(int subdivisionLevel);

  inline Mesh* getBaseMesh() { return baseMesh; }
  inline QuadMesh* getCurrentMesh() { return currentMesh; }

 private:
  void singleSubdivisionStep(int k);
  Mesh* baseMesh;
  QuadMesh* currentMesh = nullptr;
};

#endif  // SUBDIVIDER_H
