#ifndef SUBDIVIDER_H
#define SUBDIVIDER_H

#include "mesh.h"

class Subdivider {
 public:
  Subdivider(Mesh* mesh);
  double subdivide(int subdivisionLevel, int iterations);
  double subdivide(int subdivisionLevel);

  inline Mesh* getBaseMesh() { return baseMesh; }
  inline Mesh* getCurrentMesh() { return currentMesh; }

 private:
  void singleSubdivisionStep(int k);
  Mesh* baseMesh;
  Mesh* currentMesh;
};

#endif  // SUBDIVIDER_H
