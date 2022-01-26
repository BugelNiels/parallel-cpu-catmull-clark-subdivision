#ifndef SUBDIVIDER_H
#define SUBDIVIDER_H

#include "mesh.h"
#include "quadmesh.h"

/**
 * @brief The Subdivider class is responsible for subdividing a given mesh a
 * number of times and providing a summary of this process.
 */
class Subdivider {
 public:
  Subdivider(Mesh* mesh);
  double subdivide(int subdivisionLevel);

  inline Mesh* getBaseMesh() { return baseMesh; }
  inline QuadMesh* getCurrentMesh() { return currentMesh; }

 private:
  Mesh* baseMesh;
  QuadMesh* currentMesh = nullptr;
  int getNumberOfVertsAtLevel(int subdivisionLevel);
  void printSummary(int subdivisionLevel, double milsecs);
};

#endif  // SUBDIVIDER_H
