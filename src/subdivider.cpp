#include "subdivider.h"

#include <QElapsedTimer>
#include <iostream>

#include "math.h"

/**
 * @brief Subdivider::Subdivider Creates a subdivider with a given base mesh
 * @param mesh The base mesh this subdivider is responsible for subdividing
 */
Subdivider::Subdivider(Mesh* mesh) { baseMesh = mesh; }

/**
 * @brief meshSwap Swaps two pointers to mesh pointers
 * @param prevMeshPtr First mesh pointer
 * @param newMeshPtr Second mesh pointer
 */
void meshSwap(QuadMesh** prevMeshPtr, QuadMesh** newMeshPtr) {
  QuadMesh* temp = *prevMeshPtr;
  *prevMeshPtr = *newMeshPtr;
  *newMeshPtr = temp;
}

/**
 * @brief Subdivider::getNumberOfVertsAtLevel Calculates the number of vertices
 * a mesh would have after a number of subdivision steps.
 * @param subdivisionLevel Number of subdivision steps
 * @return Number of vertices after "subdivisionLevel" subdivision steps
 */
int Subdivider::getNumberOfVertsAtLevel(int subdivisionLevel) {
  int v1 = baseMesh->getNumVerts() + baseMesh->getNumFaces() +
           baseMesh->getNumEdges();
  int e1 = 2 * baseMesh->getNumEdges() + baseMesh->getNumHalfEdges();
  int f1 = baseMesh->getNumHalfEdges();
  int finalNumVerts = v1;
  if (subdivisionLevel > 1) {
    finalNumVerts += pow(2, subdivisionLevel - 1) *
                     (e1 + (pow(2, subdivisionLevel) - 1) * f1);
  }
  return finalNumVerts;
}

/**
 * @brief Subdivider::printSummary Prints a summary of the subdivision process
 * @param subdivisionLevel Number of subdivision levels applied
 * @param milsecs Time the subdivision process took in milliseconds
 */
void Subdivider::printSummary(int subdivisionLevel, double milsecs) {
  std::cout << "\n---\nTotal time elapsed for " << subdivisionLevel << " : "
            << milsecs << " milliseconds\n\n";
  std::cout << "Faces : " << currentMesh->getNumFaces() << "\n";
  std::cout << "Half Edges : " << currentMesh->getNumHalfEdges() << "\n";
  std::cout << "Vertices : " << currentMesh->getNumVerts() << "\n";
  std::cout << "Edges : " << currentMesh->getNumEdges() << "\n\n";
}

/**
 * @brief Subdivider::subdivide Performs Catmull-Clark subdivision on the base
 * mesh of this subdivider for "subdivisionLevel" levels.
 * @param subdivisionLevel Number of subdivision steps to apply
 * @return Number of milliseconds the execution took.
 */
double Subdivider::subdivide(int subdivisionLevel) {
  if (subdivisionLevel < 1) {
    return 0.0;
  }
  if (currentMesh != nullptr) {
    // prevent memory leak
    free(currentMesh);
    currentMesh = nullptr;
  }
  int finalNumHalfEdges =
      pow(4, subdivisionLevel) * baseMesh->getNumHalfEdges();
  int finalNumVerts = getNumberOfVertsAtLevel(subdivisionLevel);

  // pre-allocate
  QuadMesh* in = new QuadMesh(finalNumHalfEdges, finalNumVerts);
  QuadMesh* out = new QuadMesh(finalNumHalfEdges, finalNumVerts);

  // Only time the actual execution.
  QElapsedTimer timer;
  timer.start();

  baseMesh->subdivideCatmullClark(*in);
  for (unsigned short k = 1; k < subdivisionLevel; k++) {
    out->resetMesh();
    in->subdivideCatmullClark(*out);
    meshSwap(&in, &out);
  }
  /* Display info to user */
  long long time = timer.nsecsElapsed();
  double milsecs = time / 1000000.0;

  currentMesh = in;
  printSummary(subdivisionLevel, milsecs);
  free(out);
  return milsecs;
}
