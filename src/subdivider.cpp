#include "subdivider.h"

#include <QElapsedTimer>
#include <iostream>

#include "math.h"
#include "quadmesh.h"

Subdivider::Subdivider(Mesh* mesh) {
  baseMesh = mesh;
  currentMesh = baseMesh;
}

void meshSwap(QuadMesh** prevMeshPtr, QuadMesh** newMeshPtr) {
  QuadMesh* temp = *prevMeshPtr;
  *prevMeshPtr = *newMeshPtr;
  *newMeshPtr = temp;
}

double Subdivider::subdivide(int subdivisionLevel) {
  int finalNumHalfEdges =
      pow(4, subdivisionLevel) * baseMesh->getNumHalfEdges();
  int v1 = baseMesh->getNumVerts() + baseMesh->getNumFaces() +
           baseMesh->getNumEdges();
  int e1 = 2 * baseMesh->getNumEdges() + baseMesh->getNumHalfEdges();
  int f1 = baseMesh->getNumHalfEdges();
  int finalNumVerts = v1;
  if (subdivisionLevel > 1) {
    finalNumVerts += pow(2, subdivisionLevel - 1) *
                     (e1 + (pow(2, subdivisionLevel) - 1) * f1);
  }
  // pre-allocate
  QuadMesh* in = new QuadMesh(finalNumHalfEdges, finalNumVerts);
  QuadMesh* out = new QuadMesh(finalNumHalfEdges, finalNumVerts);

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

  std::cout << "\n---\nTotal time elapsed for " << subdivisionLevel << " : "
            << milsecs << " milliseconds\n\n";
  std::cout << "Faces : " << currentMesh->getNumFaces() << "\n";
  std::cout << "Half Edges : " << currentMesh->getNumHalfEdges() << "\n";
  std::cout << "Vertices : " << currentMesh->getNumVerts() << "\n";
  std::cout << "Edges : " << currentMesh->getNumEdges() << "\n\n";
  currentMesh = subdivisionLevel == 1 ? in : out;
  return milsecs;
}

void Subdivider::singleSubdivisionStep(int k) {
  QuadMesh* newMesh = new QuadMesh();
  QElapsedTimer timer;
  timer.start();
  currentMesh->subdivideCatmullClark(*newMesh);
  /* Display info to user */
  long long time = timer.nsecsElapsed();
  std::cout << "Subdivision time at level " << k + 1 << " is "
            << time / 1000000.0 << " milliseconds\n";
  if (currentMesh != baseMesh) {
    delete currentMesh;
  }
  currentMesh = newMesh;
}
