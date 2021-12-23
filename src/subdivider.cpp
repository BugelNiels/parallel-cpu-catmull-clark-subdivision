#include "subdivider.h"

#include <QElapsedTimer>
#include <iostream>

#include "quadmesh.h"

Subdivider::Subdivider(Mesh* mesh) {
  baseMesh = mesh;
  currentMesh = baseMesh;
}

double Subdivider::subdivide(int subdivisionLevel, int iterations) {
  double totalTime = 0;
  for (int i = 0; i < iterations; ++i) {
    currentMesh = baseMesh;
    QElapsedTimer timer;
    timer.start();
    for (unsigned short k = 0; k < subdivisionLevel; k++) {
      QuadMesh* newMesh = new QuadMesh();
      currentMesh->subdivideCatmullClark(*newMesh);
      if (currentMesh != baseMesh) {
        delete currentMesh;
      }
      currentMesh = newMesh;
    }
    long long time = timer.nsecsElapsed();
    totalTime += time / 1000000.0;
  }
  double avgTime = totalTime / iterations;
  std::cout << "\n---\nAverage time elapsed for " << subdivisionLevel << " : "
            << avgTime << " milliseconds\n\n";
  return avgTime;
}

double Subdivider::subdivide(int subdivisionLevel) {
  currentMesh = baseMesh;

  QElapsedTimer timer;
  timer.start();
  for (unsigned short k = 0; k < subdivisionLevel; k++) {
    singleSubdivisionStep(k);
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
