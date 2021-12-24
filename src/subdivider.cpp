#include "subdivider.h"

#include <QElapsedTimer>
#include <iostream>

#include "quadmesh.h"
#include "subdivide.h"

Subdivider::Subdivider(Mesh* mesh) {
  baseMesh = mesh;
  currentMesh = baseMesh;
}

int* copyIntArr(int* data, int size) {
  int* arr = (int*)malloc(size * sizeof(int));
  for(int i = 0; i < size; ++i) {
    arr[i] = data[i];
  }
  return arr;
}

void Subdivider::subdivideGPU(int subdivisionLevel) {
  // makes all calls a lot shorter;
  Mesh* m = baseMesh;
  float* xCoords = (float*)malloc(m->getNumVerts() * sizeof(float));
  float* yCoords = (float*)malloc(m->getNumVerts() * sizeof(float));
  float* zCoords = (float*)malloc(m->getNumVerts() * sizeof(float));
  int i = 0;
  for (QVector3D c : m->getVertexCoords()) {
    xCoords[i] = c.x();
    yCoords[i] = c.y();
    zCoords[i] = c.z();
    i++;
  }
  int* twins = copyIntArr(m->getTwins().data(), m->getNumHalfEdges());
  int* nexts = copyIntArr(m->getNexts().data(), m->getNumHalfEdges());
  int* prevs = copyIntArr(m->getPrevs().data(), m->getNumHalfEdges());
  int* verts = copyIntArr(m->getVerts().data(), m->getNumHalfEdges());
  int* edges = copyIntArr(m->getEdges().data(), m->getNumHalfEdges());
  int* faces = copyIntArr(m->getFaces().data(), m->getNumHalfEdges());
  double milsecs;
  // meshSubdivision(xCoords, yCoords, zCoords, m->getNumVerts(),
  //                            m->getNumHalfEdges(), m->getNumFaces(),
  //                            m->getNumEdges(), twins, nexts, prevs, verts,
  //                            edges, faces, subdivisionLevel);
  quadMeshSubdivision(xCoords, yCoords, zCoords, m->getNumVerts(),
                             m->getNumHalfEdges(), m->getNumFaces(),
                             m->getNumEdges(), twins, verts,
                             edges, subdivisionLevel);
  // TODO: if base mesh is quad mesh, call quadMeshSubdivision
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
