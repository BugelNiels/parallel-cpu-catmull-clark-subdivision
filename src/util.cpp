#include "util.h"

/**
 * @brief atomicAdd Atomic adds vector B to vector A. Performs A += B
 * @param vecA Vector A
 * @param vecB Vector B
 */
void atomicAdd(QVector3D& vecA, const QVector3D& vecB) {
  for (int k = 0; k < 3; ++k) {
    float& a = vecA[k];
    const float b = vecB[k];
#pragma omp atomic
    a += b;
  }
}
