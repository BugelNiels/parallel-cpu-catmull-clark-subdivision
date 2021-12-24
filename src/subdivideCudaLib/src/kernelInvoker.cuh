#ifndef KERNEL_INVOKER_CUH
#define KERNEL_INVOKER_CUH

#include "mesh/mesh.cuh"

void performSubdivision(Mesh in, Mesh out, int subdivisionLevel);

#endif // KERNEL_INVOKER_CUH