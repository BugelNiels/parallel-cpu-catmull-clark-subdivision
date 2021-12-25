#ifndef KERNEL_INVOKER_CUH
#define KERNEL_INVOKER_CUH

#include "mesh/mesh.cuh"

Mesh performSubdivision(Mesh in, Mesh out, int subdivisionLevel, int h0);

#endif // KERNEL_INVOKER_CUH