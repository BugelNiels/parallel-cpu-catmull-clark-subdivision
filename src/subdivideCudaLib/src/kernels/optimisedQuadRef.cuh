#ifndef OPTIMISED_QUAD_REFINEMENT_CUH
#define OPTIMISED_QUAD_REFINEMENT_CUH

#include "quadRefinement.cuh"

__global__ void optQuadVertexPoints(DeviceMesh* in, DeviceMesh* out);
__global__ void optimisedSubdivide(DeviceMesh* in, DeviceMesh* out);

#endif // OPTIMISED_QUAD_REFINEMENT_CUH