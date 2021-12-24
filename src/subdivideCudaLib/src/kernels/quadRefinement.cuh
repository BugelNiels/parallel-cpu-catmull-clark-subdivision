#ifndef QUAD_REFINEMENT_CUH
#define QUAD_REFINEMENT_CUH

#include "../mesh.cuh"

__global__ void quadRefineEdgesAndCalcFacePoints(Mesh in, Mesh out);
__global__ void quadEdgePoints(Mesh in, Mesh out);
__global__ void quadVertexPoints(Mesh in, Mesh out);

#endif // QUAD_REFINEMENT_CUH