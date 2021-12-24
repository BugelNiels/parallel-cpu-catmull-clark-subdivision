#ifndef QUAD_REFINEMENT_CUH
#define QUAD_REFINEMENT_CUH

#include "../mesh/mesh.cuh"

#define BLOCK_SIZE 64

__global__ void quadRefineEdgesAndCalcFacePoints(Mesh in, Mesh out);
__global__ void quadEdgePoints(Mesh in, Mesh out);
__global__ void quadVertexPoints(Mesh in, Mesh out);

#endif // QUAD_REFINEMENT_CUH