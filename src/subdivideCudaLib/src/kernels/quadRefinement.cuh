#ifndef QUAD_REFINEMENT_CUH
#define QUAD_REFINEMENT_CUH

#include "../mesh/deviceMesh.cuh"

#define BLOCK_SIZE 64

__global__ void quadRefineEdgesAndCalcFacePoints(DeviceMesh in, DeviceMesh out);
__global__ void quadEdgePoints(DeviceMesh in, DeviceMesh out);
__global__ void quadVertexPoints(DeviceMesh in, DeviceMesh out);
__global__ void debugKernel(DeviceMesh in);

#endif // QUAD_REFINEMENT_CUH