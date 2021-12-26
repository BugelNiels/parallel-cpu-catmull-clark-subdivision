#ifndef QUAD_REFINEMENT_CUH
#define QUAD_REFINEMENT_CUH

#include "../mesh/deviceMesh.cuh"

#define BLOCK_SIZE 64
#define MAX_GRID_SIZE 1024

__global__ void resetMesh(DeviceMesh* in, DeviceMesh* out);
__global__ void quadRefineEdges(DeviceMesh* in, DeviceMesh* out);
__global__ void quadFacePoints(DeviceMesh* in, DeviceMesh* out);
__global__ void quadEdgePoints(DeviceMesh* in, DeviceMesh* out);
__global__ void quadVertexPoints(DeviceMesh* in, DeviceMesh* out);
__global__ void debugKernel(DeviceMesh* in);
__global__ void debugKernel2(DeviceMesh in);

#endif // QUAD_REFINEMENT_CUH