#ifndef QUAD_REFINEMENT_CUH
#define QUAD_REFINEMENT_CUH

#include "../mesh/deviceMesh.cuh"

#define BLOCK_SIZE 128
#define MAX_GRID_SIZE 32768
#define WARP_SIZE 32
#define FACES_PER_BLOCK (BLOCK_SIZE / 4)

__global__ void resetMesh(DeviceMesh* in, DeviceMesh* out);
__global__ void quadRefineEdges(DeviceMesh* in, DeviceMesh* out);
__global__ void quadFacePoints(DeviceMesh* in, DeviceMesh* out);
__global__ void quadEdgePoints(DeviceMesh* in, DeviceMesh* out);
__global__ void quadVertexPoints(DeviceMesh* in, DeviceMesh* out);
__device__ int valence(int h, DeviceMesh* in);
__global__ void debugKernel(DeviceMesh* in);
__global__ void debugKernel2(DeviceMesh in);

#endif // QUAD_REFINEMENT_CUH