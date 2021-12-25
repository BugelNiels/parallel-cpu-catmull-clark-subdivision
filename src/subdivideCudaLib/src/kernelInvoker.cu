#include "kernelInvoker.cuh"
#include "kernels/quadRefinement.cuh"
#include "util/util.cuh"

#include "stdio.h"
#include "math.h"


// swaps pointers
void meshSwap(DeviceMesh **prevMeshPtr, DeviceMesh **newMeshPtr) {
  DeviceMesh *temp = *prevMeshPtr;
  *prevMeshPtr = *newMeshPtr;
  *newMeshPtr = temp;
}

DeviceMesh performSubdivision(DeviceMesh input, DeviceMesh output, int subdivisionLevel, int h0) {
  cudaError_t cuda_ret;
  Timer timer;

  // device must be synced before this point

  // swap these two, so that the initial mesh swap puts them right



  DeviceMesh* in = &input;
  DeviceMesh* out = &output;

  dim3 dim_grid, dim_block;

  // TODO: calculate
  // each thread takes 1 half edge
  dim_block.x = BLOCK_SIZE;
  dim_block.y = dim_block.z = 1;
  dim_grid.y = dim_grid.z = 1;

  printf("Performing subdivision..\n"); fflush(stdout);
  startTime(&timer);
  
  // if faces are set, means it is not a quad mesh, hence do one separate step with kernels
  // quadRefineEdgesAndCalcFacePoints<<<dim_grid, dim_block>>>(*in, *out);

  for (int d = 0; d < subdivisionLevel; d++) {
    // number of half edges at this level
    int he = pow(4, d) * h0;
    // TODO: take care of max grid size
    dim_grid.x = (he - 1) / BLOCK_SIZE + 1;
    printf("Num half edges to cover: %d -- Grid size: %d\n", he, dim_grid.x);
    // debugKernel<<<dim_grid, dim_block>>>(*in);
    quadRefineEdges<<<dim_grid, dim_block>>>(*in, *out);
    quadFacePoints<<<dim_grid, dim_block>>>(*in, *out);
    quadEdgePoints<<<dim_grid, dim_block>>>(*in, *out);
    quadVertexPoints<<<dim_grid, dim_block>>>(*in, *out);
    // result is in out; after this swap, the result is in in
    // debugKernel<<<dim_grid, dim_block>>>(*in);
    meshSwap(&in, &out);
  }

  debugKernel<<<dim_grid, dim_block>>>(*in);
  cuda_ret = cudaDeviceSynchronize();
  cudaErrCheck(cuda_ret, "Unable to execute kernel");

  stopTime(&timer); printf("Kernel execution took: %f s\n\n", elapsedTime(timer));
  return *in;
  
}