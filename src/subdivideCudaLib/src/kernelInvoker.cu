#include "kernelInvoker.cuh"
#include "kernels/quadRefinement.cuh"
#include "util/util.cuh"

#include "stdio.h"


// swaps pointers
void meshSwap(Mesh **prevMeshPtr, Mesh **newMeshPtr) {
  Mesh *temp = *prevMeshPtr;
  *prevMeshPtr = *newMeshPtr;
  *newMeshPtr = temp;
}

void performSubdivision(Mesh input, Mesh output, int subdivisionLevel) {
  cudaError_t cuda_ret;
  Timer timer;

  // device must be synced before this point

  Mesh* in = &input;
  Mesh* out = &output;

  dim3 dim_grid, dim_block;

  // TODO: calculate
  dim_block.x = BLOCK_SIZE;
  dim_block.y = dim_block.z = 1;
  dim_grid.x = BLOCK_SIZE;
  dim_grid.y = dim_grid.z = 1;

  printf("Starting subdivision.."); fflush(stdout);
  startTime(&timer);
  
  // if faces are set, means it is not a quad mesh, hence do one separate step with kernels
  quadRefineEdgesAndCalcFacePoints<<<dim_grid, dim_block>>>(*in, *out);

  // for (int d = 0; d < subdivisionLevel; d++) {
  //   quadRefineEdgesAndCalcFacePoints<<<dim_grid, dim_block>>>(*in, *out);
  //   quadEdgePoints<<<dim_grid, dim_block>>>(*in, *out);
  //   quadVertexPoints<<<dim_grid, dim_block>>>(*in, *out);
  //   meshSwap(&in, &out);
  // }
  cuda_ret = cudaDeviceSynchronize();
  cudaErrCheck(cuda_ret, "Unable to execute kernel");

  stopTime(&timer); printf("%f s\n", elapsedTime(timer));

  
}