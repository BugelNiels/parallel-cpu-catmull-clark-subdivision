#include "kernelInvoker.cuh"
#include "kernels/quadRefinement.cuh"
#include "kernels/optimisedQuadRef.cuh"
#include "util/util.cuh"

#include "stdio.h"
#include "math.h"

#define USE_OPTIMIZED_KERNEL 1

// swaps pointers
void meshSwap(DeviceMesh **prevMeshPtr, DeviceMesh **newMeshPtr) {
  DeviceMesh *temp = *prevMeshPtr;
  *prevMeshPtr = *newMeshPtr;
  *newMeshPtr = temp;
}

DeviceMesh performSubdivision(DeviceMesh* input, DeviceMesh* output, int subdivisionLevel, int h0) {
  cudaError_t cuda_ret;
  Timer timer;

  // device must be synced before this point

	DeviceMesh* in = toDevicePointer(input);
	DeviceMesh* out = toDevicePointer(output);

  dim3 dim_grid, dim_block;

  // TODO: calculate
  // each thread takes 1 half edge
  dim_block.x = BLOCK_SIZE;
  dim_block.y = dim_block.z = 1;
  dim_grid.y = dim_grid.z = 1;

  printf("\n\n------------------\nPerforming subdivision\n...\n"); fflush(stdout);
  startTime(&timer);
  
  // if faces are set, means it is not a quad mesh, hence do one separate step with kernels
  // quadRefineEdgesAndCalcFacePoints<<<dim_grid, dim_block>>>(*in, *out);

  for (int d = 0; d < subdivisionLevel; d++) {
    // each thread covers 1 half edge. Number of half edges can be much greater than blockdim * gridDim. 
    int he = pow(4, d) * h0;
    dim_grid.x = MIN((he - 1) / BLOCK_SIZE + 1, MAX_GRID_SIZE);
    if(USE_OPTIMIZED_KERNEL) {
      // debugKernel<<<dim_grid, dim_block>>>(in);
      resetMesh<<<dim_grid, dim_block>>>(in, out);
      optimisedSubdivide<<<dim_grid, dim_block>>>(in, out);
      // optQuadVertexPoints<<<dim_grid, dim_block>>>(in, out);
      // debugKernel<<<dim_grid, dim_block>>>(out);
    } else {
      resetMesh<<<dim_grid, dim_block>>>(in, out);
      quadRefineEdges<<<dim_grid, dim_block>>>(in, out);
      quadFacePoints<<<dim_grid, dim_block>>>(in, out);
      quadEdgePoints<<<dim_grid, dim_block>>>(in, out);
      quadVertexPoints<<<dim_grid, dim_block>>>(in, out);
    }
    // calculate better grid size
    // debugKernel<<<dim_grid, dim_block>>>(out);
    // result is in out; after this swap, the result is in in
    meshSwap(&in, &out);
  }

  cuda_ret = cudaDeviceSynchronize();
  cudaErrCheck(cuda_ret, "Unable to execute kernel");

  stopTime(&timer); printf("Kernel execution took: %f s\n------------------\n\n", elapsedTime(timer));
  DeviceMesh m =  devicePointerToHostMesh(in);
  // debugKernel<<<dim_grid, dim_block>>>(in);
  // cuda free in and out
  cudaFree(in);
  cudaFree(out);
  return m;
  
}