#include "kernelInvoker.cuh"
#include "kernels/quadRefinement.cuh"
#include "kernels/optimisedQuadRef.cuh"
#include "util/util.cuh"
#include "deviceCommunication.cuh"

#include "stdio.h"
#include "math.h"

#define USE_OPTIMIZED_KERNEL 1

// swaps pointers
void meshSwap(DeviceMesh **prevMeshPtr, DeviceMesh **newMeshPtr) {
  DeviceMesh *temp = *prevMeshPtr;
  *prevMeshPtr = *newMeshPtr;
  *newMeshPtr = temp;
}

DeviceMesh performSubdivision(DeviceMesh* input, DeviceMesh* output, int subdivisionLevel, Mesh* mesh) {
  cudaError_t cuda_ret;
  cudaEvent_t start, stop;

  int isQuad = mesh->nexts == NULL || mesh->prevs == NULL || mesh->faces == NULL;
  int h0 = mesh->numHalfEdges;
  int v0 = mesh->numVerts;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
	DeviceMesh* in = toDevicePointer(input);
	DeviceMesh* out = toDevicePointer(output);

  
  dim3 dim_grid, dim_block;

  // each thread takes 1 half edge
  dim_block.x = BLOCK_SIZE;
  dim_block.y = dim_block.z = 1;
  dim_grid.y = dim_grid.z = 1;

  printf("\n\n------------------\nPerforming subdivision\n...\n");

  // device must be synced before this point
  cudaEventRecord(start);
  // all the stuff before this can be pre-allocated/pre-calculated

  copyHostToDeviceMesh(mesh, input, isQuad);	

  // if faces are set, means it is not a quad mesh, hence do one separate step with kernels
  
  for (int d = 0; d < subdivisionLevel; d++) {
    // each thread covers 1 half edge. Number of half edges can be much greater than blockdim * gridDim. 
    int he = pow(4, d) * h0;
    dim_grid.x = MIN((he - 1) / BLOCK_SIZE + 1, MAX_GRID_SIZE);
    if(USE_OPTIMIZED_KERNEL) {
      resetMesh<<<dim_grid, dim_block>>>(in, out);
      optimisedSubdivide<<<dim_grid, dim_block>>>(in, out, v0);
    } else {
      resetMesh<<<dim_grid, dim_block>>>(in, out);
      quadRefineEdges<<<dim_grid, dim_block>>>(in, out);
      quadFacePoints<<<dim_grid, dim_block>>>(in, out);
      quadEdgePoints<<<dim_grid, dim_block>>>(in, out);
      quadVertexPoints<<<dim_grid, dim_block>>>(in, out);
    }
    // result is in "out"; after this swap, the result is in "in"
    meshSwap(&in, &out);
  }
  cudaEventRecord(stop);
  cuda_ret = cudaDeviceSynchronize();
  cudaErrCheck(cuda_ret, "Unable to execute kernel");
  
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Execution took: %lf msec\n------------------\n\n", milliseconds);
  DeviceMesh m =  devicePointerToHostMesh(in);
  
  // cuda free in and out
  cudaFree(in);
  cudaFree(out);
  return m;
  
}