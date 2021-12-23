#include "kernel.h"
#include "mesh.h"

#define BLOCK_SIZE 64

//------------------------------------------------------------------------
//-------------------------------- Kernels calculation -------------------
//------------------------------------------------------------------------

__global__ void subdivideMesh(Mesh in, Mesh out) {
  
}

// swaps pointers
void meshSwap(Mesh **prevMeshPtr, Mesh **newMeshPtr) {
  Mesh *temp = *prevMeshPtr;
  *prevMeshPtr = *newMeshPtr;
  *newMeshPtr = temp;
}

//------------------------------------------------------------------------
//-------------------------------- Simulation Parallel -------------------
//------------------------------------------------------------------------

void performSubdivision(Mesh input, Mesh output, int subdivisionLevel) {
  cudaError_t cuda_ret;



  Mesh* in = &input;
  Mesh* out = &output;

  dim3 dim_grid, dim_block;

  // TODO: calculate
  dim_block.x = BLOCK_SIZE;
  dim_block.y = dim_block.z = 1;
  dim_grid.x = BLOCK_SIZE;
  dim_grid.y = dim_grid.z = 1;

  for (int d = 0; d < subdivisionLevel; d++) {
    subdivideMesh<<<dim_grid, dim_block>>>(in, out);
    meshSwap(&in, &out);
  }

  
}