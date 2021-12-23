#include "kernel.h"

#define BLOCK_SIZE 64

//------------------------------------------------------------------------
//-------------------------------- Kernels calculation -------------------
//------------------------------------------------------------------------

__device__ mod_prec icNeighbours_thread(CellCompParams *params,
                                        CellState *cells, int neighbourCount) {
  int t = threadIdx.x;
  int stride = blockDim.x;
  int *neighIds = params->neighId;
  mod_prec *neighConductances = params->neighConductances;
  mod_prec prevV_dend = params->prevCellState->dend.V_dend;

  mod_prec I_c = 0;
  for (int i = t; i < neighbourCount; i += stride) {
    // merge this step with the communication step so that everything can be
    // done in a single for-loop :D
    // as a matter of fact, we don't even need the neighVDend array in the
    // params anymore this way; saving us some space
    mod_prec neighVDendVal = cells[neighIds[i]].dend.V_dend;
    mod_prec V = prevV_dend - neighVDendVal;
    mod_prec f =
        0.8 * exp(-1 * pow(V, 2) / 100) + 0.2;  // SCHWEIGHOFER 2004 VERSION
    mod_prec cond = neighConductances[i];
    I_c += (cond * f * V);
  }

  return I_c;
}

// Initially this was the communication kernel.
// However, this can be merged with the loop in iC_neighbours, making this
// kernel redundant
// I kept it here for readability sake
__global__ void communicationStepKernel(CellCompParams *params,
                                        const CellState *cells) {
  CellCompParams param = params[blockIdx.x];
  int i = threadIdx.x;
  if (i < param.total_amount_of_neighbours) {
    param.neighVdend[i] = cells[param.neighId[i]].dend.V_dend;
  }
}

__global__ void calculationStepKernel(float** vertCoords, int numVerts, int numHalfEdges, int numFaces, int numEdges, int* twins, int* nexts, int* prevs, int* verts, int* edges, int* faces) {
}

int nextPow2(int x) { return pow(2, ceil(log(x) / log(2))); }

//------------------------------------------------------------------------
//-------------------------------- Simulation Parallel -------------------
//------------------------------------------------------------------------

void performSimulationParallel(CellCompParams *cellParamsPtr,
                               CellState *prevStates, CellState *newStates,
                               int cellCount, int totalSimSteps) {
  cudaError_t cuda_ret;

  printf("Allocating device variables...\n");
  fflush(stdout);
  // copy and alloc cuda stuff
  // Allocate device variables
  CellState *prevStates_d = allocCellPtrDevice(cellCount);
  CellState *newStates_d = allocCellPtrDevice(cellCount);
  CellCompParams *cellParamsPtr_d = allocCellParamsDevice(cellCount);
  cudaDeviceSynchronize();

  printf("Copying data from host to device...\n");
  fflush(stdout);
  copyToDevice(cellParamsPtr, prevStates, newStates, cellParamsPtr_d,
               prevStates_d, newStates_d, cellCount);

  cudaDeviceSynchronize();

  dim3 dim_grid, dim_block;

  dim_block.x = BLOCK_SIZE;
  dim_block.y = dim_block.z = 1;
  dim_grid.x = cellCount;
  dim_grid.y = dim_grid.z = 1;

  printf("Executing simulation...\n");
  fflush(stdout);
  timestamp_t t0 = getTimeStamp();
  // Execute simulation
  // Loop invariant: at the start of each iteration, the current state of the
  // network is stored in prevStates

  for (int simStep = 0; simStep < totalSimSteps; simStep++) {
    mod_prec iApp;
    if ((simStep >= 20000) && (simStep < 20500 - 1)) {
      iApp = 6;
    } else {
      iApp = 0;
    }
    calculationStepKernel<<<dim_grid, dim_block>>>(
        cellParamsPtr_d, prevStates_d, newStates_d, iApp, cellCount);
    // For the next iteration, the new states become the previous states
    stateSwap(&prevStates_d, &newStates_d);
  }

  // Sync
  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess) {
    FATAL("Unable to launch/execute kernel");
  }
  timestamp_t t1 = getTimeStamp();
  printf("Kernel Execution Time : %lld usecs\n", t1 - t0);
  // Copy device variables back to host
  printf("Copying data from device to host...\n");
  fflush(stdout);
  copyToHost(prevStates, prevStates_d, cellCount);

  cudaDeviceSynchronize();
  // Free device variables

  cudaFreeCellParams(cellParamsPtr_d, cellCount);
  cudaFree(prevStates_d);
  cudaFree(newStates_d);
}