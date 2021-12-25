#include "quadRefinement.cuh"
#include "../util/util.cuh"

#define WARP_SIZE 32
#define FACES_PER_BLOCK (BLOCK_SIZE / 4)

// TODO: force inline?
__device__ int next(int h) { return h % 4 == 3 ? h - 3 : h + 1; }

__device__ int prev(int h) { return h % 4 == 0 ? h + 3 : h - 1; }

__device__ int face(int h) { return h / 4; }

__device__ void debugMesh(DeviceMesh* mesh) {
    if(blockIdx.x == 0 && threadIdx.x == 0) {
        printf("%d %d %d %d\n", *mesh->numVerts, *mesh->numHalfEdges, *mesh->numFaces, *mesh->numEdges);
        for(int i = 0; i < *mesh->numVerts; i++) {
            printf("(%.2lf %.2lf %.2lf) ", mesh->xCoords[i], mesh->yCoords[i], mesh->zCoords[i]);
        }
        printf("\nTwins:\n");
        for(int i = 0; i < *mesh->numHalfEdges; i++) {
            printf("%d ", mesh->twins[i]);
        }
        printf("\nVerts:\n");
        for(int i = 0; i < *mesh->numHalfEdges; i++) {
            printf("%d ", mesh->verts[i]);
        }
        printf("\nEdges:\n");
        for(int i = 0; i < *mesh->numHalfEdges; i++) {
            printf("%d ", mesh->edges[i]);
        }
    }
}

// TODO: change to pointers?
__device__ void edgeRefinement(int h, DeviceMesh* in, DeviceMesh* out) {
    int hp = prev(h);
    int he = in->edges[h];

    int vd = *in->numVerts;
    int fd = *in->numFaces;
    int ed = *in->numEdges;

    // For boundaries
    int ht = in->twins[h];
    int thp = in->twins[hp];
    int ehp = in->edges[hp];
    // TODO: check interleave
    out->twins[4 * h] = ht < 0 ? -1 : 4 * next(ht) + 3;
    out->twins[4 * h + 1] = 4 * next(h) + 2;
    out->twins[4 * h + 2] = 4 * hp + 1;
    out->twins[4 * h + 3] = 4 * thp;

    out->verts[4 * h] = in->verts[h];
    out->verts[4 * h + 1] = vd + fd + he;
    out->verts[4 * h + 2] = vd + face(h);
    out->verts[4 * h + 3] = vd + fd + ehp;

    out->edges[4 * h] = h > ht ? 2 * he : 2 * he + 1;
    out->edges[4 * h + 1] = 2 * ed + h;
    out->edges[4 * h + 2] = 2 * ed + hp;
    out->edges[4 * h + 3] = hp > thp ? 2 * ehp + 1 : 2 * ehp;
}

__device__ int valence(int h, DeviceMesh* in) {
  int ht = in->twins[h];
  if (ht < 0) {
    return -1;
  }
  int n = 1;
  int hp = next(ht);
  // Branch divergence issue; pay attention to this later
  while (hp != h) {
    if (hp < 0) {
      return -1;
    }
    ht = in->twins[hp];
    if (ht < 0) {
      return -1;
    }
    hp = next(ht);
    n++;
  }
  return n;
}

__device__ void facePoint(int h, DeviceMesh* in, DeviceMesh* out, float* vertCoordsX, float* vertCoordsY, float* vertCoordsZ) {
    int v = in->verts[h];
    int i = *in->numVerts + face(h);
    int ti = i % BLOCK_SIZE;
    vertCoordsX[ti] += in->xCoords[v];
    vertCoordsY[ti] += in->yCoords[v];
    vertCoordsZ[ti] += in->zCoords[v];

    __syncthreads();

    int fi = threadIdx.x % 4;
    if(fi == 0) {
        out->xCoords[i] = vertCoordsX[ti] / 4.0f;
        out->yCoords[i] = vertCoordsY[ti] / 4.0f;
        out->zCoords[i] = vertCoordsZ[ti] / 4.0f;
    } 
}

__global__ void quadRefineEdges(DeviceMesh in, DeviceMesh out) {

    int numEdges = 2 * *in.numEdges + *in.numHalfEdges;
    int numFaces = *in.numHalfEdges;
    int numHalfEdges = *in.numHalfEdges * 4;
    int numVerts = *in.numVerts + *in.numFaces + *in.numEdges;

    // TODO: check that h does not go out of bounds.
    // TODO 2: loop

    // half edge index
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    if(h < *in.numHalfEdges) {
        edgeRefinement(h, &in, &out);
    }
    
    if(blockIdx.x == 0 && threadIdx.x == 0) {
        // TODO: each statement by different thread? (with offset WARP_SIZE)
        *out.numEdges = numEdges;
        *out.numFaces = numFaces;
        *out.numHalfEdges = numHalfEdges;
        *out.numVerts = numVerts;
    }
}

__global__ void quadFacePoints(DeviceMesh in, DeviceMesh out) {
    
    // each block covers BLOCK_SIZE / 4 faces
    __shared__ float vertCoordsX[FACES_PER_BLOCK];
    __shared__ float vertCoordsY[FACES_PER_BLOCK];
    __shared__ float vertCoordsZ[FACES_PER_BLOCK];

    if(threadIdx.x < FACES_PER_BLOCK) {
        vertCoordsX[threadIdx.x] = 0;
        vertCoordsY[threadIdx.x] = 0;
        vertCoordsZ[threadIdx.x] = 0;
    }
    // probably redundent
    __syncthreads();
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    int i = *in.numVerts + face(h);
    if(h < *in.numHalfEdges) {
    
        int v = in.verts[h];
        int ti = i % BLOCK_SIZE;
        // face point calculation
        atomicAdd(&vertCoordsX[ti], in.xCoords[v]);
        atomicAdd(&vertCoordsY[ti], in.yCoords[v]);
        atomicAdd(&vertCoordsZ[ti], in.zCoords[v]);
    }
    
    __syncthreads();

    if(h < *in.numHalfEdges && threadIdx.x < FACES_PER_BLOCK) {
        out.xCoords[i] = vertCoordsX[threadIdx.x] / 4.0f;
        out.yCoords[i] = vertCoordsY[threadIdx.x] / 4.0f;
        out.zCoords[i] = vertCoordsZ[threadIdx.x] / 4.0f;
    }
}

__global__ void quadEdgePoints(DeviceMesh in, DeviceMesh out) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if(h >= *in.numHalfEdges) {
        return;
    }
    int vd = *in.numVerts;
    int fd = *in.numFaces;
    int v = in.verts[h];
    int j = vd + fd + in.edges[h];

    int th = in.twins[h] < 0;
    // if th < 0 then boundary
    int i = th < 0 ? in.verts[next(h)] : vd + face(h) ;
    int divisor = th < 0 ? 2.0f : 4.0f;

    DeviceMesh* m = th < 0 ? &in : &out;

    float x = (m->xCoords[v] + in.xCoords[i]) / divisor;
    float y = (m->yCoords[v] + in.yCoords[i]) / divisor;
    float z = (m->zCoords[v] + in.zCoords[i]) / divisor;
    // Atomic add
    // doubles?
    // atomic add not necessary for boundaries, but likely outweighs branch divergence penalties
    atomicAdd(&out.xCoords[j], x);
    atomicAdd(&out.yCoords[j], y);
    atomicAdd(&out.zCoords[j], z);
    
}
__global__ void quadVertexPoints(DeviceMesh in, DeviceMesh out) {
    // shared memory?
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if(h >= *in.numHalfEdges) {
        return;
    }
    float n = valence(h, &in);
    int v = in.verts[h];

    float x = in.xCoords[v];
    float y = in.yCoords[v];
    float z = in.zCoords[v];
    if(n >= 0) {
        int vd = *in.numVerts;
        int i = vd + face(h);
        int j = vd + *in.numFaces + in.edges[h];
        float n2 = n * n;
        x = (4 * out.xCoords[j] - out.xCoords[i] + (n - 3) * x) / n2;
        y = (4 * out.yCoords[j] - out.yCoords[i] + (n - 3) * x) / n2;
        z = (4 * out.zCoords[j] - out.zCoords[i] + (n - 3) * x) / n2;
    }
    atomicAdd(&out.xCoords[v], x);
    atomicAdd(&out.yCoords[v], y);
    atomicAdd(&out.zCoords[v], z);
}



__global__ void debugKernel(DeviceMesh in) {
    debugMesh(&in);
}