#include "quadRefinement.cuh"
#include "../util/util.cuh"



// TODO: force inline?
inline __device__ int next(int h) { return h % 4 == 3 ? h - 3 : h + 1; }

inline __device__ int prev(int h) { return h % 4 == 0 ? h + 3 : h - 1; }

inline __device__ int face(int h) { return h / 4; }

__device__ void debug(char const* m) {
    if(blockIdx.x == 0 && threadIdx.x == 0) {
        printf("%s\n", m);
    }    
}

__device__ void debugMesh(DeviceMesh* mesh) {
    if(blockIdx.x == 0 && threadIdx.x == 0) {
        printf("%d %d %d %d\n", mesh->numVerts, mesh->numHalfEdges, mesh->numFaces, mesh->numEdges);
        for(int i = 0; i < mesh->numVerts; i++) {
            printf("(%.2lf %.2lf %.2lf) ", mesh->xCoords[i], mesh->yCoords[i], mesh->zCoords[i]);
        }
        printf("\nTwins:\n");
        for(int i = 0; i < mesh->numHalfEdges; i++) {
            printf("%d ", mesh->twins[i]);
        }
        printf("\nVerts:\n");
        for(int i = 0; i < mesh->numHalfEdges; i++) {
            printf("%d ", mesh->verts[i]);
        }
        printf("\nEdges:\n");
        for(int i = 0; i < mesh->numHalfEdges; i++) {
            printf("%d ", mesh->edges[i]);
        }
        printf("\n\n");
    }
}

__device__ void edgeRefinement(int h, DeviceMesh* in, DeviceMesh* out, int vd, int fd, int ed) {
    int hp = prev(h);
    int he = in->edges[h];


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

__device__ void quadFacePoint(int h, DeviceMesh* in, DeviceMesh* out) {
    int v = in->verts[h];
    int i = in->numVerts + face(h);
    atomicAdd(&out->xCoords[i], in->xCoords[v] / 4.0f);
    atomicAdd(&out->yCoords[i], in->yCoords[v] / 4.0f);
    atomicAdd(&out->zCoords[i], in->zCoords[v] / 4.0f);
}

__device__ void quadEdgePoint(int h, DeviceMesh* in, DeviceMesh* out) {
    int vd = in->numVerts;
    int fd = in->numFaces;
    int v = in->verts[h];
    int j = vd + fd + in->edges[h];
    float x, y, z;
    // boundary
    if(in->twins[h] < 0) {
        int i = in->verts[next(h)];
        x = (in->xCoords[v] + in->xCoords[i]) / 2.0f;
        y = (in->yCoords[v] + in->yCoords[i]) / 2.0f;
        z = (in->zCoords[v] + in->zCoords[i]) / 2.0f;
        // atomic add not necessary for boundaries, but likely outweighs branch divergence penalties        
    } else {
        int i = vd + face(h);
        x = (in->xCoords[v] + out->xCoords[i]) / 4.0f;
        y = (in->yCoords[v] + out->yCoords[i]) / 4.0f;
        z = (in->zCoords[v] + out->zCoords[i]) / 4.0f;
    }    
    // TODO inline if
    atomicAdd(&out->xCoords[j], x);
    atomicAdd(&out->yCoords[j], y);
    atomicAdd(&out->zCoords[j], z);
}

__device__ void quadVertexPoint(int h, DeviceMesh* in, DeviceMesh* out) {
    // shared memory?
    int v = in->verts[h];
    float n = valence(h, in);

    float x = in->xCoords[v];
    float y = in->yCoords[v];
    float z = in->zCoords[v];

    // boundary half edge
    if(in->twins[h] < 0) {
        out->xCoords[v] = x;
        out->yCoords[v] = y;
        out->zCoords[v] = z;
    } else if(n >= 0) {
        int vd = in->numVerts;
        int i = vd + face(h);
        int j = vd + in->numFaces + in->edges[h];
        float n2 = n * n;
        x = (4 * out->xCoords[j] - out->xCoords[i] + (n - 3) * in->xCoords[v]) / n2;
        y = (4 * out->yCoords[j] - out->yCoords[i] + (n - 3) * in->yCoords[v]) / n2;
        z = (4 * out->zCoords[j] - out->zCoords[i] + (n - 3) * in->zCoords[v]) / n2;
        atomicAdd(&out->xCoords[v], x);
        atomicAdd(&out->yCoords[v], y);
        atomicAdd(&out->zCoords[v], z);
    }
}

__global__ void resetMesh(DeviceMesh* in, DeviceMesh* out) {
    int numEdges = 2 * in->numEdges + in->numHalfEdges;
    int numFaces = in->numHalfEdges;
    int numHalfEdges = in->numHalfEdges * 4;
    int numVerts = in->numVerts + in->numFaces + in->numEdges;
    
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = h; i < numVerts; i += stride) {
        out->xCoords[i] = 0;
        out->yCoords[i] = 0;
        out->zCoords[i] = 0;
    } 
    
    if(blockIdx.x == 0) {
        out->numEdges = numEdges;
        out->numFaces = numFaces;
        out->numHalfEdges = numHalfEdges;
        out->numVerts = numVerts;
    }
}

__global__ void quadRefineEdges(DeviceMesh* in, DeviceMesh* out) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int vd = in->numVerts;
    int fd = in->numFaces;
    int ed = in->numEdges;
    for(int i = h; i < in->numHalfEdges; i += stride) {
        edgeRefinement(i, in, out, vd, fd, ed);
    }    
}

__global__ void quadFacePoints(DeviceMesh* in, DeviceMesh* out) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    h *= 4;
    int stride = blockDim.x * gridDim.x;
    for(int i = h; i < in->numHalfEdges; i += stride) {
        quadFacePoint(i, in, out);
    } 
}

__global__ void quadEdgePoints(DeviceMesh* in, DeviceMesh* out) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = h; i < in->numHalfEdges; i += stride) {
        quadEdgePoint(i, in, out);
    } 
}

__global__ void quadVertexPoints(DeviceMesh* in, DeviceMesh* out) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = h; i < in->numHalfEdges; i += stride) {
        quadVertexPoint(i, in, out);
    }   
}



__global__ void debugKernel(DeviceMesh* in) {
    debugMesh(in);
}


__global__ void debugKernel2(DeviceMesh in) {
    debugMesh(&in);
}