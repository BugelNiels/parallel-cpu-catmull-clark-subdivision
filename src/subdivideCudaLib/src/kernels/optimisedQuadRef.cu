#include "quadRefinement.cuh"
#include "../util/util.cuh"

#define WARP_SIZE 32
#define FACES_PER_BLOCK (BLOCK_SIZE / 4)

// TODO: force inline?
inline __device__ int next(int h) { return h % 4 == 3 ? h - 3 : h + 1; }

inline __device__ int prev(int h) { return h % 4 == 0 ? h + 3 : h - 1; }

inline __device__ int face(int h) { return h / 4; }


__device__ void optQuadVertexPoint(int h, DeviceMesh* in, DeviceMesh* out) {
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

__global__ void optimisedSubdivide(DeviceMesh* in, DeviceMesh* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;


    // first BLOCK_SIZE / 4 are the face points
    // next come .. edge points
    __shared__ float outCoordsXFace[FACES_PER_BLOCK];
    __shared__ float outCoordsYFace[FACES_PER_BLOCK];
    __shared__ float outCoordsZFace[FACES_PER_BLOCK];

    int vd = in->numVerts;
    int fd = in->numFaces;
    int ed = in->numEdges;
    int hn = in->numHalfEdges;

    int numEdges = 2 * ed + hn;
    int numFaces = hn;
    int numHalfEdges = hn * 4;
    int numVerts = vd + fd + ed;

    for(int v = i; v < numVerts; v += stride) {
        out->xCoords[v] = 0;
        out->yCoords[v] = 0;
        out->zCoords[v] = 0;
    } 
    for(int h = i; h < hn; h += stride) {
        // might not be necessary due to warp scheduling
        __syncthreads();
        if(threadIdx.x < FACES_PER_BLOCK) {
            // reset shared memory
            outCoordsXFace[i] = 0;
            outCoordsYFace[i] = 0;
            outCoordsZFace[i] = 0;
        }
        
        // edge refinement
        int hp = prev(h);
        int he = in->edges[h];
        int v = in->verts[h];

        // For boundaries
        int ht = in->twins[h];
        int thp = in->twins[hp];
        int ehp = in->edges[hp];
        // TODO: check interleave
        out->twins[4 * h] = ht < 0 ? -1 : 4 * next(ht) + 3;
        out->twins[4 * h + 1] = 4 * next(h) + 2;
        out->twins[4 * h + 2] = 4 * hp + 1;
        out->twins[4 * h + 3] = 4 * thp;

        out->verts[4 * h] = v;
        out->verts[4 * h + 1] = vd + fd + he;
        out->verts[4 * h + 2] = vd + face(h);
        out->verts[4 * h + 3] = vd + fd + ehp;

        out->edges[4 * h] = h > ht ? 2 * he : 2 * he + 1;
        out->edges[4 * h + 1] = 2 * ed + h;
        out->edges[4 * h + 2] = 2 * ed + hp;
        out->edges[4 * h + 3] = hp > thp ? 2 * ehp + 1 : 2 * ehp;
        
        // face points
        float invX = in->xCoords[v];
        float invY = in->yCoords[v];
        float invZ = in->zCoords[v];

        int ti = threadIdx.x / 4;
        atomicAdd(&outCoordsXFace[ti], invX / 4.0f);
        atomicAdd(&outCoordsYFace[ti], invY / 4.0f);
        atomicAdd(&outCoordsZFace[ti], invZ / 4.0f);
        
        // edge points
        float x, y, z;
        // boundary
        if(ht < 0) {
            int i = in->verts[next(h)];
            x = (invX + in->xCoords[i]) / 2.0f;
            y = (invY + in->yCoords[i]) / 2.0f;
            z = (invZ + in->zCoords[i]) / 2.0f;
            // atomic add not necessary for boundaries, but likely outweighs branch divergence penalties        
        } else {
            // average the vertex of this edge and the face point
            x = (invX + outCoordsXFace[ti]) / 4.0f;
            y = (invY + outCoordsYFace[ti]) / 4.0f;
            z = (invZ + outCoordsZFace[ti]) / 4.0f;
        }    
        int j = vd + fd + he;
        atomicAdd(&out->xCoords[j], x);
        atomicAdd(&out->yCoords[j], y);
        atomicAdd(&out->zCoords[j], z);
        __syncthreads();
        if(threadIdx.x < FACES_PER_BLOCK && threadIdx.x < fd) {
            int ind = vd + threadIdx.x;
            out->xCoords[ind] = outCoordsXFace[threadIdx.x];
            out->yCoords[ind] = outCoordsYFace[threadIdx.x];
            out->zCoords[ind] = outCoordsZFace[threadIdx.x];
        }
    }  
    if(blockIdx.x == 0 && threadIdx.x == 0) {
        out->numEdges = numEdges;
        out->numFaces = numFaces;
        out->numHalfEdges = numHalfEdges;
        out->numVerts = numVerts;
    }

}

__global__ void optQuadVertexPoints(DeviceMesh* in, DeviceMesh* out) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = h; i < in->numHalfEdges; i += stride) {
        optQuadVertexPoint(i, in, out);
    }   
}