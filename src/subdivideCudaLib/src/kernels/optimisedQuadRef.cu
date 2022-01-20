#include "quadRefinement.cuh"
#include "../util/util.cuh"

inline __device__ int next(int h) { return h % 4 == 3 ? h - 3 : h + 1; }

inline __device__ int prev(int h) { return h % 4 == 0 ? h + 3 : h - 1; }

inline __device__ int face(int h) { return h / 4; }

__global__ void optimisedSubdivide(DeviceMesh* in, DeviceMesh* out, int v0) {
    __shared__ float facePointsX[FACES_PER_BLOCK];
    __shared__ float facePointsY[FACES_PER_BLOCK];
    __shared__ float facePointsZ[FACES_PER_BLOCK];

    int vd = in->numVerts;
    int fd = in->numFaces;
    int ed = in->numEdges;

    int ti = threadIdx.x / 4;
    int t2 = threadIdx.x % 4;

    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int h = start; h < in->numHalfEdges; h += stride) {
        // not all threads in the warp execute this, but it should eliminate the need for thread sync
        if(t2 == 0) {
            // reset shared memory
            facePointsX[ti] = 0;
            facePointsY[ti] = 0;
            facePointsZ[ti] = 0;
        }
        // edge refinement
        int hp = prev(h);
        int he = in->edges[h];
        int v = in->verts[h];
        int ht = in->twins[h];

        out->twins[4 * h] = ht < 0 ? -1 : 4 * next(ht) + 3;
        out->twins[4 * h + 1] = 4 * next(h) + 2;
        out->twins[4 * h + 2] = 4 * hp + 1;
        out->twins[4 * h + 3] = 4 * in->twins[hp];

        out->verts[4 * h] = v;
        out->verts[4 * h + 1] = vd + fd + he;
        out->verts[4 * h + 2] = vd + face(h);
        out->verts[4 * h + 3] = vd + fd + in->edges[hp];

        out->edges[4 * h] = h > ht ? 2 * he : 2 * he + 1;
        out->edges[4 * h + 1] = 2 * ed + h;
        out->edges[4 * h + 2] = 2 * ed + hp;
        out->edges[4 * h + 3] = hp > in->twins[hp] ? 2 * in->edges[hp] + 1 : 2 * in->edges[hp];
        
        // face points
        float invX = in->xCoords[v];
        float invY = in->yCoords[v];
        float invZ = in->zCoords[v];

        atomicAdd(&facePointsX[ti], invX / 4.0f);
        atomicAdd(&facePointsY[ti], invY / 4.0f);
        atomicAdd(&facePointsZ[ti], invZ / 4.0f);

        // edge points
        float x, y, z;
        // boundary

        int k = in->verts[next(h)];
        float edgex = (invX + in->xCoords[k]) / 2.0f;
        float edgey = (invY + in->yCoords[k]) / 2.0f;
        float edgez = (invZ + in->zCoords[k]) / 2.0f;
        
        if(ht < 0) {
            x = edgex;
            y = edgey;
            z = edgez;      
        } else {
            // average the vertex of this vertex and the face point
            x = (invX + facePointsX[ti]) / 4.0f;
            y = (invY + facePointsY[ti]) / 4.0f;
            z = (invZ + facePointsZ[ti]) / 4.0f;
        }    
        int j = vd + fd + he;
        atomicAdd(&out->xCoords[j], x);
        atomicAdd(&out->yCoords[j], y);
        atomicAdd(&out->zCoords[j], z);

        // this is pretty awesome trick yes
        float n = v >= v0 ? 4 : valence(h, in);
        if(ht < 0) {
            out->xCoords[v] = invX;
            out->yCoords[v] = invY;
            out->zCoords[v] = invZ;
        } else if (n >= 0) {
            float n2 = n * n;
            x = (2 * edgex + facePointsX[ti] + (n - 3) * invX) / n2;
            y = (2 * edgey + facePointsY[ti] + (n - 3) * invY) / n2;
            z = (2 * edgez + facePointsZ[ti] + (n - 3) * invZ) / n2;
            atomicAdd(&out->xCoords[v], x);
            atomicAdd(&out->yCoords[v], y);
            atomicAdd(&out->zCoords[v], z);
        }
        
        if(t2 == 0) {
            int ind = vd + face(h);
            out->xCoords[ind] = facePointsX[ti];
            out->yCoords[ind] = facePointsY[ti];
            out->zCoords[ind] = facePointsZ[ti];
        }
    }  
}
