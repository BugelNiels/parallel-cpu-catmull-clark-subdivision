#include <stdlib.h>
#include <stdio.h>

#include "devicemesh.cuh"
#include "../util/util.cuh"

DeviceMesh createEmptyCopyOnDevice(Mesh* mesh) {
    return initEmptyDeviceMesh(mesh->numVerts, mesh->numHalfEdges, mesh->numFaces, mesh->numEdges);
}

void setDevicePointerValue(int** loc, int val) {
    cudaError_t cuda_ret;
    cuda_ret = cudaMalloc((void**)loc, sizeof(int));
    cudaErrCheck(cuda_ret, "Unable to allocate device int pointer val");
    cuda_ret = cudaMemcpy(*loc, &val, sizeof(int), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy val to device pointer");
}

DeviceMesh initEmptyDeviceMesh(int numVerts, int numHalfEdges, int numFaces, int numEdges) {
	DeviceMesh mesh = {};
    printf("\n--\nCopying %d %d %d %d\n --\n", numVerts, numHalfEdges, numFaces, numEdges);
    setDevicePointerValue(&mesh.numVerts, numVerts);
    setDevicePointerValue(&mesh.numHalfEdges, numHalfEdges);
    setDevicePointerValue(&mesh.numFaces, numFaces);
    setDevicePointerValue(&mesh.numEdges, numEdges);
	return mesh;
} 

void freeDeviceMesh(DeviceMesh* mesh) {
    cudaFree(mesh->xCoords);
    cudaFree(mesh->yCoords);
    cudaFree(mesh->zCoords);
    cudaFree(mesh->twins);
    cudaFree(mesh->nexts);
    cudaFree(mesh->prevs);
    cudaFree(mesh->verts);
    cudaFree(mesh->edges);
    cudaFree(mesh->faces);
}
