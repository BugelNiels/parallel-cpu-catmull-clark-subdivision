#ifndef DEVICE_COMMUNICATION_CUH
#define DEVICE_COMMUNICATION_CUH

#include "mesh/mesh.cuh"

void allocateDeviceMemory(Mesh* deviceMesh, int m, int n, int n0, int isQuad);
void reallocHostMemory(Mesh* hostMesh, Mesh* deviceMesh);
void copyHostToDeviceMesh(Mesh* hostMesh, Mesh* deviceMesh, int isQuad);
void copyDeviceMeshToHostMesh(Mesh* hostMesh, Mesh* deviceMesh);

#endif // DEVICE_COMMUNICATION_CUH