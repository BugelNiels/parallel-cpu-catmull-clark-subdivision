#ifndef DEVICE_COMMUNICATION_CUH
#define DEVICE_COMMUNICATION_CUH

#include "mesh.cuh"

void allocateDeviceMemory(Mesh* deviceMesh, int m, int n);
void reallocHostMemory(Mesh* hostMesh, Mesh* deviceMesh);
void copyHostToDeviceMesh(Mesh* hostMesh, Mesh* deviceMesh);
void copyDeviceMeshToHostMesh(Mesh* hostMesh, Mesh* deviceMesh);

#endif // DEVICE_COMMUNICATION_CUH