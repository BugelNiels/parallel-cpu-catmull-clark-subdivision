#ifndef DEVICE_COMMUNICATION_CUH
#define DEVICE_COMMUNICATION_CUH

#include "mesh/mesh.cuh"
#include "mesh/deviceMesh.cuh"

void allocateDeviceMemory(DeviceMesh* deviceMesh, int m, int n, int n0, int isQuad);
void reallocHostMemory(Mesh* hostMesh, DeviceMesh* deviceMesh);
void copyHostToDeviceMesh(Mesh* hostMesh, DeviceMesh* deviceMesh, int isQuad);
void copyDeviceMeshToHostMesh(Mesh* hostMesh, DeviceMesh* deviceMesh);

#endif // DEVICE_COMMUNICATION_CUH