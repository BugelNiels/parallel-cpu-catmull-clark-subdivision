#include <stdlib.h>
#include <stdio.h>

#include "mesh.cuh"

Mesh makeEmptyCopy(Mesh* mesh) {
    return initMesh(mesh->numVerts, mesh->numHalfEdges, mesh->numFaces, mesh->numEdges);
}

Mesh initMesh(int numVerts, int numHalfEdges, int numFaces, int numEdges) {
	Mesh mesh;
	mesh.numVerts = numVerts;
	mesh.numHalfEdges = numHalfEdges;
	mesh.numEdges = numEdges;
	mesh.numFaces = numFaces;
    mesh.nexts = NULL;
    mesh.prevs = NULL;
    mesh.faces = NULL;
    mesh.twins = NULL;
    mesh.edges = NULL;
    mesh.verts = NULL;
	return mesh;
} 

void freeDeviceMesh(Mesh* mesh) {
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

void freeMesh(Mesh* mesh) {
    free(mesh->xCoords);
    free(mesh->yCoords);
    free(mesh->zCoords);
    free(mesh->twins);
    free(mesh->nexts);
    free(mesh->prevs);
    free(mesh->verts);
    free(mesh->edges);
    free(mesh->faces);
}

void toObjFile(Mesh* mesh) {
    printf("Writing mesh to file..\n");
    // TODO: add name of object file
    FILE *objFile = fopen("result.obj", "w");
    if (objFile == NULL)
    {
        printf("Error opening or creating .obj file!\n");
        exit(1);
    }
    // print vertices
    for(int v = 0; v < mesh->numVerts; v++) {
        fprintf(objFile, "v %.6lf %.6lf %.6lf\n", mesh->xCoords[v], mesh->yCoords[v], mesh->zCoords[v]);
    }
    fprintf(objFile, "# Numfaces: %d\n\n", mesh->numFaces); 
    // list of face indices
    for(int f = 0; f < mesh->numFaces; f++) {
        fprintf(objFile, "f");
        for(int v = 0; v < 4; v++) {
            // indices in .obj start at 1
            fprintf(objFile, " %d", mesh->verts[f*4 + v] + 1);
        }
        fprintf(objFile, "\n");
    }
    fclose(objFile);
}
