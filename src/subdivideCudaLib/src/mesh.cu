Mesh makeEmptyCopy(Mesh* mesh) {
	Mesh copy;
	copy.numVerts = mesh->numVerts;
	copy.numHalfEdges = mesh->numHalfEdges;
	copy.numEdges = mesh->numEdges;
	copy.numFaces = mesh->numFaces;
	return copy;
}

Mesh initMesh(int numVerts, int numHalfEdges, int numFaces, int numEdges) {
	Mesh mesh;
	mesh.numVerts = numVerts;
	mesh.numHalfEdges = numHalfEdges;
	mesh.numEdges = numEdges;
	mesh.numFaces = numFaces;
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