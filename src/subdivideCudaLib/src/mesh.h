typedef struct Mesh {
	// these have length numVerts
	float* xCoords;
	float* yCoords;
	float* zCoords;

	// these all have length numHalfEdges
	int* twins;
	int* nexts;
	int* prevs;
	int* verts;
	int* edges;
	int* faces;

	int numHalfEdges;
  	int numEdges;
  	int numFaces;
  	int numVerts;
} Mesh;