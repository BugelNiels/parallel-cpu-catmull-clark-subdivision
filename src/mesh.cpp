#include "mesh.h"

#include "math.h"

/**
 * @brief Mesh::Mesh Creates an empty mesh
 */
Mesh::Mesh() {}

/**
 * @brief Mesh::Mesh Creates a mesh from the provided .obj file
 * @param loadedOBJFile The .obj file
 */
Mesh::Mesh(OBJFile* loadedOBJFile) {}

/**
 * @brief Mesh::~Mesh Destructor
 */
Mesh::~Mesh() {}

/**
 * @brief Mesh::extractAttributes Extracts attributes for this mesh
 * @param hideRegularPatches Whether to include regular patches or not. If set
 * to true, regular patches will not be rendered
 */
void Mesh::extractAttributes() {}
