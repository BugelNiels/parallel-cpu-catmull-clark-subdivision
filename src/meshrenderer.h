#ifndef MESHRENDERER_H
#define MESHRENDERER_H

#include <QOpenGLShaderProgram>

#include "mesh.h"
#include "renderer.h"

/**
 * @brief The MeshRenderer class is responsible for rendering a mesh
 */
class MeshRenderer : public Renderer {
 public:
  MeshRenderer();
  ~MeshRenderer();

  void init(QOpenGLFunctions_4_1_Core* f, Settings* s);

  void initShaders();
  void initBuffers();

  void updateUniforms();

  void updateBuffers(Mesh& m);
  void draw();

 private:
  GLuint vao;
  GLuint meshCoordsBO, meshNormalsBO, meshIndexBO;
  int meshIBOSize;
  QOpenGLShaderProgram shaderProg;

  // Uniforms
  GLint uniModelViewMatrix, uniProjectionMatrix, uniNormalMatrix,
      uniShowNormals;
};

#endif  // MESHRENDERER_H
