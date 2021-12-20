#include "meshrenderer.h"

/**
 * @brief MeshRenderer::MeshRenderer Creates a new mesh renderer
 */
MeshRenderer::MeshRenderer() { meshIBOSize = 0; }

/**
 * @brief MeshRenderer::~MeshRenderer Destructor
 */
MeshRenderer::~MeshRenderer() {
  gl->glDeleteVertexArrays(1, &vao);

  gl->glDeleteBuffers(1, &meshCoordsBO);
  gl->glDeleteBuffers(1, &meshNormalsBO);
  gl->glDeleteBuffers(1, &meshIndexBO);
}

/**
 * @brief MeshRenderer::init Initialises the opengl functions, settings and
 * initializes the shaders, buffers and texture
 * @param f The openglfunctions
 * @param s The settings
 */
void MeshRenderer::init(QOpenGLFunctions_4_1_Core* f, Settings* s) {
  gl = f;
  settings = s;

  initShaders();
  initBuffers();
}

/**
 * @brief MeshRenderer::initShaders Initialises the shader and its uniform
 * locations
 */
void MeshRenderer::initShaders() {
  shaderProg.create();
  shaderProg.addShaderFromSourceFile(QOpenGLShader::Vertex,
                                     ":/shaders/default.vert");
  shaderProg.addShaderFromSourceFile(QOpenGLShader::Fragment,
                                     ":/shaders/default.frag");

  shaderProg.link();

  uniModelViewMatrix =
      gl->glGetUniformLocation(shaderProg.programId(), "modelviewmatrix");
  uniProjectionMatrix =
      gl->glGetUniformLocation(shaderProg.programId(), "projectionmatrix");
  uniNormalMatrix =
      gl->glGetUniformLocation(shaderProg.programId(), "normalmatrix");
  uniShowNormals =
      gl->glGetUniformLocation(shaderProg.programId(), "shownormals");
}

/**
 * @brief MeshRenderer::initBuffers Initialises the buffers for coordinates,
 * normals, and vertex indicies
 */
void MeshRenderer::initBuffers() {
  gl->glGenVertexArrays(1, &vao);
  gl->glBindVertexArray(vao);

  gl->glGenBuffers(1, &meshCoordsBO);
  gl->glBindBuffer(GL_ARRAY_BUFFER, meshCoordsBO);
  gl->glEnableVertexAttribArray(0);
  gl->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

  gl->glGenBuffers(1, &meshNormalsBO);
  gl->glBindBuffer(GL_ARRAY_BUFFER, meshNormalsBO);
  gl->glEnableVertexAttribArray(1);
  gl->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

  gl->glGenBuffers(1, &meshIndexBO);
  gl->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshIndexBO);

  gl->glBindVertexArray(0);
}

/**
 * @brief TessRenderer::updateBuffers Updates the buffers based on the provided
 * mesh
 * @param m The mesh with which to update the buffer with
 */
void MeshRenderer::updateBuffers(Mesh& currentMesh) {
  qDebug() << ".. updateBuffers";

  // gather attributes for current mesh
  QVector<QVector3D>& vertexCoords = currentMesh.getVertexCoords();
  QVector<QVector3D>& vertexNormals = currentMesh.getVertexNorms();
  QVector<unsigned int>& polyIndices = currentMesh.getPolyIndices();

  gl->glBindBuffer(GL_ARRAY_BUFFER, meshCoordsBO);
  gl->glBufferData(GL_ARRAY_BUFFER, sizeof(QVector3D) * vertexCoords.size(),
                   vertexCoords.data(), GL_DYNAMIC_DRAW);

  qDebug() << " → Updated meshCoordsBO";

  gl->glBindBuffer(GL_ARRAY_BUFFER, meshNormalsBO);
  gl->glBufferData(GL_ARRAY_BUFFER, sizeof(QVector3D) * vertexNormals.size(),
                   vertexNormals.data(), GL_DYNAMIC_DRAW);

  qDebug() << " → Updated meshNormalsBO";

  gl->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshIndexBO);
  gl->glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                   sizeof(unsigned int) * polyIndices.size(),
                   polyIndices.data(), GL_DYNAMIC_DRAW);

  qDebug() << " → Updated meshIndexBO";

  meshIBOSize = polyIndices.size();
}

/**
 * @brief MeshRenderer::updateUniforms Updates the uniforms for the current
 * shader
 */
void MeshRenderer::updateUniforms() {
  gl->glUniformMatrix4fv(uniModelViewMatrix, 1, false,
                         settings->modelViewMatrix.data());
  gl->glUniformMatrix4fv(uniProjectionMatrix, 1, false,
                         settings->projectionMatrix.data());
  gl->glUniformMatrix3fv(uniNormalMatrix, 1, false,
                         settings->normalMatrix.data());
}

/**
 * @brief MeshRenderer::draw Draw call
 */
void MeshRenderer::draw() {
  shaderProg.bind();

  if (settings->uniformUpdateRequired) {
    updateUniforms();
  }

  // enable primitive restart
  gl->glEnable(GL_PRIMITIVE_RESTART);
  gl->glPrimitiveRestartIndex(INT_MAX);

  gl->glBindVertexArray(vao);

  if (settings->wireframeMode) {
    gl->glDrawElements(GL_LINE_LOOP, meshIBOSize, GL_UNSIGNED_INT, nullptr);
  } else {
    gl->glDrawElements(GL_TRIANGLE_FAN, meshIBOSize, GL_UNSIGNED_INT, nullptr);
  }

  gl->glBindVertexArray(0);

  shaderProg.release();

  // disable it again as you might want to draw something else at some point
  gl->glDisable(GL_PRIMITIVE_RESTART);
}
