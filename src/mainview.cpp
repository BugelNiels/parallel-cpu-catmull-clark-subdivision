#include "mainview.h"

#include <QLoggingCategory>

#include "math.h"

/**
 * @brief MainView::MainView Creates a new mainview
 * @param Parent Parent widget
 */
MainView::MainView(QWidget* Parent) : QOpenGLWidget(Parent) {
  qDebug() << "✓✓ MainView constructor";

  scale = 1.0f;
}

/**
 * @brief MainView::~MainView Destructor
 */
MainView::~MainView() {
  qDebug() << "✗✗ MainView destructor";
  makeCurrent();
}

/**
 * @brief MainView::initializeGL Initializes the opengl functions and settings,
 * initialises the renderers and sets up the debugger.
 */
void MainView::initializeGL() {
  initializeOpenGLFunctions();
  qDebug() << ":: OpenGL initialized";

  connect(&debugLogger, SIGNAL(messageLogged(QOpenGLDebugMessage)), this,
          SLOT(onMessageLogged(QOpenGLDebugMessage)), Qt::DirectConnection);

  if (debugLogger.initialize()) {
    QLoggingCategory::setFilterRules(
        "qt.*=false\n"
        "qt.text.font.*=false");

    qDebug() << ":: Logging initialized";
    debugLogger.startLogging(QOpenGLDebugLogger::SynchronousLogging);
    debugLogger.enableMessages();
  }

  QString glVersion;
  glVersion = reinterpret_cast<const char*>(glGetString(GL_VERSION));
  qDebug() << ":: Using OpenGL" << qPrintable(glVersion);

  makeCurrent();

  // Enable depth buffer
  glEnable(GL_DEPTH_TEST);
  // Default is GL_LESS
  glDepthFunc(GL_LEQUAL);

  // grab the opengl context
  QOpenGLFunctions_4_1_Core* functions =
      this->context()->versionFunctions<QOpenGLFunctions_4_1_Core>();

  // initialize renderers here with the current context
  mr.init(functions, &settings);

  updateMatrices();
}

/**
 * @brief MainView::resizeGL Called when the window is resized. Updates the
 * projection matrix.
 * @param newWidth The new width in pixels
 * @param newHeight The new height in pixels
 */
void MainView::resizeGL(int newWidth, int newHeight) {
  qDebug() << ".. resizeGL";

  settings.dispRatio = float(newWidth) / float(newHeight);

  settings.projectionMatrix.setToIdentity();
  settings.projectionMatrix.perspective(settings.fov, settings.dispRatio, 0.1f,
                                        40.0f);
  updateMatrices();
}

/**
 * @brief MainView::updateMatrices Updates the modelview and normal matrices.
 */
void MainView::updateMatrices() {
  settings.modelViewMatrix.setToIdentity();
  settings.modelViewMatrix.translate(QVector3D(0.0, 0.0, -3.0));
  settings.modelViewMatrix.scale(scale);
  settings.modelViewMatrix.rotate(rotationQuaternion);

  settings.normalMatrix = settings.modelViewMatrix.normalMatrix();

  settings.uniformUpdateRequired = true;

  update();
}

/**
 * @brief MainView::updateBuffers Updates the buffers used for drawing in the
 * mesh and tesselation renderers with the provided mesh
 * @param currentMesh The mesh to be used for the mesh and tesselation renderer
 */
void MainView::updateBuffers(Mesh& currentMesh) {
  currentMesh.extractAttributes();
  mr.updateBuffers(currentMesh);
}

/**
 * @brief MainView::paintGL Draw call
 */
void MainView::paintGL() {
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if (settings.wireframeMode) {
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  } else {
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  }

  if (settings.modelLoaded) {
    mr.draw();

    if (settings.uniformUpdateRequired) {
      settings.uniformUpdateRequired = false;
    }
  }
}

/**
 * @brief MainView::toNormalizedScreenCoordinates Normalizes the mouse
 * coordinates to screen coordinates.
 * @param x The mouse x coordinate
 * @param y The mouse y coordinate
 * @return A vector containing the normalized x and y screen coordinates
 */
QVector2D MainView::toNormalizedScreenCoordinates(int x, int y) {
  float xRatio, yRatio;
  float xScene, yScene;

  xRatio = float(x) / float(width());
  yRatio = float(y) / float(height());

  xScene = (1 - xRatio) * -1 + xRatio * 1;
  yScene = yRatio * -1 + (1 - yRatio) * 1;

  return {xScene, yScene};
}

/**
 * @brief MainView::mouseMoveEvent Event that is called when the mouse is moved
 * @param Event The mouse event
 */
void MainView::mouseMoveEvent(QMouseEvent* Event) {
  if (Event->buttons() == Qt::LeftButton) {
    QVector2D sPos = toNormalizedScreenCoordinates(Event->x(), Event->y());
    QVector3D newVec = QVector3D(sPos.x(), sPos.y(), 0.0);

    // project onto sphere
    float sqrZ = 1.0f - QVector3D::dotProduct(newVec, newVec);
    if (sqrZ > 0) {
      newVec.setZ(sqrt(sqrZ));
    } else {
      newVec.normalize();
    }

    QVector3D v2 = newVec.normalized();
    // reset if we are starting a drag
    if (!dragging) {
      dragging = true;
      oldVec = newVec;
      return;
    }

    // calculate axis and angle
    QVector3D v1 = oldVec.normalized();
    QVector3D N = QVector3D::crossProduct(v1, v2).normalized();
    if (N.length() == 0.0f) {
      oldVec = newVec;
      return;
    }
    float angle = 180.0f / M_PI * acos(QVector3D::dotProduct(v1, v2));
    rotationQuaternion =
        QQuaternion::fromAxisAndAngle(N, angle) * rotationQuaternion;
    updateMatrices();

    // for next iteration
    oldVec = newVec;
  } else {
    // to reset drag
    dragging = false;
    oldVec = QVector3D();
  }
}

/**
 * @brief MainView::mousePressEvent Event that is called when the mouse is
 * pressed
 * @param event The mouse event
 */
void MainView::mousePressEvent(QMouseEvent* event) { setFocus(); }

/**
 * @brief MainView::wheelEvent Event that is called when the user scrolls
 * @param event The mouse event
 */
void MainView::wheelEvent(QWheelEvent* event) {
  float Phi;
  // Delta is usually 120
  Phi = 1.0f + (event->delta() / 2000.0f);

  scale = fmin(fmax(Phi * scale, 0.01f), 100.0f);
  updateMatrices();
}

/**
 * @brief MainView::keyPressEvent Event that is called when a key is pressed
 * @param event The key event
 */
void MainView::keyPressEvent(QKeyEvent* event) {
  switch (event->key()) {
    case 'Z':
      settings.wireframeMode = !settings.wireframeMode;
      update();
      break;
  }
}

// ---

/**
 * @brief MainView::onMessageLogged Helper function for logging messages
 * @param Message The message
 */
void MainView::onMessageLogged(QOpenGLDebugMessage Message) {
  qDebug() << " → Log:" << Message;
}
