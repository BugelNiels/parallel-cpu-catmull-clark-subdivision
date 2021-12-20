#ifndef MAINVIEW_H
#define MAINVIEW_H

#include <QMouseEvent>
#include <QOpenGLDebugLogger>
#include <QOpenGLFunctions_4_1_Core>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>

#include "mesh.h"
#include "meshrenderer.h"

/**
 * @brief The MainView class represents the main view of the UI
 */
class MainView : public QOpenGLWidget, protected QOpenGLFunctions_4_1_Core {
  Q_OBJECT

 public:
  MainView(QWidget* Parent = nullptr);
  ~MainView();

  void updateMatrices();
  void updateUniforms();
  void updateBuffers(Mesh& currentMesh);
  void updateBuffers(Mesh& mrMesh, Mesh& tessrMesh);

 protected:
  void initializeGL();
  void resizeGL(int newWidth, int newHeight);
  void paintGL();

  void mouseMoveEvent(QMouseEvent* event);
  void mousePressEvent(QMouseEvent* event);
  void wheelEvent(QWheelEvent* event);
  void keyPressEvent(QKeyEvent* event);

 private:
  QOpenGLDebugLogger debugLogger;

  QVector2D toNormalizedScreenCoordinates(int x, int y);

  // for zoom
  float scale;
  // for handling rotation
  QVector3D oldVec;
  QQuaternion rotationQuaternion;
  bool dragging;

  MeshRenderer mr;

  Settings settings;

  // we make mainwindow a friend so it can access settings
  friend class MainWindow;

 private slots:
  void onMessageLogged(QOpenGLDebugMessage Message);
};

#endif  // MAINVIEW_H
