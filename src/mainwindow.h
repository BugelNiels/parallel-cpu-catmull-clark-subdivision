#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QFileDialog>
#include <QMainWindow>

#include "mesh.h"
#include "objfile.h"

namespace Ui {
class MainWindow;
}

/**
 * @brief The MainWindow class represents the main window
 */
class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit MainWindow(QWidget *parent = 0);
  ~MainWindow();

  QVector<Mesh> meshes;
  QVector<Mesh> limitMeshes;
  void importOBJ(QString filename);

 private slots:
  void on_ImportOBJ_clicked();
  void on_SubdivSteps_valueChanged(int value);
  void on_wireframeCheckBox_toggled(bool checked);

  void on_applySubdivisionButton_pressed();

  void on_requireApplyCheckBox_toggled(bool checked);

 private:
  void subdivide();
  void updateBuffers();
  Ui::MainWindow *ui;
  int meshIndex;
};

#endif  // MAINWINDOW_H
