#include "mainwindow.h"

#include <QDebug>
#include <QElapsedTimer>

#include "meshinitializer.h"
#include "quadmesh.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
  qDebug() << "✓✓ MainWindow constructor";
  ui->setupUi(this);
  ui->applySubdivisionButton->setEnabled(false);
  ui->generalGroupBox->setEnabled(false);
}

MainWindow::~MainWindow() {
  qDebug() << "✗✗ MainWindow destructor";
  delete ui;
}

void MainWindow::importOBJ(QString filename) {
  qDebug() << filename;
  OBJFile newModel = OBJFile(filename);
  MeshInitializer initializer(&newModel);
  baseMesh = initializer.constructHalfEdgeMesh();
  currentMesh = baseMesh;

  subdivisionLevel = 0;

  updateBuffers();
  ui->MainDisplay->settings.modelLoaded = true;
  ui->generalGroupBox->setEnabled(true);
  ui->MainDisplay->update();
}

void MainWindow::on_ImportOBJ_clicked() {
  importOBJ(QFileDialog::getOpenFileName(this, "Import OBJ File", "models/",
                                         tr("Obj Files (*.obj)")));
}

void MainWindow::on_SubdivSteps_valueChanged(int value) {
  subdivisionLevel = value;
  if (ui->MainDisplay->settings.requireApply) {
    return;
  }
  subdivide();
}

void MainWindow::on_wireframeCheckBox_toggled(bool checked) {
  ui->MainDisplay->settings.wireframeMode = checked;
  ui->MainDisplay->update();
}

void MainWindow::subdivide() {
  currentMesh = baseMesh;

  QElapsedTimer timer;
  timer.start();
  for (unsigned short k = 0; k < subdivisionLevel; k++) {
    singleSubdivisionStep(k);
  }
  /* Display info to user */
  long long time = timer.nsecsElapsed();
  double milsecs = time / 1000000.0;
  qDebug() << "Total time elapsed for " << subdivisionLevel << ":" << milsecs
           << "milliseconds";
  ui->timeLabel->setNum(milsecs);
  ui->MainDisplay->settings.uniformUpdateRequired = true;
  updateBuffers();
  ui->MainDisplay->update();
}

void MainWindow::singleSubdivisionStep(int k) {
  QElapsedTimer timer;
  timer.start();

  QuadMesh* newMesh = new QuadMesh();
  currentMesh->subdivideCatmullClark(*newMesh);
  currentMesh = newMesh;

  /* Display info to user */
  long long time = timer.nsecsElapsed();
  qDebug() << "Subdivision time at " << k << time / 1000000.0 << "milliseconds";
}

void MainWindow::updateBuffers() {
  ui->MainDisplay->updateBuffers(*currentMesh);
}

void MainWindow::on_applySubdivisionButton_pressed() { subdivide(); }

void MainWindow::on_requireApplyCheckBox_toggled(bool checked) {
  ui->MainDisplay->settings.requireApply = checked;
  ui->applySubdivisionButton->setEnabled(checked);
}

void MainWindow::on_importDefaultButton_pressed() {
  importOBJ(
      "/home/niels/Documents/parallel-catmull-clark-subdivision/src/models/"
      "QuadMeshes/Brick.obj");
}

void MainWindow::on_showNormalsCheckBox_toggled(bool checked) {
  ui->MainDisplay->settings.showNormals = checked;
  ui->MainDisplay->settings.uniformUpdateRequired = true;
  ui->MainDisplay->update();
}
