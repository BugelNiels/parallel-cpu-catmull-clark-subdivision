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
  ui->h0LabelNum->setNum(baseMesh->getNumHalfEdges());
  ui->f0LabelNum->setNum(baseMesh->getNumFaces());
  ui->e0LabelNum->setNum(baseMesh->getNumEdges());
  ui->v0LabelNum->setNum(baseMesh->getNumVerts());

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
  ui->hdLabelNum->setNum(currentMesh->getNumHalfEdges());
  ui->fdLabelNum->setNum(currentMesh->getNumFaces());
  ui->edLabelNum->setNum(currentMesh->getNumEdges());
  ui->vdLabelNum->setNum(currentMesh->getNumVerts());
  ui->timeLabel->setNum(milsecs);
  ui->MainDisplay->settings.uniformUpdateRequired = true;
  updateBuffers();
  ui->MainDisplay->update();
}

void MainWindow::singleSubdivisionStep(int k) {
  QuadMesh* newMesh = new QuadMesh();
  QElapsedTimer timer;
  timer.start();
  currentMesh->subdivideCatmullClark(*newMesh);
  /* Display info to user */
  long long time = timer.nsecsElapsed();
  qDebug() << "Subdivision time at " << k << time / 1000000.0 << "milliseconds";
  if (currentMesh != baseMesh) {
    delete currentMesh;
  }
  currentMesh = newMesh;
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
      "bigguy.obj");
}

void MainWindow::on_showNormalsCheckBox_toggled(bool checked) {
  ui->MainDisplay->settings.showNormals = checked;
  ui->MainDisplay->settings.uniformUpdateRequired = true;
  ui->MainDisplay->update();
}
