#include <QApplication>
#include <QSurfaceFormat>

#include "mainwindow.h"
#include "omp.h"

int main(int argc, char *argv[]) {
  omp_set_dynamic(false);
  omp_set_num_threads(8);
  QApplication a(argc, argv);

  QSurfaceFormat glFormat;
  glFormat.setProfile(QSurfaceFormat::CoreProfile);
  glFormat.setVersion(4, 1);
  glFormat.setOption(QSurfaceFormat::DebugContext);
  QSurfaceFormat::setDefaultFormat(glFormat);

  MainWindow w;
  w.show();

  return a.exec();
}
