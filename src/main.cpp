#include <QApplication>
#include <QSurfaceFormat>

#include "mainwindow.h"

int main(int argc, char *argv[]) {
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
