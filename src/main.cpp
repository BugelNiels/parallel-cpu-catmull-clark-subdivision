#include <QApplication>
#include <QSurfaceFormat>
#include <iostream>

#include "mainwindow.h"
#include "meshinitializer.h"
#include "objfile.h"
#include "omp.h"
#include "quadmesh.h"

/**
 * @brief getCmdOption Retrieves the string after the provided flag
 * @param begin Start of the array to be searched in
 * @param end End of the array to be searched in
 * @param option Flag
 * @return The string after the provided flag if it exists, null otherwise
 */
char* getCmdOption(char** begin, char** end, const std::string& option) {
  char** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    return *itr;
  }
  return nullptr;
}

/**
 * @brief cmdOptionExists Finds whether a command flag exists
 * @param begin Start of the array to be searched in
 * @param end End of the array to be searched in
 * @param option The flag
 * @return True if the flag is found, false otherwise
 */
bool cmdOptionExists(char** begin, char** end, const std::string& option) {
  return std::find(begin, end, option) != end;
}

int startGUI(int argc, char* argv[]) {
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

int main(int argc, char* argv[]) {
  bool usingGUI = true;

  if (cmdOptionExists(argv, argv + argc, "-c")) {
    usingGUI = false;
  }

  if (usingGUI) {
    return startGUI(argc, argv);
  }

  char* filename = getCmdOption(argv, argv + argc, "-f");
  char* subdivLevel = getCmdOption(argv, argv + argc, "-l");
  char* numThreads = getCmdOption(argv, argv + argc, "-t");

  if (filename == nullptr || subdivLevel == nullptr || numThreads == nullptr) {
    std::string executable = argv[0];
    std::cerr << "Usage: "
              << executable.substr(executable.find_last_of('/') + 1)
              << "\n or \n"
              << executable.substr(executable.find_last_of('/') + 1)
              << "-c -t <num threads> -f <filename> -l <subdivision level>\n";
    exit(EXIT_FAILURE);
  }
  omp_set_num_threads(atoi(numThreads));
  OBJFile newModel = OBJFile(QString(filename));
  MeshInitializer initializer(&newModel);
  Mesh* baseMesh = initializer.constructHalfEdgeMesh();
  Subdivider subdivider(baseMesh);
  subdivider.subdivide(atoi(subdivLevel));

  exit(EXIT_SUCCESS);
}
