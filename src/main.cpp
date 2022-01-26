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

/**
 * @brief startGUI Starts the GUI
 * @param argc Argyment count from the main function
 * @param argv Arguments from main function
 * @return Success code
 */
int startGUI(int argc, char* argv[]) {
  omp_set_num_threads(8);
  QApplication a(argc, argv);
  QSurfaceFormat glFormat;
  glFormat.setProfile(QSurfaceFormat::CoreProfile);
  glFormat.setVersion(4, 1);
  glFormat.setOption(QSurfaceFormat::DebugContext);
  QSurfaceFormat::setDefaultFormat(glFormat);

  MainWindow w;
  w.setWindowTitle("Parallel CPU Catmull-Clark Subdivision");
  w.show();
  return a.exec();
}

/**
 * @brief commandlineSubdivision Performs subdivision of "subdivisionLevel"
 * levels on the provided .obj file
 * @param filePath Path of the .obj file
 * @param subdivisionLevel Number of subdivision steps to perform
 */
void commandlineSubdivision(QString filePath, int subdivisionLevel) {
  OBJFile newModel = OBJFile(filePath);
  MeshInitializer initializer(&newModel);
  Mesh* baseMesh = initializer.constructHalfEdgeMesh();
  Subdivider subdivider(baseMesh);
  std::cout << "subdividing on CPU\n";
  double milsecs = subdivider.subdivide(subdivisionLevel);
  FILE* timingsFile = fopen("timings.txt", "a");
  fprintf(timingsFile, "%lf\n", milsecs);
  std::cout << "Subdivision complete!\n";
}

/**
 * @brief main Entry point of the program. Starts either the GUI or runs from
 * the commandline based on the arguments provided.
 * @param argc Argument count
 * @param argv Arguments. Supported arguments: "-c" starts commandline, "-f"
 * path to .obj file, "-l" subdivision level and "-t" number of threads to use.
 * @return Success code
 */
int main(int argc, char* argv[]) {
  char* numThreads = getCmdOption(argv, argv + argc, "-t");
  if (numThreads == nullptr) {
    omp_set_num_threads(4);
    std::cout << "Using 4 threads\n";
  } else {
    omp_set_num_threads(atoi(numThreads));
    std::cout << "Using " << atoi(numThreads) << " threads\n";
  }

  if (!cmdOptionExists(argv, argv + argc, "-c")) {
    return startGUI(argc, argv);
  }

  char* filename = getCmdOption(argv, argv + argc, "-f");
  char* subdivLevel = getCmdOption(argv, argv + argc, "-l");

  if (filename == nullptr || subdivLevel == nullptr) {
    std::string executable = argv[0];
    std::cerr << "Usage: "
              << executable.substr(executable.find_last_of('/') + 1)
              << "\n or \n"
              << executable.substr(executable.find_last_of('/') + 1)
              << "-c -t <num threads> -f <filename> -l <subdivision level>\n";
    exit(EXIT_FAILURE);
  }
  QString filePath = "models/" + QString(filename) + ".obj";
  commandlineSubdivision(filePath, atoi(subdivLevel));
  return 0;
}
