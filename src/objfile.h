#ifndef OBJFILE_H
#define OBJFILE_H

#include <QString>
#include <QVector2D>
#include <QVector3D>
#include <QVector>

/**
 * @brief The OBJFile class is used for storing info from the .obj files
 */
class OBJFile {
 public:
  OBJFile(QString fileName);
  ~OBJFile();

  QVector<QVector3D> vertexCoords;
  QVector<QVector2D> textureCoords;
  QVector<QVector3D> vertexNormals;
  QVector<QVector<int>> faceCoordInd;
  QVector<QVector<int>> faceTexInd;
  QVector<QVector<int>> faceNormalInd;

  bool isQuad;
};

#endif  // OBJFILE_H
