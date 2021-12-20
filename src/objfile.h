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
  QVector<unsigned short> faceValences;
  QVector<unsigned int> faceCoordInd;
  QVector<unsigned int> faceTexInd;
  QVector<unsigned int> faceNormalInd;
};

#endif  // OBJFILE_H
