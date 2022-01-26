#ifndef SETTINGS_H
#define SETTINGS_H

#include <QMatrix4x4>

/**
 * @brief The Settings class contains all the settings values. These are
 * generally set in the UI.
 */
class Settings {
 public:
  Settings();

  bool modelLoaded;
  bool wireframeMode;

  float fov;
  float dispRatio;
  float rotAngle;

  bool uniformUpdateRequired;

  QMatrix4x4 modelViewMatrix, projectionMatrix;
  QMatrix3x3 normalMatrix;

  bool requireApply;
  bool showNormals;
};

#endif  // SETTINGS_H
