#include "settings.h"

/**
 * @brief Settings::Settings Creates settings with some default values
 */
Settings::Settings() {
  modelLoaded = false;
  wireframeMode = false;
  uniformUpdateRequired = true;

  rotAngle = 0.0;
  dispRatio = 16.0f / 9.0f;
  fov = 120.0;

  requireApply = false;
}
