#ifndef RENDERER_H
#define RENDERER_H

#include <QOpenGLFunctions_4_1_Core>

#include "settings.h"

/**
 * @brief The Renderer class represents a generic renderer. All specific
 * renderers have the functions provided in this header file.
 */
class Renderer {
 public:
  Renderer() { gl = nullptr; }
  Renderer(QOpenGLFunctions_4_1_Core *functions, Settings *settings);
  ~Renderer() {}

 protected:
  QOpenGLFunctions_4_1_Core *gl;
  Settings *settings;
};

#endif  // RENDERER_H
