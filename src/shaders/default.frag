#version 410
// Fragment shader

layout (location = 0) in vec3 vertcoords_camera_fs;
layout (location = 1) in vec3 vertnormal_camera_fs;

out vec4 fColor;

uniform bool shownormals;

/**
 * Phong shading
 */
vec3 phongShading() {
  vec3 lightpos = vec3(3.0, 0.0, 2.0);
  vec3 lightcolour = vec3(1.0);

  vec3 matcolour = vec3(0.53, 0.80, 0.87);
  vec3 matspeccolour = vec3(1.0);

  float matambientcoeff = 0.2;
  float matdiffusecoeff = 0.6;
  float matspecularcoeff = 0.4;

  vec3 normal = normalize(vertnormal_camera_fs);

  vec3 surftolight = normalize(lightpos - vertcoords_camera_fs);
  float diffusecoeff = max(0.0, dot(surftolight, normal));

  vec3 camerapos = vec3(0.0);
  vec3 surftocamera = normalize(camerapos - vertcoords_camera_fs);
  vec3 reflected = 2 * diffusecoeff * normal - surftolight;
  float specularcoeff = max(0.0, dot(reflected, surftocamera));

  vec3 compcolour = min(1.0, matambientcoeff + matdiffusecoeff * diffusecoeff) * lightcolour * matcolour;
  compcolour += matspecularcoeff * specularcoeff * lightcolour * matspeccolour;

  return compcolour;
}

/**
 * Shading for normals
 */
vec3 normalShading() {
  vec3 normal = normalize(vertnormal_camera_fs);
  return normal * 0.5 + 0.5;
}

/**
 * Fragment shader
 */
void main() {
  vec3 col;
  if(shownormals) {
    col = normalShading();
  } else {
    col = phongShading();
  }

  fColor = vec4(col, 1.0);
}
