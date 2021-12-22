#version 410
// Compute shader
// need twins and edge array
// need vd, fd and ed input variables

// output twins, verts and edges

int next(int h) {
  if (h < 0) {
    return -1;
  }
  return h % 4 == 3 ? h - 3 : h + 1;
}

int prev(int h) { return h % 4 == 0 ? h + 3 : h - 1; }

// executed for every half edge
void main() {
  int hp = prev(h);

  // For boundaries
  mesh.twins[4 * h] = 4 * next(twin(h)) + 3;
  mesh.twins[4 * h + 1] = 4 * next(h) + 2;
  mesh.twins[4 * h + 2] = 4 * prev(h) + 1;
  mesh.twins[4 * h + 3] = 4 * twin(hp);

  mesh.verts[4 * h] = vert(h);
  mesh.verts[4 * h + 1] = vd + fd + edge(h);
  mesh.verts[4 * h + 2] = vd + face(h);
  mesh.verts[4 * h + 3] = vd + fd + edge(hp);

  mesh.edges[4 * h] = h > twin(h) ? 2 * edge(h) : 2 * edge(h) + 1;
  mesh.edges[4 * h + 1] = 2 * ed + h;
  mesh.edges[4 * h + 2] = 2 * ed + hp;
  mesh.edges[4 * h + 3] = hp > twin(hp) ? 2 * edge(hp) + 1 : 2 * edge(hp);
}
