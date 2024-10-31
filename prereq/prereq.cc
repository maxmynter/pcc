struct Result {
  float avg[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- horizontal position: 0 <= x0 < x1 <= nx
- vertical position: 0 <= y0 < y1 <= ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
- output: avg[c]
*/
Result calculate(int ny, int nx, const float *data, int y0, int x0, int y1,
                 int x1) {
  Result result{{0.0f, 0.0f, 0.0f}};

  const int stride = 3 * nx;
  double sums[3] = {0.0, 0.0, 0.0};

  for (int y = y0; y < y1; y++) {
    const int row_offset = stride * y;
    for (int x = x0; x < x1; x++) {
      const int pixel_offset = row_offset + 3 * x;
      sums[0] += data[pixel_offset];
      sums[1] += data[pixel_offset + 1];
      sums[2] += data[pixel_offset + 2];
    }
  }
  const double pixel_count = (x1 - x0) * (y1 - y0);
  for (int c = 0; c < 3; c++) {
    result.avg[c] = static_cast<float>(sums[c] / pixel_count);
  }

  return result;
}
