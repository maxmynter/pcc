/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
#include <cmath>
void correlate(int ny, int nx, const float *data, float *result) {
  // Pre-Calculate the means of all cols
  double *means = new double[ny];
  for (int i = 0; i < ny; i++) {
    means[i] = 0.0;
    for (int k = 0; k < nx; k++) {
      means[i] += data[k + i * nx];
    }
    means[i] /= nx;
  }

  // Pre-Calc the differences to mean and their square
  double *diffs = new double[ny * nx];
  double *sq_sums = new double[ny];
  for (int i = 0; i < ny; i++) {
    sq_sums[i] = 0.0;
    for (int k = 0; k < nx; k++) {
      diffs[k + i * nx] = data[k + i * nx] - means[i];
      // Use multiplication instead of std::pow because the former is faster for
      // only squaring
      sq_sums[i] += (diffs[k + i * nx]) * (diffs[k + i * nx]);
    }
  }

  // Calculate the correlations
  for (int i = 0; i < ny; i++) {
    for (int j = 0; j <= i; j++) {
      double numerator = 0.0;
      for (int k = 0; k < nx; k++) {
        numerator += diffs[k + i * nx] * diffs[k + j * nx];
      }
      double denominator = std::sqrt(sq_sums[i] * sq_sums[j]);
      result[i + j * ny] = numerator / denominator;
    }
  }

  delete[] means;
  delete[] diffs;
  delete[] sq_sums;
}
