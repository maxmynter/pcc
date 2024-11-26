/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
#include <cmath>
#include <vector>

typedef double double4_t __attribute__((vector_size(4 * sizeof(double))));
constexpr float infty = std::numeric_limits<float>::infinity();
constexpr double4_t d4infty{
    infty,
    infty,
    infty,
    infty,
};

static inline double4_t mul4(double4_t x, double4_t y) { return x * y; }

void correlate(int ny, int nx, const float *data, float *result) {
  // Elements per vector
  constexpr int nb = 4;
  int na = (nx + nb - 1) / nb;

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
  double *sq_sums = new double[ny];
  double *diffs = new double[ny * nx];
  for (int i = 0; i < ny; i++) {
    sq_sums[i] = 0.0;
    for (int k = 0; k < nx; k++) {
      diffs[k + i * nx] = data[k + i * nx] - means[i];
      // Use multiplication instead of std::pow because the former is faster for
      // only squaring
      sq_sums[i] += (diffs[k + i * nx]) * (diffs[k + i * nx]);
    }
  }

  std::vector<double4_t> vdiffs(ny * na);
#pragma omp parallel for
  for (int j = 0; j < ny; ++j) {
    for (int ka = 0; ka < na; ++ka) {
      for (int kb = 0; kb < nb; ++kb) {
        int i = ka * nb + kb;
        vdiffs[na * j + ka][kb] = i < nx ? diffs[nx * j + i] : 0;
      }
    }
  }

  // Calculate the correlations
  for (int i = 0; i < ny; i++) {
    for (int j = 0; j <= i; j++) {
      double4_t vv = {0, 0, 0, 0};
      for (int ka = 0; ka < na; ++ka) {
        double4_t x = vdiffs[na * i + ka];
        double4_t y = vdiffs[na * j + ka];
        vv += mul4(x, y);
      }
      double numerator = 0.0;
      for (int k = 0; k < 4; k++) {
        numerator += vv[k];
      }
      double denominator = std::sqrt(sq_sums[i] * sq_sums[j]);
      result[i + j * ny] = numerator / denominator;
    }
  }

  delete[] means;
  delete[] diffs;
  delete[] sq_sums;
}
