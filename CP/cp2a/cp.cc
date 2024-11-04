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
    // Unroll loop to utilize 4-fold instruction level parallelism
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    int k = 0;
    for (; k + 3 < nx; k += 4) {
      sum0 += data[k + i * nx];
      sum1 += data[k + 1 + i * nx];
      sum2 += data[k + 2 + i * nx];
      sum3 += data[k + 3 + i * nx];
    }
    double sum = sum0 + sum1 + sum2 + sum3;
    for (; k < nx; k++) {
      sum += data[k + i * nx];
    }
    means[i] = sum / nx;
  }

  // Pre-Calc the differences to mean and their square
  double *diffs = new double[ny * nx];
  double *sq_sums = new double[ny];
  for (int i = 0; i < ny; i++) {
    double sq_sum0 = 0.0, sq_sum1 = 0.0, sq_sum2 = 0.0, sq_sum3 = 0.0;
    int k = 0; // define outside such that we can process modulo elements
    const double mean_i = means[i]; // For register lookup which is faster than
                                    // getting value from memory
    for (; k + 3 < nx; k += 4) {
      double diff0 = data[k + i * nx] - mean_i;
      double diff1 = data[k + 1 + i * nx] - mean_i;
      double diff2 = data[k + 2 + i * nx] - mean_i;
      double diff3 = data[k + 3 + i * nx] - mean_i;

      diffs[k + i * nx] = diff0;
      diffs[k + 1 + i * nx] = diff1;
      diffs[k + 2 + i * nx] = diff2;
      diffs[k + 3 + i * nx] = diff3;
      // Use multiplication instead of std::pow because the former is faster for
      // only squaring
      sq_sum0 += diff0 * diff0;
      sq_sum1 += diff1 * diff1;
      sq_sum2 += diff2 * diff2;
      sq_sum3 += diff3 * diff3;
    }
    sq_sums[i] = sq_sum0 + sq_sum1 + sq_sum2 + sq_sum3;
    for (; k < nx; k++) {
      double diff = data[k + i * nx] - mean_i;
      diffs[k + i * nx] = diff;
      sq_sums[i] += diff * diff;
    }
  }

  // Calculate the correlations
  for (int i = 0; i < ny; i++) {
    const double sq_sum_i = sq_sums[i];
    for (int j = 0; j <= i; j++) {
      double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
      int k = 0;
      for (; k + 3 < nx; k += 4) {
        sum0 += diffs[k + i * nx] * diffs[k + j * nx];
        sum1 += diffs[k + 1 + i * nx] * diffs[k + 1 + j * nx];
        sum2 += diffs[k + 2 + i * nx] * diffs[k + 2 + j * nx];
        sum3 += diffs[k + 3 + i * nx] * diffs[k + 3 + j * nx];
      }

      double numerator = sum0 + sum1 + sum2 + sum3;
      for (; k < nx; k++) {
        numerator += diffs[k + i * nx] * diffs[k + j * nx];
      }
      double denominator = std::sqrt(sq_sum_i * sq_sums[j]);
      result[i + j * ny] = numerator / denominator;
    }
  }

  delete[] means;
  delete[] diffs;
  delete[] sq_sums;
}
