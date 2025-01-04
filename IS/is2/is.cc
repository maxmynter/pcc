#include <array>
#include <limits>
#include <vector>

struct Result {
  int y0;
  int x0;
  int y1;
  int x1;
  float outer[3];
  float inner[3];
};

// Helper function to compute cumulative sums
std::vector<std::array<double, 3>> compute_cumsum(const float *data, int ny,
                                                  int nx) {
  std::vector<std::array<double, 3>> cumsum((ny + 1) * (nx + 1));

  // Initialize first row and column to 0
  for (int i = 0; i <= nx; i++) {
    cumsum[i] = {0, 0, 0};
  }
  for (int i = 0; i <= ny; i++) {
    cumsum[i * (nx + 1)] = {0, 0, 0};
  }

  // Compute cumulative sum
  for (int y = 1; y <= ny; y++) {
    for (int x = 1; x <= nx; x++) {
      for (int c = 0; c < 3; c++) {
        cumsum[y * (nx + 1) + x][c] =
            cumsum[(y - 1) * (nx + 1) + x][c] +
            cumsum[y * (nx + 1) + (x - 1)][c] -
            cumsum[(y - 1) * (nx + 1) + (x - 1)][c] +
            static_cast<double>(data[c + 3 * (x - 1) + 3 * nx * (y - 1)]);
      }
    }
  }
  return cumsum;
}

// Helper function to get sum of a rectangle
std::array<double, 3>
get_rect_sum(const std::vector<std::array<double, 3>> &cumsum, int x0, int y0,
             int x1, int y1, int nx) {
  std::array<double, 3> sum;
  for (int c = 0; c < 3; c++) {
    sum[c] = cumsum[y1 * (nx + 1) + x1][c] - cumsum[y1 * (nx + 1) + x0][c] -
             cumsum[y0 * (nx + 1) + x1][c] + cumsum[y0 * (nx + 1) + x0][c];
  }
  return sum;
}

Result segment(int ny, int nx, const float *data) {
  Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
  double best_error = std::numeric_limits<double>::infinity();

  // Precompute cumulative sums
  auto cumsum = compute_cumsum(data, ny, nx);

  // Get total sums
  auto total = get_rect_sum(cumsum, 0, 0, nx, ny, nx);
  int n_total = nx * ny;

  // Try all possible rectangles
  for (int y0 = 0; y0 < ny; y0++) {
    for (int y1 = y0 + 1; y1 <= ny; y1++) {
      for (int x0 = 0; x0 < nx; x0++) {
        for (int x1 = x0 + 1; x1 <= nx; x1++) {
          // Get inner rectangle sums
          auto inner = get_rect_sum(cumsum, x0, y0, x1, y1, nx);
          std::array<double, 3> outer;

          int n_inner = (x1 - x0) * (y1 - y0);
          int n_outer = n_total - n_inner;

          // Compute means
          std::array<double, 3> inner_means, outer_means;
          double total_error = 0;

          for (int c = 0; c < 3; c++) {
            outer[c] = total[c] - inner[c];
            inner_means[c] = inner[c] / n_inner;
            outer_means[c] = outer[c] / n_outer;
          }

          // Compute errors efficiently using sum of squares formula
          for (int c = 0; c < 3; c++) {
            double inner_sum_sq = 0;
            double outer_sum_sq = 0;

            for (int y = y0; y < y1; y++) {
              for (int x = x0; x < x1; x++) {
                double val = data[c + 3 * x + 3 * nx * y];
                inner_sum_sq += val * val;
              }
            }

            double inner_error = inner_sum_sq - 2 * inner_means[c] * inner[c] +
                                 n_inner * inner_means[c] * inner_means[c];

            for (int y = 0; y < ny; y++) {
              for (int x = 0; x < nx; x++) {
                if (x >= x0 && x < x1 && y >= y0 && y < y1)
                  continue;
                double val = data[c + 3 * x + 3 * nx * y];
                outer_sum_sq += val * val;
              }
            }

            double outer_error = outer_sum_sq - 2 * outer_means[c] * outer[c] +
                                 n_outer * outer_means[c] * outer_means[c];

            total_error += inner_error + outer_error;
          }

          if (total_error < best_error) {
            best_error = total_error;
            result.x0 = x0;
            result.y0 = y0;
            result.x1 = x1;
            result.y1 = y1;
            for (int c = 0; c < 3; c++) {
              result.inner[c] = static_cast<float>(inner_means[c]);
              result.outer[c] = static_cast<float>(outer_means[c]);
            }
          }
        }
      }
    }
  }

  return result;
}
