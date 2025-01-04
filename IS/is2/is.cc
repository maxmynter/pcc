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

// Precompute sums for all possible rectangles
void compute_prefix_sums(const float *data, int ny, int nx,
                         std::vector<std::array<double, 3>> &prefix_sum,
                         std::vector<std::array<double, 3>> &prefix_sum_sq) {
  prefix_sum.resize((ny + 1) * (nx + 1));
  prefix_sum_sq.resize((ny + 1) * (nx + 1));

  // Initialize first row and column
  for (int i = 0; i <= nx; i++) {
    prefix_sum[i].fill(0);
    prefix_sum_sq[i].fill(0);
  }
  for (int i = 1; i <= ny; i++) {
    prefix_sum[i * (nx + 1)].fill(0);
    prefix_sum_sq[i * (nx + 1)].fill(0);
  }

  // Compute 2D prefix sums for both values and squared values
  for (int y = 0; y < ny; y++) {
    for (int x = 0; x < nx; x++) {
      int idx = (y + 1) * (nx + 1) + (x + 1);
      int data_idx = 3 * (x + nx * y);

      for (int c = 0; c < 3; c++) {
        double val = static_cast<double>(data[data_idx + c]);
        double sq = val * val;

        prefix_sum[idx][c] = val + prefix_sum[(y + 1) * (nx + 1) + x][c] +
                             prefix_sum[y * (nx + 1) + (x + 1)][c] -
                             prefix_sum[y * (nx + 1) + x][c];

        prefix_sum_sq[idx][c] = sq + prefix_sum_sq[(y + 1) * (nx + 1) + x][c] +
                                prefix_sum_sq[y * (nx + 1) + (x + 1)][c] -
                                prefix_sum_sq[y * (nx + 1) + x][c];
      }
    }
  }
}

// Get sum and sum of squares for a rectangle using prefix sums
inline void
get_rect_sums(const std::vector<std::array<double, 3>> &prefix_sum,
              const std::vector<std::array<double, 3>> &prefix_sum_sq, int x0,
              int y0, int x1, int y1, int nx, std::array<double, 3> &sums,
              std::array<double, 3> &sum_squares) {
  for (int c = 0; c < 3; c++) {
    sums[c] =
        prefix_sum[y1 * (nx + 1) + x1][c] - prefix_sum[y1 * (nx + 1) + x0][c] -
        prefix_sum[y0 * (nx + 1) + x1][c] + prefix_sum[y0 * (nx + 1) + x0][c];

    sum_squares[c] = prefix_sum_sq[y1 * (nx + 1) + x1][c] -
                     prefix_sum_sq[y1 * (nx + 1) + x0][c] -
                     prefix_sum_sq[y0 * (nx + 1) + x1][c] +
                     prefix_sum_sq[y0 * (nx + 1) + x0][c];
  }
}

Result segment(int ny, int nx, const float *data) {
  Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
  double best_error = std::numeric_limits<double>::infinity();

  // Precompute sums
  std::vector<std::array<double, 3>> prefix_sum, prefix_sum_sq;
  compute_prefix_sums(data, ny, nx, prefix_sum, prefix_sum_sq);

  // Get total sums
  std::array<double, 3> total_sums, total_sum_squares;
  get_rect_sums(prefix_sum, prefix_sum_sq, 0, 0, nx, ny, nx, total_sums,
                total_sum_squares);
  const int total_pixels = nx * ny;

  // Temporary arrays for inner and outer sums
  std::array<double, 3> inner_sums, inner_sum_squares;

  // Try all possible rectangles
  for (int y0 = 0; y0 < ny; y0++) {
    for (int y1 = y0 + 1; y1 <= ny; y1++) {
      for (int x0 = 0; x0 < nx; x0++) {
        for (int x1 = x0 + 1; x1 <= nx; x1++) {
          const int inner_pixels = (x1 - x0) * (y1 - y0);
          const int outer_pixels = total_pixels - inner_pixels;

          if (inner_pixels == 0 || outer_pixels == 0)
            continue;

          // Get sums for inner rectangle
          get_rect_sums(prefix_sum, prefix_sum_sq, x0, y0, x1, y1, nx,
                        inner_sums, inner_sum_squares);

          double total_error = 0;

          // Calculate error in one pass
          for (int c = 0; c < 3; c++) {
            // Inner rectangle error
            const double inner_mean = inner_sums[c] / inner_pixels;
            double inner_error = inner_sum_squares[c] -
                                 2 * inner_mean * inner_sums[c] +
                                 inner_pixels * inner_mean * inner_mean;

            // Outer rectangle error
            const double outer_sum = total_sums[c] - inner_sums[c];
            const double outer_sum_sq =
                total_sum_squares[c] - inner_sum_squares[c];
            const double outer_mean = outer_sum / outer_pixels;
            double outer_error = outer_sum_sq - 2 * outer_mean * outer_sum +
                                 outer_pixels * outer_mean * outer_mean;

            total_error += inner_error + outer_error;

            if (total_error >= best_error)
              goto next_rectangle;
          }

          if (total_error < best_error) {
            best_error = total_error;
            result.x0 = x0;
            result.y0 = y0;
            result.x1 = x1;
            result.y1 = y1;

            for (int c = 0; c < 3; c++) {
              result.inner[c] =
                  static_cast<float>(inner_sums[c] / inner_pixels);
              result.outer[c] = static_cast<float>(
                  (total_sums[c] - inner_sums[c]) / outer_pixels);
            }
          }

        next_rectangle:;
        }
      }
    }
  }

  return result;
}
