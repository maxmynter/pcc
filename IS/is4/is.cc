#include <limits>
#include <vector>

typedef double double4_t __attribute__((vector_size(4 * sizeof(double))));
constexpr double4_t d4zero{0.0, 0.0, 0.0, 0.0};

static inline double sum3(double4_t v) { return v[0] + v[1] + v[2]; }

static inline double4_t mul4(double4_t v, double scalar) { return v * scalar; }

struct Result {
  int y0;
  int x0;
  int y1;
  int x1;
  float outer[3];
  float inner[3];
};

void compute_prefix_sums(const float *data, int ny, int nx,
                         std::vector<double4_t> &prefix_sums) {
  prefix_sums.resize((ny + 1) * (nx + 1), d4zero);

  for (int y = 0; y < ny; ++y) {
    double4_t row_sum = d4zero;
    for (int x = 0; x < nx; ++x) {
      int data_idx = 3 * (x + nx * y);
      double4_t val{static_cast<double>(data[data_idx]),
                    static_cast<double>(data[data_idx + 1]),
                    static_cast<double>(data[data_idx + 2]), 0.0};
      row_sum += val;
      int idx = (y + 1) * (nx + 1) + (x + 1);
      prefix_sums[idx] = row_sum + prefix_sums[idx - (nx + 1)];
    }
  }
}

Result segment(int ny, int nx, const float *data) {
  const int total_pixels = nx * ny;
  std::vector<double4_t> prefix_sums;
  compute_prefix_sums(data, ny, nx, prefix_sums);
  double4_t total_sum = prefix_sums[ny * (nx + 1) + nx];
  Result result{0, 0, 0, 0, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
  double best_error = std::numeric_limits<double>::infinity();

#pragma omp parallel shared(best_error, result)
  {
    double local_best_error = best_error;
    Result local_result = result;

#pragma omp for schedule(dynamic, 1)
    for (int h = 1; h <= ny; ++h) {
      for (int w = 1; w <= nx; ++w) {
        const int rect_size = h * w;
        if (rect_size == total_pixels)
          continue;

        const int outer_pixels = total_pixels - rect_size;
        const double inv_rect_size = 1.0 / rect_size;
        const double inv_outer_pixels = 1.0 / outer_pixels;

        for (int y0 = 0; y0 <= ny - h; ++y0) {
          const int y1 = y0 + h;
          for (int x0 = 0; x0 <= nx - w; ++x0) {
            const int x1 = x0 + w;
            const int n11 = x1 + (nx + 1) * y1;
            const int n01 = x0 + (nx + 1) * y1;
            const int n10 = x1 + (nx + 1) * y0;
            const int n00 = x0 + (nx + 1) * y0;

            double4_t inner_sum = prefix_sums[n11] - prefix_sums[n01] -
                                  prefix_sums[n10] + prefix_sums[n00];
            double4_t outer_sum = total_sum - inner_sum;
            double4_t inner_mean = mul4(inner_sum, inv_rect_size);
            double4_t outer_mean = mul4(outer_sum, inv_outer_pixels);

            double4_t t_cost4 =
                -inner_sum * inner_mean - outer_sum * outer_mean;
            double total_error = sum3(t_cost4);

            if (total_error < local_best_error) {
              local_best_error = total_error;
              local_result = {y0,
                              x0,
                              y1,
                              x1,
                              {static_cast<float>(outer_mean[0]),
                               static_cast<float>(outer_mean[1]),
                               static_cast<float>(outer_mean[2])},
                              {static_cast<float>(inner_mean[0]),
                               static_cast<float>(inner_mean[1]),
                               static_cast<float>(inner_mean[2])}};
            }
          }
        }
      }
    }

#pragma omp critical
    {
      if (local_best_error < best_error) {
        best_error = local_best_error;
        result = local_result;
      }
    }
  }
  return result;
}
