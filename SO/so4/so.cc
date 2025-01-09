#include <algorithm>
#include <cmath>
#include <omp.h>

typedef unsigned long long data_t;

void ppsort(int t, int n, data_t *data) {
  if (t <= 0 || n <= 4096) {
    std::sort(data, data + n);
    return;
  }

  const int m = (n + 1) / 2;

#pragma omp task untied
  ppsort(t - 1, m, data);

#pragma omp task untied
  ppsort(t - 1, n - m, data + m);

#pragma omp taskwait
  std::inplace_merge(data, data + m, data + n);
}

void psort(int n, data_t *data) {
  // Optimal task depth
  int t = std::log2(omp_get_max_threads()) * 1.5;

#pragma omp parallel
  {
#pragma omp single nowait
    {
      ppsort(t, n, data);
    }
  }
}
