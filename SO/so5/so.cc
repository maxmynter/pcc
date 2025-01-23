#include <algorithm>
#include <cmath>
#include <omp.h>

typedef unsigned long long data_t;

int DEFAULT_SORT_THRES = 4096;
constexpr int MEDIAN_N = 32;

int adaptive_threshold(int n) {
  return std::max(DEFAULT_SORT_THRES, n / (omp_get_max_threads() * 16));
}

int partition(data_t *data, int low, int high) {
  data_t pivots[MEDIAN_N];
  if (high - low >= MEDIAN_N) {
    int step = (high - low) / (MEDIAN_N - 1);
    for (int i = 0; i < MEDIAN_N; i++) {
      pivots[i] = data[low + i * step];
    }
    std::sort(pivots, pivots + MEDIAN_N);
    data_t pivot_val = pivots[MEDIAN_N / 2];

    for (int i = low; i <= high; i++) {
      if (data[i] == pivot_val) {
        std::swap(data[low], data[i]);
        break;
      }
    }
  } else {
    int mid = low + (high - low) / 2;

    if (data[high] < data[low])
      std::swap(data[low], data[high]);
    if (data[mid] < data[low])
      std::swap(data[mid], data[low]);
    if (data[high] < data[mid])
      std::swap(data[high], data[mid]);

    std::swap(data[low], data[mid]);
  }
  data_t pivot = data[low];

  int i = low + 1;
  int j = high;
  while (true) {
    while (i <= j && data[i] < pivot)
      i++;
    while (i <= j && data[j] > pivot)
      j--;
    if (i > j)
      break;
    std::swap(data[i++], data[j--]);
  }
  std::swap(data[low], data[j]);
  return j;
}

void quicksort(data_t *data, int low, int high, int t) {
  while (low < high) {
    if (t <= 0 || (high - low) <= adaptive_threshold(high - low + 1)) {
      std::sort(data + low, data + high + 1);
      return;
    }

    int pi = partition(data, low, high);
    int left_n = pi - low;
    int right_n = high - pi - 1;
    if (left_n > right_n) {
#pragma omp task untied priority(t) if (t > 1)
      quicksort(data, low, pi - 1, t - 1);
      low = pi + 1;
    } else {
#pragma omp task untied priority(t) if (t > 1)
      quicksort(data, pi + 1, high, t - 1);
      high = pi - 1;
    }

    t = t - 1;
  }
}

void psort(int n, data_t *data) {
#pragma omp parallel
#pragma omp single
  {
    int threads = omp_get_max_threads();
    int t = 2 * std::log2(threads) + (n > 1e6 ? 3 : 0);
    quicksort(data, 0, n - 1, t);
  }
}
