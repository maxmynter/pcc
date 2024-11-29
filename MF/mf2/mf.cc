
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/
#include <algorithm>
#include <vector>

int partition(std::vector<float> &arr, int left, int right) {
  float pivot = arr[right];
  int i = left - 1;

  for (int j = left; j < right; j++) {
    if (arr[j] <= pivot) {
      i++;
      std::swap(arr[i], arr[j]);
    }
  }
  std::swap(arr[i + 1], arr[right]);
  return i + 1;
}

float quickselect(std::vector<float> &arr, int left, int right, int k) {
  if (left == right) {
    return arr[left];
  }
  int pivot_idx = partition(arr, left, right);
  int length = pivot_idx - left + 1;
  if (k == length) {
    return arr[pivot_idx];
  } else if (k < length) {
    return quickselect(arr, left, pivot_idx - 1, k);
  } else {
    return quickselect(arr, pivot_idx + 1, right, k - length);
  }
}

void mf(int ny, int nx, int hy, int hx, const float *in, float *out) {
#pragma omp parallel
  {
    std::vector<float> window;
    window.reserve((2 * hx + 1) * (2 * hy + 1));

#pragma omp for collapse(2)
    for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
        window.clear();
        for (int j = std::max(0, y - hy); j <= std::min(ny - 1, y + hy); j++) {
          for (int i = std::max(0, x - hx); i <= std::min(nx - 1, x + hx);
               i++) {
            window.push_back(in[i + nx * j]);
          }
        }
        int n = window.size();
        if (n % 2 == 1) {
          out[x + nx * y] = quickselect(window, 0, n - 1, n / 2 + 1);
        } else {
          float lower = quickselect(window, 0, n - 1, n / 2);
          float upper = quickselect(window, 0, n - 1, n / 2 + 1);
          out[x + nx * y] = (lower + upper) / 2.0f;
        }
      }
    }
  }
}
