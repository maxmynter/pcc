#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

static inline void check(cudaError_t err, const char *context) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << context << ": " << cudaGetErrorString(err)
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

static inline int divup(int a, int b) { return (a + b - 1) / b; }

#define CHECK(x) check(x, #x)

__global__ void compute_means_and_diffs(const float *data, float *diffs, int ny,
                                        int nx) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= ny)
    return;

  // Compute mean
  float mean = 0.0f;
  for (int k = 0; k < nx; k++) {
    mean += data[k + i * nx];
  }
  mean /= nx;

  // Compute differences and store them
  for (int k = 0; k < nx; k++) {
    diffs[k + i * nx] = data[k + i * nx] - mean;
  }
}

__global__ void compute_correlations(const float *diffs, float *result, int ny,
                                     int nx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= ny || j > i)
    return;

  float sum_ij = 0.0f;
  float sum_i = 0.0f;
  float sum_j = 0.0f;

  for (int k = 0; k < nx; k++) {
    float diff_i = diffs[k + i * nx];
    float diff_j = diffs[k + j * nx];
    sum_ij += diff_i * diff_j;
    sum_i += diff_i * diff_i;
    sum_j += diff_j * diff_j;
  }

  float denominator = sqrt(sum_i * sum_j);
  result[i + j * ny] = denominator != 0.0f ? sum_ij / denominator : 0.0f;
}

__global__ void initialize_result(float *result, int ny) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < ny && j < ny) {
    result[i + j * ny] = 0.0f;
  }
}

void correlate(int ny, int nx, const float *data, float *result) {
  float *d_data = nullptr;
  float *d_diffs = nullptr;
  float *d_result = nullptr;

  // Allocate device memory
  CHECK(cudaMalloc(&d_data, nx * ny * sizeof(float)));
  CHECK(cudaMalloc(&d_diffs, nx * ny * sizeof(float)));
  CHECK(cudaMalloc(&d_result, ny * ny * sizeof(float)));

  // Initialize result array with zeros
  dim3 init_block(16, 16);
  dim3 init_grid(divup(ny, 16), divup(ny, 16));
  initialize_result<<<init_grid, init_block>>>(d_result, ny);
  CHECK(cudaGetLastError());

  // Copy input data to device
  CHECK(cudaMemcpy(d_data, data, nx * ny * sizeof(float),
                   cudaMemcpyHostToDevice));

  // Compute means and differences
  const int BLOCK_SIZE = 256;
  int grid_size = divup(ny, BLOCK_SIZE);
  compute_means_and_diffs<<<grid_size, BLOCK_SIZE>>>(d_data, d_diffs, ny, nx);
  CHECK(cudaGetLastError());

  // Compute correlations
  dim3 corr_block(16, 16);
  dim3 corr_grid(divup(ny, 16), divup(ny, 16));
  compute_correlations<<<corr_grid, corr_block>>>(d_diffs, d_result, ny, nx);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  // Copy results back to host
  CHECK(cudaMemcpy(result, d_result, ny * ny * sizeof(float),
                   cudaMemcpyDeviceToHost));

  // Cleanup
  CHECK(cudaFree(d_data));
  CHECK(cudaFree(d_diffs));
  CHECK(cudaFree(d_result));
}
