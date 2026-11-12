// Experimental outgoing propagation for validated integral anatomical weights.
// Integer-valued FP32 atomic sums are order independent when each destination's
// absolute incoming sum is below 2^24. The Python boundary validates this bound.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

__global__ void compact_spikes(const float* spikes, int* active, int* counts,
                              int n, int batch) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * batch || spikes[index] == 0.f) return;
    int row = index / n;
    int slot = atomicAdd(counts + row, 1);
    active[row * n + slot] = index % n;
}

__global__ void outgoing_spikes(const int* crow, const int* posts,
                               const float* values, const int* active,
                               const int* counts, float* output, int n) {
    int row = blockIdx.y;
    int count = counts[row];
    for (int item = blockIdx.x; item < count; item += gridDim.x) {
        int pre = active[row * n + item];
        int end = crow[pre + 1];
        for (int edge = crow[pre] + threadIdx.x; edge < end; edge += blockDim.x) {
            atomicAdd(output + row * n + posts[edge], values[edge]);
        }
    }
}

void event_step(torch::Tensor crow, torch::Tensor posts, torch::Tensor values,
                torch::Tensor spikes, torch::Tensor active,
                torch::Tensor counts, torch::Tensor output) {
    TORCH_CHECK(spikes.is_cuda() && spikes.scalar_type() == torch::kFloat32 &&
                spikes.dim() == 2 && spikes.is_contiguous(), "CUDA contiguous FP32 spikes required");
    c10::cuda::CUDAGuard guard(spikes.device());
    const auto n = spikes.size(1), batch = spikes.size(0);
    TORCH_CHECK(n > 0 && batch > 0 && batch <= 65535 && n * batch < INT_MAX,
                "event propagation dimensions exceed launch/index range");
    for (const auto& t : {crow, posts, values, active, counts, output})
        TORCH_CHECK(t.device() == spikes.device() && t.is_contiguous(), "same-device contiguous buffers required");
    TORCH_CHECK(crow.scalar_type() == torch::kInt32 && posts.scalar_type() == torch::kInt32 &&
                active.scalar_type() == torch::kInt32 && counts.scalar_type() == torch::kInt32 &&
                values.scalar_type() == torch::kFloat32 && output.scalar_type() == torch::kFloat32,
                "event propagation dtype mismatch");
    TORCH_CHECK(crow.dim() == 1 && crow.numel() == n + 1 && posts.dim() == 1 &&
                values.dim() == 1 && posts.numel() == values.numel() &&
                active.sizes() == spikes.sizes() && output.sizes() == spikes.sizes() &&
                counts.dim() == 1 && counts.numel() == batch, "event propagation shape mismatch");
    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr(), 0, batch * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0, batch * n * sizeof(float), stream));
    compact_spikes<<<(n * batch + 255) / 256, 256, 0, stream>>>(
        spikes.data_ptr<float>(), active.data_ptr<int>(), counts.data_ptr<int>(), n, batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    outgoing_spikes<<<dim3(256, batch), 128, 0, stream>>>(crow.data_ptr<int>(), posts.data_ptr<int>(),
        values.data_ptr<float>(), active.data_ptr<int>(), counts.data_ptr<int>(), output.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("event_step", &event_step); }
