#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

__global__ void lif_kernel(float* g, float* spikes, float* v, int16_t* refractory,
                          int32_t* counts, const float* poisson, const float* recurrent,
                          const float* delayed, float* target, int64_t n,
                          float decay, float membrane, float resting, float reset,
                          float threshold, float gain, int refractory_steps) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Match the reference's int16 arithmetic, including wraparound.
    int16_t r = static_cast<int16_t>(refractory[i] * (1 - static_cast<int16_t>(spikes[i])) + 1);
    float next_g = g[i] * decay + delayed[i] * float(r > refractory_steps);
    float next_v = v[i] + membrane * (g[i] - (v[i] - resting));
    float spike = float(next_v > threshold);
    target[i] = gain * (poisson[i] + recurrent[i]);
    refractory[i] = r;
    v[i] = spike ? reset : next_v;
    g[i] = spike ? 0.0f : next_g;
    spikes[i] = spike;
    counts[i] += static_cast<int32_t>(spike);
}

void lif_step(torch::Tensor g, torch::Tensor spikes, torch::Tensor v,
              torch::Tensor refractory, torch::Tensor counts,
              torch::Tensor poisson, torch::Tensor recurrent,
              torch::Tensor delayed, torch::Tensor target,
              double decay, double membrane, double resting, double reset,
              double threshold, double gain, int64_t refractory_steps) {
    TORCH_CHECK(g.is_cuda(), "native LIF requires CUDA tensors");
    c10::cuda::CUDAGuard guard(g.device());
    for (const auto& t : {g, spikes, v, poisson, recurrent, delayed, target}) {
        TORCH_CHECK(t.device() == g.device() && t.scalar_type() == torch::kFloat32 &&
                    t.is_contiguous() && t.sizes() == g.sizes(), "invalid native LIF float tensor");
    }
    for (const auto& t : {refractory, counts}) {
        TORCH_CHECK(t.device() == g.device() && t.is_contiguous() && t.sizes() == g.sizes(),
                    "invalid native LIF integer tensor");
    }
    TORCH_CHECK(refractory.scalar_type() == torch::kInt16 && counts.scalar_type() == torch::kInt32,
                "native LIF integer dtypes must be int16/int32");
    if (!g.numel()) return;
    lif_kernel<<<(g.numel() + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        g.data_ptr<float>(), spikes.data_ptr<float>(), v.data_ptr<float>(),
        refractory.data_ptr<int16_t>(), counts.data_ptr<int32_t>(),
        poisson.data_ptr<float>(), recurrent.data_ptr<float>(), delayed.data_ptr<float>(),
        target.data_ptr<float>(), g.numel(), decay, membrane, resting, reset, threshold,
        gain, refractory_steps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// One fixed reduction tree per learner/post neuron. No floating-point atomics.
__global__ void plastic_kernel(float* fixed, const float* spikes, const float* efficacy,
    const float* anatomical, const int64_t* pre, const int64_t* edges,
    const int64_t* offsets, const int64_t* posts, int64_t neurons,
    int64_t edge_count, int64_t post_count, int64_t efficacy_rows) {
    int row = blockIdx.y;
    int post_slot = blockIdx.x;
    float value = 0.0f;
    for (int64_t cursor = offsets[post_slot] + threadIdx.x;
         cursor < offsets[post_slot + 1]; cursor += blockDim.x) {
        int64_t edge = edges[cursor];
        float contribution = spikes[row * neurons + pre[edge]] *
            efficacy[(efficacy_rows == 1 ? 0 : row) * edge_count + edge];
        value += contribution * anatomical[edge];
    }
    __shared__ float sums[128];
    sums[threadIdx.x] = value;
    __syncthreads();
    for (int stride = 64; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (!threadIdx.x) fixed[row * neurons + posts[post_slot]] += sums[0];
}

void plastic_add(torch::Tensor fixed, torch::Tensor spikes, torch::Tensor efficacy,
    torch::Tensor anatomical, torch::Tensor pre, torch::Tensor edges,
    torch::Tensor offsets, torch::Tensor posts) {
    TORCH_CHECK(fixed.is_cuda() && fixed.dim() == 2, "plastic recurrence requires CUDA matrix");
    c10::cuda::CUDAGuard guard(fixed.device());
    for (const auto& t : {fixed, spikes, efficacy, anatomical})
        TORCH_CHECK(t.device() == fixed.device() && t.is_contiguous() &&
                    t.scalar_type() == torch::kFloat32, "invalid plastic float tensor");
    for (const auto& t : {pre, edges, offsets, posts})
        TORCH_CHECK(t.device() == fixed.device() && t.is_contiguous() &&
                    t.dim() == 1 && t.scalar_type() == torch::kInt64, "invalid plastic index tensor");
    TORCH_CHECK(spikes.sizes() == fixed.sizes() && efficacy.dim() == 2 &&
        (efficacy.size(0) == 1 || efficacy.size(0) == fixed.size(0)) &&
        efficacy.size(1) == pre.numel() && anatomical.numel() == pre.numel() &&
        edges.numel() == pre.numel() && offsets.numel() == posts.numel() + 1,
        "plastic shape mismatch");
    TORCH_CHECK(fixed.size(0) <= 65535, "plastic batch exceeds CUDA grid dimension");
    // Immutable index ranges are checked by backend topology construction.
    if (!posts.numel()) return;
    plastic_kernel<<<dim3(posts.numel(), fixed.size(0)), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        fixed.data_ptr<float>(), spikes.data_ptr<float>(), efficacy.data_ptr<float>(),
        anatomical.data_ptr<float>(), pre.data_ptr<int64_t>(), edges.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(), posts.data_ptr<int64_t>(), fixed.size(1), pre.numel(),
        posts.numel(), efficacy.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("plastic_add", &plastic_add, "Deterministic per-post plastic reduction");
    m.def("lif_step", &lif_step, "Fused reference-order FP32 LIF update");
}
