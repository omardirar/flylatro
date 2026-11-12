// Experimental cuSPARSE algorithm/workspace probe, not selected by training.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cusparse.h>

static void check(cusparseStatus_t status) {
    TORCH_CHECK(status == CUSPARSE_STATUS_SUCCESS, "cuSPARSE status ", int(status), ": ", cusparseGetErrorString(status));
}

class SparsePlan {
    torch::Tensor crow_, col_, values_, input_, output_, workspace_;
    cusparseHandle_t handle_ = nullptr;
    cusparseSpMatDescr_t matrix_ = nullptr;
    cusparseDnVecDescr_t vector_x_ = nullptr, vector_y_ = nullptr;
    cusparseDnMatDescr_t dense_x_ = nullptr, dense_y_ = nullptr;
    bool vector_;
    int algorithm_;
    float alpha_ = 1.f, beta_ = 0.f;
    void cleanup() noexcept {
        if (handle_) { cudaStream_t stream; if (cusparseGetStream(handle_, &stream) == CUSPARSE_STATUS_SUCCESS) cudaStreamSynchronize(stream); }
        if (vector_x_) cusparseDestroyDnVec(vector_x_);
        if (vector_y_) cusparseDestroyDnVec(vector_y_);
        if (dense_x_) cusparseDestroyDnMat(dense_x_);
        if (dense_y_) cusparseDestroyDnMat(dense_y_);
        if (matrix_) cusparseDestroySpMat(matrix_);
        if (handle_) cusparseDestroy(handle_);
    }
public:
    SparsePlan(torch::Tensor crow, torch::Tensor col, torch::Tensor values,
               torch::Tensor input, bool vector, int algorithm, bool row_major,
               bool preprocess): crow_(crow), col_(col), values_(values), input_(input),
               vector_(vector), algorithm_(algorithm) {
        TORCH_CHECK(values.is_cuda() && values.scalar_type() == torch::kFloat32 && values.is_contiguous(), "CUDA FP32 values required");
        c10::cuda::CUDAGuard guard(values.device());
        TORCH_CHECK(crow.device() == values.device() && col.device() == values.device() &&
            crow.scalar_type() == torch::kInt32 && col.scalar_type() == torch::kInt32 &&
            crow.is_contiguous() && col.is_contiguous(), "contiguous CUDA int32 CSR indices required");
        TORCH_CHECK(input.device() == values.device() && input.scalar_type() == torch::kFloat32 && input.dim() == 2,
            "CUDA FP32 dense matrix required");
        int64_t n = crow.numel() - 1, batch = input.size(1);
        TORCH_CHECK(n > 0 && batch > 0 && input.size(0) == n && col.numel() == values.numel(), "CSR/dense shape mismatch");
        TORCH_CHECK(!vector || batch == 1, "SpMV requires batch one");
        TORCH_CHECK(row_major ? (input.stride(1) == 1 && input.stride(0) == batch) :
            (input.stride(0) == 1 && input.stride(1) == n), "dense layout mismatch");
        output_ = torch::empty_strided(input.sizes(), input.strides(), input.options());
        try {
            check(cusparseCreate(&handle_));
            check(cusparseSetStream(handle_, at::cuda::getCurrentCUDAStream()));
            check(cusparseCreateCsr(&matrix_, n, n, values.numel(), crow.data_ptr(), col.data_ptr(), values.data_ptr(),
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
            size_t bytes = 0;
            if (vector_) {
                check(cusparseCreateDnVec(&vector_x_, n, input_.data_ptr(), CUDA_R_32F));
                check(cusparseCreateDnVec(&vector_y_, n, output_.data_ptr(), CUDA_R_32F));
                check(cusparseSpMV_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, matrix_, vector_x_,
                    &beta_, vector_y_, CUDA_R_32F, (cusparseSpMVAlg_t)algorithm_, &bytes));
            } else {
                auto order = row_major ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
                int64_t ld = row_major ? batch : n;
                check(cusparseCreateDnMat(&dense_x_, n, batch, ld, input_.data_ptr(), CUDA_R_32F, order));
                check(cusparseCreateDnMat(&dense_y_, n, batch, ld, output_.data_ptr(), CUDA_R_32F, order));
                check(cusparseSpMM_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha_, matrix_, dense_x_, &beta_, dense_y_, CUDA_R_32F, (cusparseSpMMAlg_t)algorithm_, &bytes));
            }
            workspace_ = torch::empty({(int64_t)bytes}, values.options().dtype(torch::kUInt8));
            if (preprocess) {
                if (vector_) check(cusparseSpMV_preprocess(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, matrix_, vector_x_,
                    &beta_, vector_y_, CUDA_R_32F, (cusparseSpMVAlg_t)algorithm_, workspace_.data_ptr()));
                else check(cusparseSpMM_preprocess(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha_, matrix_, dense_x_, &beta_, dense_y_, CUDA_R_32F, (cusparseSpMMAlg_t)algorithm_, workspace_.data_ptr()));
            }
        } catch (...) { cleanup(); throw; }
    }
    ~SparsePlan() { cleanup(); }
    torch::Tensor run() {
        c10::cuda::CUDAGuard guard(values_.device());
        check(cusparseSetStream(handle_, at::cuda::getCurrentCUDAStream()));
        if (vector_) check(cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, matrix_, vector_x_,
            &beta_, vector_y_, CUDA_R_32F, (cusparseSpMVAlg_t)algorithm_, workspace_.data_ptr()));
        else check(cusparseSpMM(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha_, matrix_, dense_x_, &beta_, dense_y_, CUDA_R_32F, (cusparseSpMMAlg_t)algorithm_, workspace_.data_ptr()));
        return output_;
    }
    int64_t workspace_bytes() const { return workspace_.numel(); }
};
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<SparsePlan>(m, "SparsePlan")
        .def(pybind11::init<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,bool,int,bool,bool>())
        .def("run", &SparsePlan::run)
        .def("workspace_bytes", &SparsePlan::workspace_bytes);
}
