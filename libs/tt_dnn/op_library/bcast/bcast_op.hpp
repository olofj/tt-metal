#pragma once

#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

struct BcastOpMath {
    enum Enum { ADD = 0, SUB = 1, MUL = 2 };
    static const vector<Enum> all() { return { ADD, SUB, MUL }; }
};

struct BcastOpDim {
    enum Enum { H = 0, W = 1, HW = 2 };
    static const vector<Enum> all() { return { H, W, HW }; }
};

// TODO: Accept parallelization
struct BcastOpParallelizationStrategy {
    enum Enum { MULTI_CORE_H = 0, MULTI_CORE_W = 1, MULTI_CORE_HW = 2, SINGLE_CORE = 3 };
    static const vector<Enum> all() { return { MULTI_CORE_H, MULTI_CORE_W, MULTI_CORE_HW, SINGLE_CORE }; }
};

Program bcast_single_core(const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, BcastOpMath::Enum bcast_op, BcastOpDim::Enum bcast_dim);
Program bcast_multi_core_h(const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, BcastOpMath::Enum bcast_op, BcastOpDim::Enum bcast_dim);
Program bcast_multi_core_w(const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, BcastOpMath::Enum bcast_op, BcastOpDim::Enum bcast_dim);
Program bcast_multi_core_hw(const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, BcastOpMath::Enum bcast_op, BcastOpDim::Enum bcast_dim);

struct EltwiseBinaryBroadcast {
    const BcastOpMath::Enum math_op;
    const BcastOpDim::Enum dim;

    void validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const;
    Program create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const;
};

inline Tensor bcast(const Tensor &input_tensor_a, const Tensor &input_tensor_b, BcastOpMath::Enum bcast_op, BcastOpDim::Enum bcast_dim) {
    if (bcast_dim == BcastOpDim::W) {
        TT_ASSERT(input_tensor_a.shape()[2] == input_tensor_b.shape()[2]);
    }
    else if (bcast_dim == BcastOpDim::H) {
        TT_ASSERT(input_tensor_a.shape()[3] == input_tensor_b.shape()[3]);
    }
    return operation::run_with_autopad(EltwiseBinaryBroadcast{bcast_op, bcast_dim}, input_tensor_a, input_tensor_b);
}


}  // namespace tt_metal

}  // namespace tt

namespace bcast_op_utils {

using namespace tt::tt_metal;

const char* get_reader_name(BcastOpDim::Enum bcast_dim, BcastOpParallelizationStrategy::Enum bcast_parallelization_strategy);

const char* get_compute_name(BcastOpDim::Enum bcast_dim);

const char* get_math_to_op_define(BcastOpMath::Enum bcast_math);

void add_defines(ComputeKernel * bcast_kernel, BcastOpDim::Enum bcast_dim, BcastOpMath::Enum bcast_math);

BcastOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a);

} // namespace bcast_op_utils
