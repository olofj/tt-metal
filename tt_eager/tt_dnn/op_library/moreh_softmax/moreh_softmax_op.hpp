/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tt_dnn/op_library/operation.hpp"
#include "tt_eager/tensor/tensor.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

enum class MorehSoftmaxOpParallelizationStrategy {
    NONE = 0,
    SMALL_W = 1,
    SMALL_H = 2,
    LARGE_W = 3,
    LARGE_H = 4,
    LARGE_C = 5,
};

enum class MorehSoftmaxOp { SOFTMAX = 0, SOFTMIN = 1 };

bool is_moreh_softmax_w_small_available(const Tensor &tensor);
bool is_moreh_softmax_h_small_available(const Tensor &tensor);

operation::ProgramWithCallbacks moreh_softmax_w_small(
    const Tensor &input, Tensor &output, const CoreRange core_range, const MorehSoftmaxOp op);
operation::ProgramWithCallbacks moreh_softmax_w_large(
    const Tensor &input, Tensor &output, const CoreRange core_range, const MorehSoftmaxOp op);
operation::ProgramWithCallbacks moreh_softmax_h_small(
    const Tensor &input, Tensor &output, const CoreRange core_range, const MorehSoftmaxOp op);
operation::ProgramWithCallbacks moreh_softmax_h_large(
    const Tensor &input, Tensor &output, const CoreRange core_range, const MorehSoftmaxOp op);
operation::ProgramWithCallbacks moreh_softmax_c_large(
    const Tensor &input, Tensor &output, uint32_t dim, const CoreRange core_range, const MorehSoftmaxOp op);

struct MorehSoftmax {
    const uint32_t dim;
    const MemoryConfig output_mem_config;
    const CoreRange core_range;  // unused for now
    const MorehSoftmaxOp op;
    const MorehSoftmaxOpParallelizationStrategy strategy;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    MorehSoftmaxOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
    static constexpr auto attribute_names = std::make_tuple("dim", "output_mem_config");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->dim), std::cref(this->output_mem_config));
    }
};

// const ref prevents
Tensor moreh_softmax(
    const Tensor &input_tensor,
    uint32_t dim,
    const MorehSoftmaxOpParallelizationStrategy strategy = MorehSoftmaxOpParallelizationStrategy::NONE,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor moreh_softmin(
    const Tensor &input_tensor,
    uint32_t dim,
    const MorehSoftmaxOpParallelizationStrategy strategy = MorehSoftmaxOpParallelizationStrategy::NONE,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary
}  // namespace operations
}  // namespace tt
