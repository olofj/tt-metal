# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union


import tt_lib as ttl

import ttnn


def _torch_pad_to_tile(padded_tensor: ttnn.Tensor):
    import torch

    padded_tensor = ttnn.from_device(padded_tensor)
    padded_tensor = ttnn.to_layout(padded_tensor, ttnn.ROW_MAJOR_LAYOUT)
    shape = padded_tensor.shape
    padded_tensor = ttnn.to_torch(padded_tensor)
    output_tensor = torch.narrow(padded_tensor, shape)
    return output_tensor


def _pad_to_tile_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(1, 2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.uint32),
        layouts=(ttnn.ROW_MAJOR_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=True,
    )


@ttnn.register_operation(
    name="ttnn.pad_to_tile",
    validate_input_tensors=_pad_to_tile_validate_input_tensors,
    torch_function=_torch_pad_to_tile,
)
def pad_to_tile(input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    r"""
    pad(input_tensor: ttnn.Tensor) -> ttnn.Tensor

    Pad :attr:`input_tensor` so that the last two dimensions are multiples of 32.

    Args:
        * :attr:`input_tensor`: the input tensor off of device

    Example::

        >>> tensor = ttnn.from_torch(torch.zeros((62, 30), dtype=torch.bfloat16))
        >>> output = ttnn.pad_to_tile(tensor)
        >>> print(tensor.shape)
        >>> print(tensor.shape.padded())

    """
    height_multiple = 32
    width_multiple = 32

    # if len(input_tensor.shape) < 2:
    #     input_tensor = _reshape_to_2D(input_tensor)

    # *_, height, width = input_tensor.shape

    def ttnn_pad(tensor):
        if len(tensor.shape) > 1:
            *original_batch_sizes, height, width = tensor.shape
            pad_h = (height_multiple - height % height_multiple) % height_multiple
            pad_w = (width_multiple - width % width_multiple) % width_multiple

            padded_height = height + pad_h
            padded_width = width + pad_w
            tensor = ttnn.unsqueeze_to_4D(tensor)
            *batch_sizes, _, _ = tensor.shape
            ttl_input_tensor = tensor.value
            if ttnn.has_storage_type_of(input_tensor, ttl.tensor.StorageType.DEVICE):
                tensor = ttnn.Tensor(
                    ttl.tensor.tilize_with_val_padding(
                        ttl_input_tensor,
                        batch_sizes + [padded_height, padded_width],
                        [0, 0, 0, 0],
                        float("-inf"),
                    )
                )
            else:
                tensor = ttnn.Tensor(
                    ttl_input_tensor.pad(batch_sizes + [padded_height, padded_width], [0, 0, 0, 0], 0.0)
                )
                tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
            tensor = ttnn.reshape(
                tensor,
                ttnn.Shape(
                    original_batch_sizes + [height, width], original_batch_sizes + [padded_height, padded_width]
                ),
            )
            return tensor
        else:
            (width,) = tensor.shape
            if width % width_multiple == 0:
                return tensor

            pad_w = (width_multiple - width % width_multiple) % width_multiple
            padded_width = width + pad_w
            tensor = ttnn.unsqueeze_to_4D(tensor)
            *batch_sizes, height, _ = tensor.shape
            tensor = ttnn.Tensor(tensor.value(batch_sizes + [height, padded_width], [0, 0, 0, 0], 0.0))
            tensor = ttnn.reshape(tensor, ttnn.Shape([width], [padded_width]))
            return tensor

    return ttl.tensor.decorate_external_operation(ttnn_pad, function_name="ttnn.pad_to_tile")(input_tensor)


def _torch_unpad_from_tile(padded_tensor: ttnn.Tensor):
    import torch

    padded_tensor = ttnn.from_device(padded_tensor)
    padded_tensor = ttnn.to_layout(padded_tensor, ttnn.ROW_MAJOR_LAYOUT)
    shape = padded_tensor.shape
    padded_tensor = ttnn.to_torch(padded_tensor)
    output_tensor = torch.narrow(padded_tensor, shape)
    return output_tensor


def _unpad_from_tile_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.uint32),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=True,
    )


@ttnn.register_operation(
    name="ttnn.unpad_from_tile",
    validate_input_tensors=_unpad_from_tile_validate_input_tensors,
    torch_function=_torch_unpad_from_tile,
)
def unpad_from_tile(input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    r"""
    unpad(input_tensor: ttnn.Tensor) -> ttnn.Tensor

    Unpad :attr:`input_tensor`.

    Args:
        * :attr:`input_tensor`: the input tensor off of device

    Example::

        >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((62, 30), dtype=torch.bfloat16)), device)
        >>> tensor = ttnn.pad_to_tile(tensor)
        >>> print(tensor.shape)
        >>> print(tensor.shape.padded())
        >>> tensor = ttnn.from_device(tensor)
        >>> output = ttnn.unpad_from_tile(tensor)
        >>> print(output.shape)
        >>> print(output.shape.padded())

    """

    def ttnn_unpad(tensor):
        nonlocal input_tensor
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            raise RuntimeError("input tensor must be in ttnn.TILE_LAYOUT")
        # input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
        intended_shape = tuple(tensor.shape)
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
        intended_4D_shape = tuple(x - 1 for x in input_tensor.shape)
        ttl_input_tensor = input_tensor.value
        if ttnn.has_storage_type_of(input_tensor, ttl.tensor.StorageType.DEVICE):
            output_tensor = ttnn.Tensor(
                ttl.tensor.untilize_with_unpadding(
                    ttl_input_tensor,
                    (0, 0, 0, 0),
                    intended_4D_shape,
                )
            )
        else:
            output_tensor = ttnn.Tensor(
                ttl_input_tensor.to(ttnn.ROW_MAJOR_LAYOUT).unpad_from_tile(list(input_tensor.shape))
            )
        output_tensor = ttnn.reshape(output_tensor, intended_shape)
        return output_tensor

    return ttl.tensor.decorate_external_operation(ttnn_unpad, function_name="ttnn.unpad_from_tile")(input_tensor)


def _torch_embedding(input_tensor: ttnn.Tensor, weight: ttnn.Tensor, **_):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_torch(input_tensor)

    weight = ttnn.from_device(weight)
    weight = ttnn.to_torch(weight)

    output_tensor = torch.nn.functional.embedding(input_tensor, weight)

    return output_tensor


def _embedding_validate_input_tensors(operation_name, input_tensor, weight, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.uint32,),
        layouts=(ttnn.ROW_MAJOR_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )
    ttnn.validate_input_tensor(
        operation_name,
        weight,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16,),
        layouts=(ttnn.ROW_MAJOR_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(
    name="ttnn.embedding",
    validate_input_tensors=_embedding_validate_input_tensors,
    torch_function=_torch_embedding,
)
def embedding(
    input_tensor: ttnn.Tensor,
    weight: ttnn.Tensor,
    *,
    layout: ttnn.Layout = ttnn.ROW_MAJOR_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
):
    r"""
    embedding(inxput_tensor: ttnn.Tensor, weight: ttnn.Tensor, *, layout: ttnn.Layout = ttnn.ROW_MAJOR_LAYOUT, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Retrieves word embeddings using input_tensor. The input_tensor is a list of indices, and the embedding matrix, and the output is the corresponding word embeddings.

    Args:
        * :attr:`input_tensor`: the indices ttnn.Tensor
        * :attr:`weight`: the embeddings ttnn.Tensor that correspond to the indices ttnn.Tensor

    Example::
        >>> device_id = 0
        >>> device = ttnn.open(device_id)
        >>> input_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]]), dtype=ttnn.uint32), device)
        >>> # an embedding matrix containing 10 tensors of size 4
        >>> weight = ttnn.to_device(ttnn.from_torch(torch.rand(10, 4), dtype=ttnn.bfloat16), device)
        >>> ttnn.embedding(input_tensor, weight)
        ttnn.Tensor([ [[1, 0.106445, 0.988281, 0.59375],
            [0.212891, 0.964844, 0.199219, 0.996094],
            [3.78362e-38, 0, 7.89785e-39, 0],
            [8.04479e-38, 0, 1.25815e-38, 0]],
           [[2.71833e-38, 0, 3.59995e-38, 0],
            [7.60398e-38, 0, 1.83671e-38, 0],
            [2.22242e-38, 0, 1.88263e-38, 0],
            [1.35917e-38, 0, 4.49994e-39, 0]]], dtype=bfloat16 )

    """
    if len(input_tensor.shape) != 2:
        raise RuntimeError("Input Tensor must have rank of 2!")
    if len(weight.shape) not in {2, 4}:
        raise RuntimeError("Weight Tensor must either have rank of 2 or 4!")

    *_, hidden_embedding_dim = tuple(weight.shape)
    weight = ttnn.unsqueeze_to_4D(weight)

    batch_size, sentence_size = input_tensor.shape
    input_tensor = ttnn.reshape(input_tensor, shape=(batch_size, 1, 1, sentence_size))

    tilized = layout == ttnn.TILE_LAYOUT
    embeddings = ttnn.Tensor(
        ttl.tensor.embeddings(input_tensor.value, weight.value, tilized, output_mem_config=memory_config)
    )
    embeddings = ttnn.reshape(embeddings, shape=(batch_size, sentence_size, hidden_embedding_dim))

    return embeddings


def _torch_softmax(input_tensor: ttnn.Tensor, dim: int, **_):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    return torch.softmax(input_tensor, dim)


def _softmax_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(
    name="ttnn.softmax",
    validate_input_tensors=_softmax_validate_input_tensors,
    torch_function=_torch_softmax,
)
def softmax(
    input_tensor: ttnn.Tensor, dim: int, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
) -> ttnn.Tensor:
    r"""
    softmax(input_tensor: ttnn.Tensor, dim: int, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Compute softmax over :attr:`input_tensor` along :attr:`dim`.

    Args:
        * :attr:`input_tensor`: the input tensor
        * :attr:`dim`: the dimension along which to compute softmax.

    Example::

        >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16)), device)
        >>> output = ttnn.softmax(tensor, -1)
        >>> print(output[0, 0, 0, :3])
        ttnn.Tensor([ 0.0310059, 0.0310059, 0.0310059], dtype=bfloat16 )

    """

    input_shape = input_tensor.shape
    rank = len(input_shape)
    if dim < 0:
        dim = rank + dim

    input_tensor = ttnn.unsqueeze_to_4D(input_tensor)

    is_padded_and_using_tile = (
        input_tensor.layout == ttnn.TILE_LAYOUT
        and list(input_tensor.shape)[-2:] != list(input_tensor.shape.padded())[-2:]
    )
    if not is_padded_and_using_tile and dim == rank - 1:
        ttl_input_tensor = input_tensor.value
        # TODO: #4599 Research why softmax appears to not be stable when using a padded ttnn.TILE_LAYOUT
        ttl_output_tensor = ttl.tensor.softmax(ttl_input_tensor, output_mem_config=memory_config)
    else:
        dim_4D = dim + 4 - rank

        input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
        ttl_input_tensor = input_tensor.value

        ttl_output_tensor = ttl.operations.primary.moreh_softmax(
            ttl_input_tensor, dim=dim_4D, output_mem_config=memory_config
        )
    output_tensor = ttnn.Tensor(ttl_output_tensor)
    output_tensor = ttnn.reshape(output_tensor, input_shape)
    return output_tensor


def _torch_mean(input_tensor: ttnn.Tensor, dim: int, keepdim=False, **_):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    return torch.mean(input_tensor, dim=dim, keepdim=keepdim)


def _mean_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(
    name="ttnn.mean",
    validate_input_tensors=_mean_validate_input_tensors,
    torch_function=_torch_mean,
)
def mean(input_tensor: ttnn.Tensor, dim: Union[int, Tuple[int]], keepdim: bool = False) -> ttnn.Tensor:
    """
    mean(input_tensor: ttnn.Tensor, dim: Union[int, Tuple[int]], keepdim: bool = False) -> ttnn.Tensor
    """

    input_shape = tuple(input_tensor.shape)
    rank = len(input_shape)

    if isinstance(dim, int):
        if dim < 0:
            dim = rank + dim
        dim = (dim,)

    if isinstance(dim, tuple):
        if dim == (rank - 1,):
            reduce_op_dim = ttl.tensor.ReduceOpDim.W
        elif dim == (rank - 2,):
            reduce_op_dim = ttl.tensor.ReduceOpDim.H
        elif dim == (rank - 1, rank - 2):
            reduce_op_dim = ttl.tensor.ReduceOpDim.HW
        else:
            raise RuntimeError("Unsupported dim")
    else:
        raise RuntimeError("Invalid dim")

    output_shape = []
    padded_output_shape = []
    for axis, size in enumerate(input_shape):
        if axis in dim:
            if keepdim:
                output_shape.append(1)
                padded_output_shape.append(ttnn.TILE_SIZE)
        else:
            output_shape.append(size)
            padded_output_shape.append(size)
    output_shape = tuple(output_shape)
    padded_output_shape = tuple(padded_output_shape)

    input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
    ttl_input_tensor = input_tensor.value
    ttl_output_tensor = ttl.tensor.reduce(
        ttl_input_tensor, ttl.tensor.ReduceOpMath.SUM, reduce_op_dim, 1 / input_shape[-1]
    )
    ttl_output_tensor = ttl.tensor.reduce(
        ttl_input_tensor, ttl.tensor.ReduceOpMath.SUM, reduce_op_dim, 1 / input_shape[-1]
    )

    output_tensor = ttnn.Tensor(ttl_output_tensor)
    output_tensor = ttnn.reshape(output_tensor, ttnn.Shape(output_shape, padded_output_shape))

    return output_tensor


__all__ = []
