# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from functools import wraps
import inspect

from loguru import logger

import ttnn


def compare(torch_outputs, outputs, pcc):
    import torch

    from models.utility_functions import comp_pcc

    if isinstance(outputs, ttnn.Tensor):
        if not isinstance(torch_outputs, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(torch_outputs)}")
        outputs = [outputs]
        torch_outputs = [torch_outputs]
    else:
        if not isinstance(outputs, (list, tuple)):
            raise TypeError(f"Expected list or tuple, got {type(outputs)}")
        if not isinstance(torch_outputs, (list, tuple)):
            raise TypeError(f"Expected list or tuple, got {type(torch_outputs)}")

    matches = True
    last_message = None
    for torch_output, output in zip(torch_outputs, outputs):
        shape = torch_output.shape
        slices = [slice(0, dim) for dim in shape]
        output = ttnn.to_torch(output)
        output = output[slices]
        passed, last_message = comp_pcc(torch_output, output, pcc)
        matches &= passed
    return matches, last_message


ENABLE_VALIDATE_DECORATOR = True
ENABLE_DEBUG_DECORATOR = False
USE_TORCH_OUTPUT_IF_MISMATCHES = False


@contextmanager
def disable_validate_decorator():
    global ENABLE_VALIDATE_DECORATOR
    ENABLE_VALIDATE_DECORATOR = False
    yield
    ENABLE_VALIDATE_DECORATOR = True


PEARSON_CORRELATION_COEFFICIENT = 0.9999


@contextmanager
def override_pearson_correlation_coefficient(value):
    global PEARSON_CORRELATION_COEFFICIENT
    old_value = PEARSON_CORRELATION_COEFFICIENT
    PEARSON_CORRELATION_COEFFICIENT = value
    yield
    PEARSON_CORRELATION_COEFFICIENT = old_value


def convert_torch_output_to_be_like_ttnn_output(torch_output, output):
    torch_output = ttnn.from_torch(torch_output, dtype=output.dtype, layout=output.layout)
    if ttnn.has_storage_type_of(output, ttnn.DEVICE_STORAGE_TYPE):
        torch_output = ttnn.to_device(torch_output, output.device)
    return torch_output


def document_input_tensors(name, function, validate_input_tensors):
    signature = inspect.signature(validate_input_tensors)
    arguments = {arg_name: None for arg_name in signature.parameters}
    arguments["operation_name"] = name
    tensor_names = {
        arg_name: None for arg_name in arguments if arg_name not in {"operation_name", "args", "kwargs", "_"}
    }

    tensor_schemas = []

    def document_validate_input_tensor(_):
        nonlocal tensor_schemas

        def wrapper(*_, **kwargs):
            tensor_schemas.append(kwargs)

        return wrapper

    original_validate_input_tensor = ttnn.validate_input_tensor
    ttnn.validate_input_tensor = document_validate_input_tensor(ttnn.validate_input_tensor)
    try:
        validate_input_tensors(**arguments)
    except:
        pass
    ttnn.validate_input_tensor = original_validate_input_tensor

    if not tensor_schemas:
        # Handle the case for the functions without input tensors
        return

    if len(tensor_names) != len(tensor_schemas):
        raise RuntimeError(f"Expected {len(tensor_names)} tensor_schemas, got {len(tensor_schemas)}")

    doc = function.__doc__ if function.__doc__ is not None else ""
    for tensor_name, tensor_schema in zip(tensor_names, tensor_schemas):
        doc = f"{doc}\n    .. list-table:: {tensor_name}\n\n"

        for index, arg_name in enumerate(tensor_schema.keys()):
            arg_name = arg_name.replace("_", " ")
            bullet_point = f"* -" if index == 0 else "  -"
            doc = f"{doc}        {bullet_point} {arg_name}\n"

        for index, value in enumerate(tensor_schema.values()):
            if isinstance(value, (list, tuple)):

                def to_string(object):
                    try:
                        if isinstance(object, ttnn.DataType):
                            return f"ttnn.{object.name.lower()}"
                        elif isinstance(object, ttnn.Layout):
                            return f"ttnn.{object.name}_LAYOUT"
                        else:
                            return f"{object}"
                    except Exception as e:
                        logger.warning(str(e))
                        return f"{object}"

                value = f"{', '.join([to_string(element) for element in value])}"
            bullet_point = f"* -" if index == 0 else "  -"
            doc = f"{doc}        {bullet_point} {value}\n"

    function.__doc__ = f"{doc}\n"


def register_operation(*, name, validate_input_tensors, torch_function=None):
    def operation_decorator(function):
        document_input_tensors(name, function, validate_input_tensors)

        def validate_decorator(function):
            def call_wrapper(*function_args, **function_kwargs):
                if validate_input_tensors is not None:
                    validate_input_tensors(name, *function_args, **function_kwargs)
                return function(*function_args, **function_kwargs)

            return call_wrapper

        def debug_decorator(function):
            def call_wrapper(*function_args, **function_kwargs):
                if torch_function is not None:
                    logger.info(f"{name} : Comparing against PyTorch")

                if torch_function is not None:
                    torch_output = torch_function(*function_args, **function_kwargs)
                else:
                    torch_output = None

                output = function(*function_args, **function_kwargs)

                if torch_output is not None:
                    matches, last_message = compare(torch_output, output, pcc=PEARSON_CORRELATION_COEFFICIENT)
                    if not matches:
                        if USE_TORCH_OUTPUT_IF_MISMATCHES:
                            logger.warning(f"{name}: Comparing against PyTorch failed, using PyTorch output")
                            if not isinstance(output, Tensor):
                                raise TypeError(f"Expected Tensor, got {type(output)}")
                            output = convert_torch_output_to_be_like_ttnn_output(torch_output, output)
                        else:
                            output = ttnn.to_torch(output)
                            raise RuntimeError(
                                f"{name}: Comparing against PyTorch failed with: {last_message} compared: {torch_output} vs {output}"
                            )

                return output

            return call_wrapper

        @wraps(function)
        def call_wrapper(*function_args, **function_kwargs):
            decorated_function = function
            if ENABLE_VALIDATE_DECORATOR:
                decorated_function = validate_decorator(decorated_function)
            if ENABLE_DEBUG_DECORATOR:
                decorated_function = debug_decorator(decorated_function)
            return decorated_function(*function_args, **function_kwargs)

        return call_wrapper

    return operation_decorator
