from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../../../..")

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from loguru import logger
import pytest

import tt_lib
from utility_functions_new import (
    profiler,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    comp_pcc,
    get_oom_of_float,
)
from mnist import *

_batch_size = 1


def run_mnist_inference(pcc, PERF_CNT=1):
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    host = tt_lib.device.GetHost()

    # Data preprocessing/loading
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root="data", train=False, transform=transform, download=True
    )
    dataloader = DataLoader(test_dataset, batch_size=_batch_size)

    # Trained to 68% accuracy in modelzoo
    state_dict = torch.load(f"{Path(__file__).parent}/mnist_model.pt")

    tt_mnist_model = TtMnistModel(device, host, state_dict)
    pytorch_mnist_model = PytorchMnistModel(state_dict)

    with torch.no_grad():
        first_input = next(iter(dataloader))
        x, actual_output = first_input

        profiler.enable()

        tt_image = tt_lib.tensor.Tensor(
            x.reshape(-1).tolist(),
            [1, 1, 1, x.shape[2] * x.shape[3]],
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
            device,
        )

        # Run one input through the network
        profiler.start("\nExecution time of tt_mnist first run")
        tt_output = tt_mnist_model(tt_image)
        profiler.end("\nExecution time of tt_mnist first run")

        profiler.start("\nExec time of reference model")
        pytorch_out = pytorch_mnist_model(first_input)
        profiler.end("\nExec time of reference model")

        enable_persistent_kernel_cache()

        logger.info(f"\nRunning the tt_mnist model for {PERF_CNT} iterations . . . ")
        for i in range(PERF_CNT):
            profiler.start("\nAverage execution time of tt_mnist model")
            tt_output = tt_mnist_model(tt_image)
            profiler.end("\nAverage execution time of tt_mnist model")

        tt_output = tt_output.to(host)
        tt_output = torch.Tensor(tt_output.data()).reshape(tt_output.shape())

        logger.info(f"Correct Output: {actual_output}")
        logger.info(f"Predicted Output: {tt_output.topk(1).indices}\n")

        pcc_passing, pcc_output = comp_pcc(pytorch_out, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."

        assert (
            tt_output.topk(10).indices == pytorch_out.topk(10).indices
        ).all(), "The outputs from device and pytorch must have the same topk indices"

        # Check that the scale of each output is the same
        tt_out_oom = get_oom_of_float(tt_output.view(-1).tolist())
        pytorch_out_oom = get_oom_of_float(pytorch_out.tolist())

        assert (
            tt_out_oom == pytorch_out_oom
        ), "The order of magnitudes of the outputs must be the same"

        profiler.print()

        tt_lib.device.CloseDevice(device)


@pytest.mark.parametrize(
    "pcc, iter",
    ((0.99, 2),),
)
def test_mnist_inference(pcc, iter):
    disable_persistent_kernel_cache()
    run_mnist_inference(pcc, iter)


if __name__ == "__main__":
    run_mnist_inference(0.99, 2)
