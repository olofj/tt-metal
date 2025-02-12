# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from models.experimental.functional_stable_diffusion.tt.ttnn_functional_upsample_nearest_2d import upsample_nearest2d
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.utility_functions import skip_for_wormhole_b0, torch_random


@skip_for_wormhole_b0()
@pytest.mark.parametrize("input_shape", [(2, 1280, 4, 4), (2, 1280, 8, 8), (2, 640, 16, 16)])
@pytest.mark.parametrize("scale_factor", [2])
def test_upsample_nearest2d_256x256(reset_seeds, device, input_shape, scale_factor):
    torch_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output = torch.repeat_interleave(torch_tensor, scale_factor, dim=3)
    torch_output = torch.repeat_interleave(torch_output, scale_factor, dim=2)

    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    tt_out = upsample_nearest2d(input_tensor, scale_factor)
    tt_out = ttnn.from_device(tt_out)
    tt_out = ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.to_torch(tt_out)

    assert_with_pcc(torch_output, tt_output, 0.9999)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("input_shape", [(2, 1280, 8, 8), (2, 1280, 16, 16), (2, 640, 32, 32)])
@pytest.mark.parametrize("scale_factor", [2])
def test_upsample_nearest2d_512x512(reset_seeds, device, input_shape, scale_factor):
    torch_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output = torch.repeat_interleave(torch_tensor, scale_factor, dim=3)
    torch_output = torch.repeat_interleave(torch_output, scale_factor, dim=2)

    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    tt_out = upsample_nearest2d(input_tensor, scale_factor)
    tt_out = ttnn.from_device(tt_out)
    tt_out = ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.to_torch(tt_out)

    assert_with_pcc(torch_output, tt_output, 0.9999)
