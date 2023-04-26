from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from libs import tt_lib as ttm

from transformers import BloomForCausalLM
from utility_functions import print_diff_argmax
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from loguru import logger

import python_api_testing.models.bloom.bloom_utils as bloom_utils
import python_api_testing.models.bloom.bloom_attention as bloom_attention

def run_bloom_attention_test(device):

    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m", torchscript=False)

    tt_bloom_attention = bloom_attention.TtBloomAttention(device, "transformer.h", 0, hugging_bloom_reference_model, 1024, 32, 0.0, 0.0)
    pt_bloom_attention = bloom_attention.BloomAttention("transformer.h",0, hugging_bloom_reference_model, 1024, 32, 0.0, 0.0)

    # Prepare input
    torch.manual_seed(0)

    hidden_states = torch.rand(1, 64, 1024)
    residual = torch.rand(1, 64, 1024)
    alibi = torch.rand(32, 64, 64)
    attention_mask = torch.randint(0, 2, (1, 1, 64, 64))

    pt_out = pt_bloom_attention.forward(hidden_states, residual, alibi, attention_mask)
    print("Finished calc pt")
    print(pt_out)

    tt_out = tt_bloom_attention.forward(device, hidden_states, residual, alibi, attention_mask)
    print("Finished calc tt")

    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)
    pt_out_unsqueezed = pt_out.unsqueeze(0)

    print_diff_argmax(pt_out_unsqueezed, tt_out_converted)
    does_pass, pcc_message = comp_pcc(pt_out_unsqueezed, tt_out_converted, 0.35)

    print(comp_allclose(pt_out_unsqueezed, tt_out_converted))
    print(pcc_message)

    if does_pass:
        logger.info("bloom_attention: Passed!")
    else:
        logger.warning("bloom_attention: Failed!")


    assert does_pass

def test_bloom_attention():
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_bloom_attention_test(device)
    ttm.device.CloseDevice(device)
