# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from loguru import logger
from transformers import BertForQuestionAnswering, BertTokenizer

import tt_lib

from models.demos.metal_BERT_large_11.tt.bert_model import TtBertBatchDram
from models.demos.metal_BERT_large_11.tt.model_config import get_model_config, get_tt_cache_path
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    profiler,
    is_e75,
)
from models.perf.perf_utils import prep_perf_report

BATCH_SIZE = 12
model_version = "phiyodr/bert-large-finetuned-squad2"
comments = "Large"
seq_len = 384
real_input = True
attention_mask = True
token_type_ids = True
model_config_str = "BFLOAT8_B-SHARDED_BATCH12"


def run_perf_bert11(
    expected_inference_time, expected_compile_time, inference_iterations, model_location_generator, device
):
    model_config = get_model_config(BATCH_SIZE, model_config_str)
    tt_cache_path = get_tt_cache_path(model_version)
    model_name = str(model_location_generator(model_version, model_subdir="Bert"))
    tokenizer_name = str(model_location_generator(model_version, model_subdir="Bert"))

    disable_persistent_kernel_cache()
    first_embedding_key = "first_embedding_preprocessing"
    first_attention_mask_key = "first_attention_mask"
    first_run_key = "first_run"
    second_run_accum_key = "second_run_accum"
    second_run_key = "second_run"
    cpu_key = "ref_key"

    HF_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    HF_model.eval()
    tt_model = TtBertBatchDram(
        HF_model.config,
        HF_model,
        device,
        model_config,
        tt_cache_path,
    )

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    context = BATCH_SIZE * [
        "Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. The prophet and founding hero of modern archaeology, Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art."
    ]
    question = BATCH_SIZE * ["What discipline did Winkelmann create?"]
    inputs = tokenizer.batch_encode_plus(
        zip(question, context),
        max_length=seq_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=attention_mask,
        return_token_type_ids=token_type_ids,
        return_tensors="pt",
    )

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_out = HF_model(**inputs)
        profiler.end(cpu_key)

        profiler.start(first_attention_mask_key)
        tt_attention_mask_host = tt_model.model_attention_mask(**inputs)
        profiler.end(first_attention_mask_key, force_enable=True)

        profiler.start(first_embedding_key)
        tt_embedding_inputs_host = tt_model.embeddings.preprocess_embedding_inputs(**inputs)
        profiler.end(first_embedding_key, force_enable=True)

        profiler.start(first_run_key)
        tt_attention_mask = tt_attention_mask_host.to(device, model_config["OP4_SOFTMAX_ATTENTION_MASK_MEMCFG"])
        tt_embedding_inputs = {
            key: value.to(device, model_config["INPUT_EMBEDDINGS_MEMCFG"])
            for (key, value) in tt_embedding_inputs_host.items()
        }
        tt_embedding = tt_model.model_embedding(**tt_embedding_inputs)
        tt_output = tt_model(tt_embedding, tt_attention_mask).cpu()
        profiler.end(first_run_key, force_enable=True)
        del tt_attention_mask
        del tt_embedding_inputs
        del tt_embedding
        del tt_output
        enable_persistent_kernel_cache()

        profiler.start(second_run_accum_key)
        # First input to device
        tt_attention_mask = tt_attention_mask_host.to(device, model_config["OP4_SOFTMAX_ATTENTION_MASK_MEMCFG"])
        tt_embedding_inputs = {
            key: value.to(device, model_config["INPUT_EMBEDDINGS_MEMCFG"])
            for (key, value) in tt_embedding_inputs_host.items()
        }
        # Run inference iterations - 1 times with sending next input/reading output interleaved
        for _ in range(inference_iterations - 1):
            tt_embedding = tt_model.model_embedding(**tt_embedding_inputs)
            tt_output = tt_model(tt_embedding, tt_attention_mask)
            tt_attention_mask = tt_attention_mask_host.to(device, model_config["OP4_SOFTMAX_ATTENTION_MASK_MEMCFG"])
            tt_embedding_inputs = {
                key: value.to(device, model_config["INPUT_EMBEDDINGS_MEMCFG"])
                for (key, value) in tt_embedding_inputs_host.items()
            }
            tt_output = tt_output.cpu()
        # Run last inference iteration
        tt_embedding = tt_model.model_embedding(**tt_embedding_inputs)
        tt_output = tt_model(tt_embedding, tt_attention_mask)
        tt_output = tt_output.cpu()

        profiler.end(second_run_accum_key, force_enable=True)

        del tt_attention_mask
        del tt_embedding_inputs
        del tt_embedding
        del tt_output

    first_iter_time = profiler.get(first_run_key)
    second_iter_time = profiler.get(second_run_accum_key) / inference_iterations
    cpu_time = profiler.get(cpu_key)

    prep_perf_report(
        model_name="bert11",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )
    compile_time = first_iter_time - second_iter_time
    logger.info(f"bert11 inference time: {second_iter_time}")
    logger.info(f"bert11 compile time: {compile_time}")


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, inference_iterations",
    ([0.05, 14.5, 10],),
)
def test_perf_virtual_machine(
    use_program_cache,
    expected_inference_time,
    expected_compile_time,
    inference_iterations,
    model_location_generator,
    device,
):
    if is_e75(device):
        pytest.skip("Bert large 15 is not supported on E75")

    run_perf_bert11(
        expected_inference_time, expected_compile_time, inference_iterations, model_location_generator, device
    )


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, inference_iterations",
    ([0.0364, 10, 10],),
)
def test_perf_bare_metal(
    use_program_cache,
    expected_inference_time,
    expected_compile_time,
    inference_iterations,
    model_location_generator,
    device,
):
    if is_e75(device):
        pytest.skip("Bert large 15 is not supported on E75")

    run_perf_bert11(
        expected_inference_time, expected_compile_time, inference_iterations, model_location_generator, device
    )
