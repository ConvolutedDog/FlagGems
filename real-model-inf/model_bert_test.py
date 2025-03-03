import os
import sys
from typing import Generator

import pytest
import torch
from attri_util import DEFAULT_METRICS, FLOAT_DTYPES
from performance_utils import RealModelBenchmark
from transformers import BertConfig, BertModel

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../src"))
sys.path.append(src_dir)

# flake8: noqa: E402
import flag_gems

device = flag_gems.device

"""
Usage:
    pytest model_bert_test.py --level core --warmup 1000 --iter 1000 \
        --dtypes "float16" --shape_file configs/shape.yaml --record log

--mode: "cpu" or NOTSET
    Specify how to measure latency, 'cpu' for CPU-side and NOTSET for Device-size.

--level: str
    Specify the benchmark level: comprehensive, or core.
    Optional: "comprehensive" or "core"

--warmup: int
    Number of warmup runs before benchmark run.

--iter: int
    Number of repeats for each benchmark run.

--metrics: str
    'latency', 'latency_base', 'gbps_base', 'utilization', 'tflops', 'accuracy',
    'error_msg', 'speedup', 'gbps'

--dtypes: str
    Data type used in GPU devices.
    Optional: "float16", "float32", "bfloat16", "int16", "int32", "bool", "cfloat"
    Corresponding to: torch.float16, torch.float32, torch.bfloat16, torch.int16,
                      torch.int32, torch.bool, torch.cfloat

--shape_file: str
    Specify the shape file name for benchmarks. If not specified, a default shape
    list will be used.

--record: str
    Benchmark info recorded in log files or not.
    Optional: "none" or "log"
"""

# Initialize the model with the desired configuration
config = BertConfig(
    attention_probs_dropout_prob=0.0,  # Set attention dropout to be 0.0
    hidden_dropout_prob=0.0,  # Set hidden dropout to be 0.0
)

# Donot load the pretrained weights
model = BertModel(config).to(device).to(torch.float16).eval()


class BERTSMALLBenchmark(RealModelBenchmark):
    """Benchmark for Bert-small."""

    # ['latency_base', 'latency', 'speedup', 'tflops']
    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["legacy_shape"] + ["compared_speedup"]
    # ['latency_base', 'latency', 'speedup', 'tflops', 'latency_torch_compile', 'latency_native_flaggems']
    # TODO: for eval, please open this 4 metrices.
    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["latency_torch_compile"]
    # DEFAULT_METRICS = DEFAULT_METRICS[:] + ["latency_native_flaggems"]
    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["speedup_vs_torch_compile"]
    # DEFAULT_METRICS = DEFAULT_METRICS[:] + ["speedup_vs_native_flaggems"]
    # TODO: fix this, this has to read excel each time, maybe give up this func.
    # DEFAULT_METRICS = DEFAULT_METRICS[:] + ["speedup_vs_native_flaggems_trainset"]

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            yield from self.input_fn(shape, cur_dtype, self.device)

    def set_more_shapes(self):
        return []


def bert_small_input_fn(shape, dtype, device):
    input_ids = torch.randint(0, 10000, shape, dtype=torch.int, device=device)
    # Here the ',' cannot be removed, otherwise it will be treated as a dict.
    yield {"input_ids": input_ids},


@pytest.mark.parametrize(
    "real_model_name, torch_real_model, input_fn",
    [
        pytest.param(
            "Bert-Small",
            model,
            bert_small_input_fn,
            marks=pytest.mark.bert_small,
        ),
    ],
)
def test_bert_small_benchmark(real_model_name, torch_real_model, input_fn):
    bench = BERTSMALLBenchmark(
        input_fn=input_fn,
        real_model_name=real_model_name,
        torch_real_model=torch_real_model,
        dtypes=FLOAT_DTYPES,
        return_all_times=True,  # Return all latencies in a list and print to the log
    )
    bench.run()
