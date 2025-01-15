from typing import Generator

import pycuda.driver as cuda
import pytest
import torch
import yaml
from attri_util import DEFAULT_METRICS, FLOAT_DTYPES
from performance_utils import Benchmark, get_yaml_path

device = cuda.Device(0)  # MUST set `export CUDA_VISIBLE_DEVICES=?`
max_threads_per_block = device.get_attribute(
    cuda.device_attribute.MAX_THREADS_PER_BLOCK
)
max_warps_per_cta = int(max_threads_per_block / cuda.device_attribute.WARP_SIZE)


"""
Usage:
    pytest test_mm_perf.py --level core --warmup 1000 --iter 1000 \
        --dtypes "float16" --shape_file mm_shape.yaml --record log

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


class MMBenchmark(Benchmark):
    """Benchmark for mm."""

    # ['latency_base', 'latency', 'speedup', 'tflops']
    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"] + ["legacy_shape"]

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def get_input_iter(self, cur_dtype) -> Generator:
        """`self.shapes` can be defined in `core_shapes.yaml`. And if so, this
        configuration is preferred and can be identified by the op name (e.g.
        `mm`) or the class name of the benchmark (e.g. `MMBenchmark`)."""
        for m, n, k in self.shapes:
            yield from self.input_fn(m, n, k, cur_dtype, self.device)

    def set_more_shapes(self):
        """Other appended shapes can be defined here. Only if `--level comprehensive`
        is used and not set `--query`, the appended shapes here will be used."""
        pass

    def get_tflops(self, op, *args, **kwargs):
        """Calculate the total FLOPs of the op."""
        total_flops = 0
        # shape(m,k)(k,n)
        # total_flops mxnx2k
        assert self.op_name == "mm", "The operation is not mm!"
        total_flops = args[0].shape[0] * args[0].shape[1] * args[1].shape[1] * 2

        return total_flops


def mm_input_fn(m, n, k, cur_dtype, device):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
    yield inp1, inp2


def read_config_from_yaml(result_dict):
    with open(get_yaml_path(), "r") as f:
        config = yaml.safe_load(f)
        if (
            "mm" in config
            and isinstance(config["mm"], list)
            and "META" in config["mm"][0]
        ):
            meta = config["mm"][0]["META"]
            # Update result_dict with values from META and other keys
            items_to_add = {
                "BLOCK_M": meta.get("BLOCK_M", None),
                "BLOCK_N": meta.get("BLOCK_N", None),
                "BLOCK_K": meta.get("BLOCK_K", None),
                "SPLIT_K": meta.get("SPLIT_K", None),
                "num_stages": config["mm"][0].get("num_stages", None),
                "num_warps": config["mm"][0].get("num_warps", None),
                "num_ctas": config["mm"][0].get("num_ctas", None),
            }
            autotune_key = "autotune_configs"
            if autotune_key not in result_dict:
                result_dict[autotune_key] = items_to_add


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(
            "mm",
            torch.Tensor.mm,
            mm_input_fn,
            marks=pytest.mark.mm,
        ),
    ],
)
def test_mm_benchmark(op_name, torch_op, input_fn):
    bench = MMBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
        return_all_times=True,  # Return all latencies in a list and print to the log
    )
    bench.run(read_config_from_yaml)
