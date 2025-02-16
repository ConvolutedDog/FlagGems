import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import pytest
import torch
from attri_util import DEFAULT_METRICS, FLOAT_DTYPES, BenchLevel
from conftest import Config
from performance_utils import GenericBenchmark

device = cuda.Device(0)  # MUST set `export CUDA_VISIBLE_DEVICES=?`
max_threads_per_block = device.get_attribute(
    cuda.device_attribute.MAX_THREADS_PER_BLOCK
)
max_warps_per_cta = int(max_threads_per_block / cuda.device_attribute.WARP_SIZE)


"""
Usage:
    pytest test_batch_norm_perf.py --level core --warmup 1000 --iter 1000 \
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


class BATCHNORMBenchmark(GenericBenchmark):
    """Benchmark for batch_norm."""

    # ['latency_base', 'latency', 'speedup', 'tflops']
    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["legacy_shape"]
    # ['latency_base', 'latency', 'speedup', 'tflops', 'latency_torch_compile', 'latency_native_flaggems']
    # TODO: for eval, please open this 4 metrices.
    # DEFAULT_METRICS = DEFAULT_METRICS[:] + ["latency_torch_compile"]
    # DEFAULT_METRICS = DEFAULT_METRICS[:] + ["latency_native_flaggems"]
    # DEFAULT_METRICS = DEFAULT_METRICS[:] + ["speedup_vs_torch_compile"]
    # DEFAULT_METRICS = DEFAULT_METRICS[:] + ["speedup_vs_native_flaggems"]
    # TODO: fix this, this has to read excel each time, maybe give up this func.
    # DEFAULT_METRICS = DEFAULT_METRICS[:] + ["speedup_vs_native_flaggems_trainset"]

    def set_more_shapes(self):
        """Other appended shapes can be defined here. Only if `--level comprehensive`
        is used and not set `--query`, the appended shapes here will be used."""
        pass


def batch_norm_input_fn(shape, cur_dtype, device):
    C = shape[1]
    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    weight = torch.randn((C,), dtype=cur_dtype, device=device)
    bias = torch.randn((C,), dtype=cur_dtype, device=device)
    running_mean = None
    running_var = None
    training = True
    momentum = 0.1
    eps = 1e-5
    cudnn_enabled = True
    yield inp, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled

    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        running_mean = torch.randn((C,), dtype=cur_dtype, device=device)
        running_var = torch.randn((C,), dtype=cur_dtype, device=device)
        yield inp, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(
            "batch_norm",
            torch.batch_norm,
            batch_norm_input_fn,
            marks=pytest.mark.batch_norm,
        ),
    ],
)
def test_mean_benchmark(op_name, torch_op, input_fn):
    bench = BATCHNORMBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
        return_all_times=True,  # Return all latencies in a list and print to the log
    )
    bench.run()
