import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import pytest
import torch
from attri_util import DEFAULT_METRICS, FLOAT_DTYPES
from performance_utils import GenericBenchmark

device = cuda.Device(0)  # MUST set `export CUDA_VISIBLE_DEVICES=?`
max_threads_per_block = device.get_attribute(
    cuda.device_attribute.MAX_THREADS_PER_BLOCK
)
max_warps_per_cta = int(max_threads_per_block / cuda.device_attribute.WARP_SIZE)


"""
Usage:
    pytest test_mm_perf.py --level core --warmup 1000 --iter 1000 \
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


class ConvBenchmark(GenericBenchmark):
    """Benchmark for conv2d."""

    # ['latency_base', 'latency', 'speedup', 'tflops']
    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"] + ["legacy_shape"]
    # ['latency_base', 'latency', 'speedup', 'tflops', 'latency_torch_compile', 'latency_native_flaggems']
    # TODO: for eval, please open this 4 metrices.
    # DEFAULT_METRICS = DEFAULT_METRICS[:] + ["latency_torch_compile"]
    # DEFAULT_METRICS = DEFAULT_METRICS[:] + ["latency_native_flaggems"]
    # DEFAULT_METRICS = DEFAULT_METRICS[:] + ["speedup_vs_torch_compile"]
    # DEFAULT_METRICS = DEFAULT_METRICS[:] + ["speedup_vs_native_flaggems"]
    # TODO: fix this, this has to read excel each time, maybe give up this func.
    # DEFAULT_METRICS = DEFAULT_METRICS[:] + ["speedup_vs_native_flaggems_trainset"]

    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (N, C, H, W).
        return None


@pytest.mark.conv2d
def test_perf_conv2d():
    def conv2d_input_fn(shape, dtype, device):
        (
            batch,
            input_c,
            input_h,
            input_w,
            out_c,
            kernel_h,
            kernel_w,
            stride,
            padding,
            groups,
        ) = shape
        input_shape = (batch, input_c, input_h, input_w)
        weight_shape = (out_c, input_c // groups, kernel_h, kernel_w)
        input = torch.randn(size=input_shape, device=device, dtype=dtype)
        weight = torch.randn(size=weight_shape, device=device, dtype=dtype)

        yield {
            "input": input,
            "weight": weight,
            "bias": None,
            "groups": groups,
            "stride": stride,
            "padding": padding,
        },

    torch.backends.cudnn.allow_tf32 = False
    bench = ConvBenchmark(
        input_fn=conv2d_input_fn,
        op_name="conv2d",
        torch_op=torch.nn.functional.conv2d,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
