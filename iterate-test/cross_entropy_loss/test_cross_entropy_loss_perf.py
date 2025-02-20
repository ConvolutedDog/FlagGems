import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import pytest
import torch
from attri_util import DEFAULT_METRICS, FLOAT_DTYPES
from performance_utils import GenericBenchmark2DOnly, generate_tensor_input

device = cuda.Device(0)  # MUST set `export CUDA_VISIBLE_DEVICES=?`
max_threads_per_block = device.get_attribute(
    cuda.device_attribute.MAX_THREADS_PER_BLOCK
)
max_warps_per_cta = int(max_threads_per_block / cuda.device_attribute.WARP_SIZE)


"""
Usage:
    pytest test_cross_entropy_loss_perf.py --level core --warmup 1000 --iter 1000 \
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


class CROSSENTROPYLOSSBenchmark(GenericBenchmark2DOnly):
    """Benchmark for cross_entropy_loss."""

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


def cross_entropy_loss_input_fn(shape, cur_dtype, device):
    """
    "form_detail_has_weight",
    "form_detail_reduction",  # 0 | 1 | 2 for 'none' | 'mean' | 'sum'.
    "form_detail_label_smoothing",
    """
    (
        shape_detail_N,
        shape_detail_L,
        form_detail_has_weight,
        form_detail_reduction,
        form_detail_label_smoothing,
    ) = shape
    shape = (
        shape_detail_N,
        shape_detail_L,
    )
    inp = generate_tensor_input(shape, cur_dtype, device)
    target = torch.randint(0, shape[-1], (shape[0],), device=device)
    from_detail_dict = {}
    if form_detail_has_weight:
        weight = torch.randn(shape[-1], dtype=cur_dtype, device=device)
        from_detail_dict["weight"] = weight
    if form_detail_reduction == 0:
        from_detail_dict["reduction"] = "none"
    elif form_detail_reduction == 1:
        from_detail_dict["reduction"] = "mean"
    elif form_detail_reduction == 2:
        from_detail_dict["reduction"] = "sum"
    from_detail_dict["label_smoothing"] = form_detail_label_smoothing
    yield inp, target, from_detail_dict


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(
            "cross_entropy_loss",
            torch.nn.functional.cross_entropy,
            cross_entropy_loss_input_fn,
            marks=pytest.mark.cross_entropy_loss,
        ),
    ],
)
def test_cross_entropy_loss_benchmark(op_name, torch_op, input_fn):
    bench = CROSSENTROPYLOSSBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
        return_all_times=True,  # Return all latencies in a list and print to the log
    )
    bench.run()
