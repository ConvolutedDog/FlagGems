import itertools
from typing import Generator

import os
import pytest
import torch
import yaml

import pycuda.driver as cuda
import pycuda.autoinit

device = cuda.Device(0) # MUST set `export CUDA_VISIBLE_DEVICES=?` !!!!!!
max_threads_per_block = device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
max_warps_per_cta = int(max_threads_per_block / cuda.device_attribute.WARP_SIZE)

from .attri_util import DEFAULT_METRICS, FLOAT_DTYPES, BenchLevel, llama_shapes
from .conftest import Config
from .performance_utils import Benchmark
from .performance_utils import remove_triton_cache, get_yaml_path

import flag_gems
flag_gems.runtime.config_loader.update_yamlname("IterateBench.yaml")


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


def generate_triton_config(block_m, block_n, block_k, split_k, num_stages, num_warps, num_ctas):
    """
    Generate a Triton configuration string with specific parameters.

    Args:
        block_m (int): Block M size.
        block_n (int): Block N size.
        block_k (int): Block K size.
        split_k (int): Split K factor.
        num_stages (int): Number of pipeline stages.
        num_warps (int): Number of warps.
        num_ctas (int): Number of cooperative thread arrays.

    Returns:
        str: A configuration string that should be written into IterateBench.yaml
    """
#     config = f"""
# mm:
# - META:
#     BLOCK_M: {block_m}
#     BLOCK_N: {block_n}
#     BLOCK_K: {block_k}
#     SPLIT_K: {split_k}
#   num_stages: {num_stages}
#   num_warps: {num_warps}
#   num_ctas: {num_ctas}
# """
    config = f"""
mm:
- META:
    BLOCK_M: {block_m}
    BLOCK_N: {block_n}
    BLOCK_K: {block_k}
    SPLIT_K: {split_k}
  num_stages: {num_stages}
  num_warps: {num_warps}
"""
    return config

class ParameterIterator:
    def __init__(self, block_m_range, block_n_range, block_k_range, 
                 split_k_range, num_stages_range, num_warps_range, num_ctas_range):
        # Define the ranges for each parameter, this should be defined as a tuple
        # or a list, like (start_val, end_val, step, bool_power_of_2).
        self.ranges = {
            "block_m": block_m_range,
            "block_n": block_n_range,
            "block_k": block_k_range,
            "split_k": split_k_range,
            "num_stages": num_stages_range,
            "num_warps": num_warps_range,
            "num_ctas": num_ctas_range,
        }
        # Initialize the current values for each parameter
        self.current_indices = {
            key: 0
            for key in self.ranges
        }
        # Compute the full product of all parameter combinations
        self.keys = list(self.ranges.keys())
        self.total_combinations = 1
        for key in self.keys:
            if len(self.ranges[key]) == 3:
                start, end, step = self.ranges[key]
                self.ranges[key] = list(range(start, end + 1, step))
            elif len(self.ranges[key]) == 4:
                start, end, step, bool_power_of_2 = self.ranges[key]
                if not bool_power_of_2:
                    self.ranges[key] = list(range(start, end + 1, step))
                else:
                    self.ranges[key] = list()
                    value = start
                    while value <= end:
                        self.ranges[key].append(value)
                        value *= step
            self.total_combinations *= len(self.ranges[key])
        self.current_combination = 0

    def get_next_val(self):
        if self.current_combination >= self.total_combinations:
            raise StopIteration("All combinations have been iterated over!")

        # Build the current combination of values
        result = {}
        idx = self.current_combination
        for key in self.keys:
            range_len = len(self.ranges[key])
            current_idx = idx % range_len
            result[key] = self.ranges[key][current_idx]
            idx //= range_len

        self.current_combination += 1
        return result

    def reset(self):
        self.current_combination = 0

paramIter = ParameterIterator(block_m_range=(32, 256, 2, True), # 4
                              block_n_range=(32, 256, 2, True), # 4
                              block_k_range=(32, 128, 2, True), # 3
                              split_k_range=(1, 1, 2, True), # 1
                              num_stages_range=(2, 4, 2, True), # 2
                              num_warps_range=(2, 8, 2, True), # 3
                              num_ctas_range=(1, 1, 2, True)) # 1  # If donnot write num_ctas, should set here to be 1 choice, and 
                                                                   # make comment on generate_triton_config's config string.
def write_next_yaml():
    try:
        params = paramIter.get_next_val()
        # params is like {'block_m': 3, 'block_n': 3, 'block_k': 3, 'split_k': 3, 'num_stages': 3, 'num_warps': 3, 'num_ctas': 3}
        block_m, block_n, block_k, split_k, num_stages, num_warps, num_ctas = \
            params['block_m'], params['block_n'], params['block_k'], params['split_k'], \
            params['num_stages'], params['num_warps'], params['num_ctas']
        new_config = generate_triton_config(block_m, block_n, block_k, split_k, num_stages, num_warps, num_ctas)
        with open(get_yaml_path(), 'w') as file:
            file.write(new_config)
        return True
    except StopIteration as e:
        print(e)
        return False

def read_config_from_yaml(result_dict):
    with open(get_yaml_path(), "r") as f:
        config = yaml.safe_load(f)
        if "mm" in config and isinstance(config["mm"], list) and "META" in config["mm"][0]:
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


class MMShapeGenerator:
    def __init__(self, start, end, step):
        """
        Initializes the MMShapeGenerator with ranges for M, N, and K.
        :param start: Start of the range (inclusive) for M, N, K.
        :param end: End of the range (exclusive) for M, N, K.
        :param step: Step size for the range of M, N, K.
        """
        self.start = start
        self.end = end
        self.step = step

    def generate_shapes(self):
        """
        Generate all combinations of (M, N, K) within the specified ranges.
        :return: A list of [M, N, K] combinations.
        """
        # Create ranges for M, N, K
        m_range = range(self.start, self.end+1, self.step)
        n_range = range(self.start, self.end+1, self.step)
        k_range = range(self.start, self.end+1, self.step)
        
        # Generate the Cartesian product of all combinations for M, N, K.
        shapes = list(itertools.product(m_range, n_range, k_range))
        
        # Convert tuples to lists for output consistency
        shapes = [list(shape) for shape in shapes]
        return shapes

    def save_to_yaml(self, filename="mm_shape.yaml"):
        """
        Save the generated shapes into a YAML file.
        :param filename: Name of the YAML file to save the shapes.
        """

        shapes = self.generate_shapes()
        data = {
            "MMBenchmark": {
                "shapes": shapes,
                "shape_desc": "M, N, K"
            }
        }
        # 自定义 Dumper，确保列表按照扁平格式输出
        class CustomDumper(yaml.SafeDumper):
            def increase_indent(self, flow=False, indentless=False):
                return super(CustomDumper, self).increase_indent(flow, False)
        
        # Dump with PyYAML and ensure proper `[a, b, c]` formatting
        with open(filename, "w") as f:
            yaml.dump(data, f, Dumper=CustomDumper, default_flow_style=None, allow_unicode=True, sort_keys=False, indent=2)

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
    generator = MMShapeGenerator(start=16384, end=16384, step=2048)
    generator.save_to_yaml("mm_shape.yaml")
    # exit()
    while(write_next_yaml()): # iterate config
        # iterate Shape
        remove_triton_cache()
        bench = MMBenchmark(
            input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=FLOAT_DTYPES
        )
        bench.run(read_config_from_yaml)
