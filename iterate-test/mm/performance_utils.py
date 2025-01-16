import gc
import importlib
import itertools
import logging
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Generator, List, Optional, Tuple

import pytest
import torch
import yaml
from ruamel.yaml import YAML

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../src"))
sys.path.append(src_dir)

# flake8: noqa: E402
import flag_gems

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# flake8: noqa: E402
from attri_util import (
    BOOL_DTYPES,
    DEFAULT_METRICS,
    DEFAULT_SHAPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    BenchLevel,
    BenchmarkMetrics,
    BenchmarkResult,
    OperationAttribute,
    check_metric_dependencies,
)
from conftest import Config

torch_backend_device = flag_gems.runtime.torch_backend_device
torch_device_fn = flag_gems.runtime.torch_device_fn
device = flag_gems.device
torch_backend_device.matmul.allow_tf32 = False


def triton_testing_do_bench_rewritting(
    fn,
    warmup=25,
    rep=100,
    grad_to_none=None,
    quantiles=None,
    fast_flush=True,
    return_mode="mean",
    device_type="cuda",
    fixed_warmup_rep_runs=True,
    return_all_times=True,
):
    """
    This is a rewritten version of the original `triton.testing.do_bench` function.

    Benchmark the runtime of the provided function. By default, return the median runtime
    of :code:`fn` along with the 20-th and 80-th performance percentile.

    This function supports two modes for determining the number of warmup and repetition
    runs, by appending a parameter called `fixed_warmup_rep_runs`:
    1. Dynamic Mode (the original implementation of `triton.testing.do_bench`):
       Estimates the runtime of the kernel and dynamically adjusts the number of warmup and
       repetition runs based on the provided `warmup` and `rep` times (in milliseconds).
    2. Fixed Mode (default in this rewritten version, and consistent with torch's testing):
       Uses the provided `warmup` and `rep` values directly as the number of warmup and
       repetition runs.

    Please refer to the original implementation of `triton.testing.do_bench` function for
    more details:
    https://github.com/triton-lang/triton/blob/199fd8a239068318e94d39843c4c676f44883bd3/python/triton/testing.py#L162
    """
    assert return_mode in ["min", "max", "mean", "median"]

    di = torch._dynamo.device_interface.get_interface_for_device(device_type)

    fn()
    di.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=device_type)
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device=device_type)

    if not fixed_warmup_rep_runs:
        # Estimate the runtime of the function
        start_event = di.Event(enable_timing=True)
        end_event = di.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            cache.zero_()
            fn()
        end_event.record()
        di.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5
        # compute number of warmup and repeat
        n_warmup = max(1, int(warmup / estimate_ms))
        n_repeat = max(1, int(rep / estimate_ms))
    else:
        n_warmup = warmup
        n_repeat = rep
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float
    )
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_all_times:
        return times.tolist()
    return getattr(torch, return_mode)(times).item()


def remove_triton_cache():
    """
    Remove the Triton cache directory located at `~/.triton/cache`.

    This function checks if the cache directory exists. If it exists, the directory
    and its contents are deleted. If it does not exist, a message is printed to
    indicate that the directory is not found.
    """

    # Check if the cache_dir is already calculated and stored as a function attribute
    if not hasattr(remove_triton_cache, "cache_dir"):
        # Calculate and store the cache_dir path as a function attribute
        remove_triton_cache.cache_dir = os.path.expanduser("~/.triton/cache")

    cache_dir = remove_triton_cache.cache_dir

    try:
        subprocess.run(["rm", "-rf", cache_dir], check=False)
        print(f"Deleted Triton cache directory: {cache_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to delete Triton cache directory: {e}")


class NativeFlagGemsParameterIterator:
    @classmethod
    def _load_yaml(cls, nativeYAML="native.yaml"):
        """
        Load and parse the YAML file.

        Args:
            nativeYAML (str): Path to the YAML file. Defaults to "native.yaml".

        Returns:
            dict: Parsed YAML content as a dictionary.
        """
        with open(nativeYAML, "r") as file:
            return yaml.safe_load(file)

    @classmethod
    def write_to_yaml(cls, nativeYAML="native.yaml", entry="mm"):
        """
        Write the configuration of the specified entry to the output YAML file.

        Args:
            nativeYAML (str): Path to the input YAML file. Defaults to "native.yaml".
            entry (str): The entry to extract and write. Defaults to "mm".

        Raises:
            ValueError: If the specified entry is not found in the input YAML file.
        """
        # Load the input YAML file
        config = cls._load_yaml(nativeYAML)

        if entry not in config:
            raise ValueError(f"Entry '{entry}' not found in the '{nativeYAML}' file.")

        entry_config = config[entry]

        # Initialize YAML parser
        yaml = YAML()
        yaml_path = get_yaml_path()

        # Load existing YAML data or initialize an empty dictionary
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as file:
                yaml_data = yaml.load(file)
        else:
            yaml_data = {}

        # Replace or add the specified entry with the new configuration
        yaml_data[entry] = entry_config

        # Write the updated YAML data back to the file
        with open(yaml_path, "w") as file:
            yaml.dump(yaml_data, file)

        print(f"Configuration for '{entry}' has been written to {yaml_path}")


class Benchmark:
    device: str = device
    # ['latency_base', 'latency', 'speedup']
    DEFAULT_METRICS = DEFAULT_METRICS
    # [torch.float16, torch.float32, torch.bfloat16]
    DEFAULT_DTYPES = FLOAT_DTYPES
    # [
    #     (1024 * 1024 * 1024,),  # from perf
    #     (64, 64),
    #     (4096, 4096),
    #     (64, 512, 512),
    #     (1024, 1024, 1024),  # from perf
    # ]
    DEFAULT_SHAPES = DEFAULT_SHAPES
    DEFAULT_SHAPE_DESC = "M, N"
    DEFAULT_SHAPE_FILES = "core_shapes.yaml"
    """
    the base class for the operations benchmark
    """

    def __init__(
        self,
        op_name,
        torch_op,
        dtypes=None,
        is_backward=False,
        **kwargs,
    ):
        # In `test_blas_perf.py`:
        # pytest.param(
        #     "addmm",                 # self.op_name
        #     torch.addmm,             # self.torch_op
        #     addmm_input_fn,          # BlasBenchmark::input_fn
        #     marks=pytest.mark.addmm,
        # ),
        self.op_name = op_name
        # False
        if is_backward:
            self.op_name += " backward"
        self.torch_op = torch_op
        self.gems_op = None
        self.is_backward = is_backward
        self._input_iter = None

        # Theoretical supported dtypes, metrics for the operation.
        # These are set by default.

        # [torch.float16, torch.float32, torch.bfloat16]
        self.dtypes = dtypes if dtypes is not None else self.DEFAULT_DTYPES
        # ['latency_base', 'latency', 'speedup']
        self.metrics = self.DEFAULT_METRICS
        # [
        #     (1024 * 1024 * 1024,),  # from perf
        #     (64, 64),
        #     (4096, 4096),
        #     (64, 512, 512),
        #     (1024, 1024, 1024),  # from perf
        # ]
        self.shapes = self.DEFAULT_SHAPES
        # "M, N"
        self.shape_desc = self.DEFAULT_SHAPE_DESC
        # "core_shapes.yaml"
        self.shape_file = self.DEFAULT_SHAPE_FILES

        # Actual dtypes and metrics to be used in the benchmark,
        # can be influenced by user input.
        self.to_bench_dtypes = self.dtypes
        self.to_bench_metrics = self.metrics

        self.return_all_times = False

        # additional properties
        for k in kwargs:
            if hasattr(self, k):
                setattr(self, k, kwargs[k])

        self.previous_torch_op_latency = {}

    def set_metrics(self, user_desired_metrics: Optional[List[str]]):
        # Validate user-specified metrics
        if user_desired_metrics:
            invalid_metrics = [
                metric for metric in user_desired_metrics if metric not in self.metrics
            ]
            if invalid_metrics:
                raise ValueError(
                    f"Invalid metrics: {', '.join(invalid_metrics)} for operation: '{self.op_name}'"
                )
            unsatisfied_metrics = check_metric_dependencies(user_desired_metrics)
            if unsatisfied_metrics:
                raise ValueError(
                    f"Unsatisfied metric dependencies: {', '.join(unsatisfied_metrics)}"
                )

        self.to_bench_metrics = user_desired_metrics or self.metrics
        if (
            hasattr(self, "set_more_metrics")
            and callable(getattr(self, "set_more_metrics"))
            and Config.bench_level == BenchLevel.COMPREHENSIVE
            and not Config.query
        ):
            for metric in self.set_more_metrics():
                if metric not in self.to_bench_metrics:
                    self.to_bench_metrics.append(metric)

    def set_more_metrics(self):
        """Base method (optional to override in subclasses). Returns additional shapes if applicable."""
        return []

    def set_dtypes(self, user_desired_dtypes: Optional[List[torch.dtype]]):
        # Validate user-specified dtypes
        if user_desired_dtypes and not all(
            dtype in self.dtypes for dtype in user_desired_dtypes
        ):
            invalid_dtypes = [
                dtype for dtype in user_desired_dtypes if dtype not in self.dtypes
            ]
            raise ValueError(
                f"Given dtype(s) '{', '.join(str(dtype) for dtype in invalid_dtypes)}'"
                f"can't be supported by this op '{self.op_name}'"
            )
        self.to_bench_dtypes = (
            user_desired_dtypes if user_desired_dtypes else self.dtypes
        )

    def set_shapes(self, shape_file_path: Optional[List[Any]] = None):
        # Validate user-spicified shapes files
        import os

        if not os.path.isfile(shape_file_path):
            raise FileNotFoundError(f"Shape file '{shape_file_path}' does not exist.")
        try:
            with open(shape_file_path, "r") as file:
                yaml_config = yaml.safe_load(file)
                if self.op_name in yaml_config:
                    self.shapes = yaml_config[self.op_name].get(
                        "shapes", self.DEFAULT_SHAPES
                    )
                    self.shape_desc = yaml_config[self.op_name].get(
                        "shape_desc", self.DEFAULT_SHAPE_DESC
                    )
                else:
                    for cls in type(self).__mro__:
                        class_name = cls.__name__
                        if class_name in yaml_config:
                            # For `BlasBenchmark`:
                            # >>> yaml_config['BlasBenchmark'].get("shapes", [...])
                            # [[2, 4096, 4096, 4096], [16, 384, 384, 384], [16, 1024, 1024, 1024], [16, 2048, 2048, 2048], [16, 4096, 4096, 4096]]
                            # >>> yaml_config['BlasBenchmark'].get("shape_desc", "M, N")
                            # 'B, M, N, K'
                            self.shapes = yaml_config[class_name].get(
                                "shapes", self.DEFAULT_SHAPES
                            )
                            self.shape_desc = yaml_config[class_name].get(
                                "shape_desc", self.DEFAULT_SHAPE_DESC
                            )
                            break
                    else:
                        self.shapes = self.DEFAULT_SHAPES

            self.shapes = [tuple(shape) for shape in self.shapes]
            # merge shapes from subclass If subclass has `set_more_shapes`, call it to merge shapes
            if (
                # If the subclass has overwittend the `set_more_shapes` func.
                hasattr(self, "set_more_shapes")
                and callable(getattr(self, "set_more_shapes"))
                and Config.bench_level == BenchLevel.COMPREHENSIVE
                # If we set `Config.query`, we read form `core_shapes.yaml`.
                and not Config.query
            ):
                # Merge shapes using subclass-specific logic
                additional_shapes = self.set_more_shapes()
                # self.shapes = additional_shapes
                if additional_shapes:
                    self.shapes = list(dict.fromkeys(self.shapes + additional_shapes))
        except yaml.YAMLError as e:
            raise ValueError(
                f"Shape file '{shape_file_path}' is not a valid YAML file. Error: {e}"
            )

    def set_more_shapes(self) -> Optional[List[List[int]]]:
        """Base method (optional to override in subclasses). Returns additional shapes if applicable."""
        return None

    def record_shapes(self, *args, **kwargs):
        def deep_parse(item):
            if isinstance(item, torch.Tensor):
                return item.size()
            elif isinstance(item, (int, float, str, torch.dtype)):
                return item
            elif isinstance(item, (list, tuple)):
                return [deep_parse(sub_item) for sub_item in item]
            elif isinstance(item, dict):
                return {key: deep_parse(value) for key, value in item.items()}
            return None

        parsed_args = [deep_parse(arg) for arg in args]
        parsed_kwargs = {key: deep_parse(value) for key, value in kwargs.items()}
        if parsed_args and parsed_kwargs:
            return parsed_args, parsed_kwargs
        return parsed_args if parsed_args else parsed_kwargs

    def init_default_config(self):
        # "core_shapes.yaml"
        self.set_shapes(self.DEFAULT_SHAPE_FILES)

    def init_user_config(self):
        # TODO: device setting
        self.cpu_mode = Config.cpu_mode
        self.set_dtypes(Config.user_desired_dtypes)
        self.set_metrics(Config.user_desired_metrics)
        self.set_shapes(Config.shape_file)

    def set_gems(self, gems_op):
        self.gems_op = gems_op

    def get_latency(self, op, *args, **kwargs):
        fn = lambda: op(*args, **kwargs)
        if self.is_backward:
            if self.return_all_times:
                raise NotImplementedError(
                    "Not support return_all_times in `is_backward`"
                )
            out = fn()
            dout = torch.randn_like(out)
            fn = lambda: out.backward(dout, retain_graph=True)
        if Config.cpu_mode:
            if self.return_all_times:
                raise NotImplementedError("Not support return_all_times in `cpu_mode`")
            for i in range(Config.warm_up):
                fn()
            torch_device_fn.synchronize()
            start = time.time()
            for i in range(Config.repetition):
                fn()
            torch_device_fn.synchronize()
            end = time.time()
            latency = (end - start) / Config.repetition * 1000
        else:
            # triton_testing_do_bench_rewritting will return all times as
            # a list if not set `return_all_times=False`.
            latency = triton_testing_do_bench_rewritting(
                fn,
                warmup=Config.warm_up,
                rep=Config.repetition,
                return_mode="median",
                return_all_times=self.return_all_times,
            )
        # average latency in ms
        return latency

    def get_gbps(self, args, latency=None):
        # """Return the dynamic input iterator for each Operator."""
        raise NotImplementedError(
            "Each Benchmark must implement its own input iterator."
        )

    def get_tflops(self, op, *args, **kwargs):
        """This method is currently not really implemented and serves as a placeholder.
        A proper implementation will be developed in the future."""
        from torch.utils.flop_counter import FlopCounterMode

        fn = lambda: op(*args, **kwargs)
        with FlopCounterMode(display=False) as flop_counter:
            fn()
        return flop_counter.get_total_flops()

    def get_input_iter(self, dtype) -> Generator:
        # """Return the dynamic input iterator for each Operator."""
        raise NotImplementedError(
            "Each Benchmark must implement its own input iterator."
        )

    def get_inputs(self, dtype):
        if self._input_iter is None:
            self._input_iter = self.get_input_iter(dtype)
        try:
            return next(self._input_iter)
        except StopIteration:
            return None

    def unpack_to_args_kwargs(self, input_tuple: Tuple[Any, ...]):
        args = []
        kwargs = {}
        for item in input_tuple:
            if (
                isinstance(item, torch.Tensor)
                or isinstance(item, (int, float))
                or item is None
                or isinstance(item, (list, tuple))
            ):
                args.append(item)
            elif isinstance(item, dict):
                kwargs.update(item)
        if self.is_backward:
            args = [
                (
                    a.clone().requires_grad_()
                    if torch.is_tensor(a) and torch.is_floating_point(a)
                    else a
                )
                for a in args
            ]
        return args, kwargs

    def run(self, read_config_from_yaml=None):
        if Config.query:
            self.init_default_config()
            attri = OperationAttribute(
                op_name=self.op_name,
                recommended_core_shapes=self.shapes,
                shape_desc=self.shape_desc,
            )
            print(attri)
            logging.info(attri.to_dict())
            return
        self.init_user_config()
        # self.to_bench_dtypes = (
        #     user_desired_dtypes if user_desired_dtypes else self.dtypes
        # )
        for dtype in self.to_bench_dtypes:
            metrics = []
            res_read_config_from_yaml = {}
            for input in self.get_input_iter(dtype):
                metric = BenchmarkMetrics()
                try:
                    args, kwargs = self.unpack_to_args_kwargs(input)
                    metric.shape_detail = self.record_shapes(*args, **kwargs)
                    if "latency_base" in self.to_bench_metrics:
                        # shape_dtype_dict = "latency_base" + "_" + self.torch_op.__name__ + "_"
                        # for item in args:
                        #     if isinstance(item, torch.Tensor):
                        #         # print(item.shape)
                        #         shape_dtype_dict += "_".join(map(str, tuple(item.shape))) + "_"
                        #         shape_dtype_dict += item.dtype.__str__() + "_"
                        # dictkey = shape_dtype_dict.strip('"\'')
                        # if (dictkey in self.previous_torch_op_latency.keys()):
                        #     metric.latency_base = self.previous_torch_op_latency[dictkey]
                        #     logging.info("xxxxxxxxxxxx")
                        # else:
                        #     metric.latency_base = self.get_latency(
                        #         self.torch_op, *args, **kwargs
                        #     )
                        #     self.previous_torch_op_latency[dictkey] = metric.latency_base
                        #     logging.info(shape_dtype_dict+"   "+str(metric.latency_base)+"  " )
                        #     logging.info(self.previous_torch_op_latency.keys())
                        #     logging.info(dictkey in self.previous_torch_op_latency.keys())
                        #     logging.info(f"dictkey: {dictkey}, type: {type(dictkey)}")
                        #     logging.info(f"keys in dict: {self.previous_torch_op_latency.keys()}, types: {[type(key) for key in self.previous_torch_op_latency.keys()]}")
                        #     logging.info(f"dictkey repr: {repr(dictkey)}")
                        #     logging.info(f"keys in dict repr: {[repr(key) for key in self.previous_torch_op_latency.keys()]}")
                        metric.latency_base = self.get_latency(
                            self.torch_op, *args, **kwargs
                        )
                    if "latency" in self.to_bench_metrics:
                        if self.gems_op:
                            metric.latency = self.get_latency(
                                self.gems_op, *args, **kwargs
                            )
                        else:
                            with flag_gems.use_gems():
                                metric.latency = self.get_latency(
                                    self.torch_op, *args, **kwargs
                                )
                        read_config_from_yaml(res_read_config_from_yaml)
                    if "speedup" in self.to_bench_metrics:
                        if self.return_all_times:
                            metric.speedup = statistics.mean(
                                metric.latency_base
                            ) / statistics.mean(metric.latency)
                        else:
                            metric.speedup = metric.latency_base / metric.latency
                    if "gbps" in self.to_bench_metrics:
                        if self.return_all_times:
                            metric.gbps_base = self.get_gbps(
                                args, latency=statistics.mean(metric.latency_base)
                            )
                        else:
                            metric.gbps_base = self.get_gbps(
                                args, latency=metric.latency_base
                            )
                        metric.gbps = self.get_gbps(args, latency=metric.latency)
                    if "tflops" in self.to_bench_metrics:
                        if self.return_all_times:
                            metric.tflops = (
                                self.get_tflops(self.torch_op, *args, **kwargs)
                                / statistics.mean(metric.latency)
                                / 1e12
                                * 1e3
                            )
                        else:
                            metric.tflops = (
                                self.get_tflops(self.torch_op, *args, **kwargs)
                                / metric.latency
                                / 1e12
                                * 1e3
                            )
                            # utilization = metric.tflops / metric.latency / 1e12 * 1e3
                    if "latency_torch_compile" in self.to_bench_metrics:
                        metric.latency_torch_compile = self.get_latency(
                            torch.compile(self.torch_op), *args, **kwargs
                        )
                    if "latency_native_flaggems" in self.to_bench_metrics:
                        remove_triton_cache()
                        # BUG: Here, the written configs has not impact, cause the config has been
                        # loaded when the pytest started.
                        NativeFlagGemsParameterIterator.write_to_yaml()
                        # must have written the original yaml to config files and also remove the cache.
                        flag_gems.runtime.config_loader = (
                            flag_gems.runtime.ConfigLoader.reset_instance()
                        )
                        importlib.reload(sys.modules["flag_gems.ops.mm"])
                        if self.gems_op:
                            metric.latency_native_flaggems = self.get_latency(
                                self.gems_op, *args, **kwargs
                            )
                        else:
                            with flag_gems.use_gems():
                                metric.latency_native_flaggems = self.get_latency(
                                    self.torch_op, *args, **kwargs
                                )
                    if "speedup_vs_torch_compile" in self.to_bench_metrics:
                        if self.return_all_times:
                            metric.speedup_vs_torch_compile = statistics.mean(
                                metric.latency_torch_compile
                            ) / statistics.mean(metric.latency)
                        else:
                            metric.speedup_vs_torch_compile = (
                                metric.latency_torch_compile / metric.latency
                            )
                    if "speedup_vs_native_flaggems" in self.to_bench_metrics:
                        if self.return_all_times:
                            metric.speedup_vs_native_flaggems = statistics.mean(
                                metric.latency_native_flaggems
                            ) / statistics.mean(metric.latency)
                        else:
                            metric.speedup_vs_native_flaggems = (
                                metric.latency_native_flaggems / metric.latency
                            )
                except Exception as e:
                    metric.error_msg = str(e)
                    pytest.fail(str(e))  # raise exception again
                finally:
                    metrics.append(metric)
                    gc.collect()
            result = BenchmarkResult(
                level=Config.bench_level.value,
                op_name=self.op_name,
                dtype=str(dtype),
                mode="cpu" if Config.cpu_mode else device,
                result=metrics,
            )
            print(result)
            logging.info(result.to_json(res_read_config_from_yaml))


class GenericBenchmark(Benchmark):
    """
    A generic benchmark class for most of the operations.

    This class extends the Benchmark base class. It allows users to specify custom
    input functions and shapes, making it suitable for a wide range of tensor
    operations including both unary and binary operations.

    Usage example:
        benchmark = GenericBenchmark(op_name="add", torch_op=torch.add, input_fn=binary_input_fn)
        benchmark.run()
    """

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def set_more_shapes(self):
        more_shapes_1d = [
            (2**28,),
        ]
        more_shapes_2d = [(10000, 2**i) for i in (0, 8, 16)]
        more_shapes_3d = [(100, 2**i, 100) for i in (0, 8, 16)]
        return more_shapes_1d + more_shapes_2d + more_shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            yield from self.input_fn(shape, cur_dtype, self.device)


class GenericBenchmarkFilterShapes(GenericBenchmark):
    def __init__(self, exclude_dims: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exclude_dims = exclude_dims

    def set_more_shapes(self):
        shapes = super().set_more_shapes()
        if self.exclude_dims is not None:
            return [shape for shape in shapes if len(shape) != self.exclude_dims]
        return shapes


class GenericBenchmarkExcluse1D(GenericBenchmarkFilterShapes):
    """
    exclude 1d shapes
    """

    def __init__(self, *args, **kwargs):
        super().__init__(exclude_dims=1, *args, **kwargs)


class GenericBenchmarkExcluse3D(GenericBenchmarkFilterShapes):
    """
    exclude 3d shapes
    """

    def __init__(self, *args, **kwargs):
        super().__init__(exclude_dims=3, *args, **kwargs)


class GenericBenchmark2DOnly(GenericBenchmarkFilterShapes):
    """
    2d shapes only
    """

    def __init__(self, *args, **kwargs):
        super().__init__(exclude_dims=None, *args, **kwargs)

    def set_more_shapes(self):
        shapes = super().set_more_shapes()
        return [shape for shape in shapes if len(shape) == 2]


def generate_tensor_input(shape, dtype, device):
    if dtype in FLOAT_DTYPES:
        return torch.randn(shape, dtype=dtype, device=device)
    elif dtype in INT_DTYPES:
        return torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
            device=device,
        )
    elif dtype in BOOL_DTYPES:
        return torch.randint(0, 2, size=shape, dtype=dtype, device=device)


def binary_input_fn(shape, cur_dtype, device):
    inp1 = generate_tensor_input(shape, cur_dtype, device)
    inp2 = generate_tensor_input(shape, cur_dtype, device)
    yield inp1, inp2


def unary_input_fn(shape, cur_dtype, device):
    yield generate_tensor_input(shape, cur_dtype, device),


def get_yaml_path():
    """
    This func returns the absolute path of `tune_configs.yaml`, please note that
    this is not in the package path of miniconda, but in the source code path.

    Raises:
        FileNotFoundError: If the file does not exist.
    """

    yaml_path = os.path.abspath(
        "../../src/flag_gems/runtime/backend/_nvidia/tune_configs.yaml"
    )

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"The file `{yaml_path}` does not exist.")

    return yaml_path


def generate_triton_config_only_single_config(
    block_m, block_n, block_k, split_k, num_stages, num_warps, num_ctas
):
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
        str: A configuration string that should be written into tune_configs.yaml
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


def generate_triton_config(
    block_m, block_n, block_k, split_k, num_stages, num_warps, num_ctas
):
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
        str: A configuration string that should be written into tune_configs.yaml
    """
    config = f"""
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
    """
    A class to iterate over all combinations of parameter values within specified ranges.

    This class is designed to generate and iterate through all possible combinations of
    parameter values based on the provided ranges. Each parameter can have its own range
    defined as a tuple or list, and the class supports both regular ranges and ranges
    restricted to powers of 2.

    Attributes:
        ranges (dict): A dictionary storing the range for each parameter.
        current_indices (dict): A dictionary storing the current index for each parameter.
        keys (list): A list of parameter names.
        total_combinations (int): The total number of combinations of parameter values.
        current_combination (int): The current combination index being iterated over.

    Methods:
        __init__: Initializes the ParameterIterator with ranges for each parameter.
        get_next_val: Returns the next combination of parameter values.
        reset: Resets the iterator to the first combination.
    """

    def __init__(
        self,
        block_m_range,
        block_n_range,
        block_k_range,
        split_k_range,
        num_stages_range,
        num_warps_range,
        num_ctas_range,
    ):
        """
        Initialize the ParameterIterator with ranges for each parameter.

        Args:
            block_m_range: Range for the `block_m` parameter.
            block_n_range: Range for the `block_n` parameter.
            block_k_range: Range for the `block_k` parameter.
            split_k_range: Range for the `split_k` parameter.
            num_stages_range: Range for the `num_stages` parameter.
            num_warps_range: Range for the `num_warps` parameter.
            num_ctas_range: Range for the `num_ctas` parameter.

        Each range should be defined as a tuple or list:
        - If the range has 3 elements: (start_val, end_val, step).
        - If the range has 4 elements: (start_val, end_val, step, bool_power_of_2).
          - If `bool_power_of_2` is True, the range will only include powers of 2.
        """
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
        self.current_indices = {key: 0 for key in self.ranges}

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
        """
        Get the next combination of parameter values.

        Returns:
            A dictionary containing the current combination of parameter values.

        Raises:
            StopIteration: If all combinations have been iterated over.
        """

        # Check if all combinations have been exhausted
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

        # Increment the combination counter
        self.current_combination += 1
        return result

    def reset(self):
        """
        Reset the iterator to the first combination.
        """
        self.current_combination = 0

    def write_next_yaml_only_single_config(self):
        try:
            params = self.get_next_val()
            # params is like {'block_m': 3, 'block_n': 3, 'block_k': 3, 'split_k': 3, 'num_stages': 3, 'num_warps': 3, 'num_ctas': 3}
            block_m, block_n, block_k, split_k, num_stages, num_warps, num_ctas = (
                params["block_m"],
                params["block_n"],
                params["block_k"],
                params["split_k"],
                params["num_stages"],
                params["num_warps"],
                params["num_ctas"],
            )
            new_config = generate_triton_config_only_single_config(
                block_m, block_n, block_k, split_k, num_stages, num_warps, num_ctas
            )
            with open(get_yaml_path(), "w") as file:
                file.write(new_config)
            return True
        except StopIteration as e:
            print(e)
            return False

    def write_next_yaml(self, entry="mm"):
        """
        Write the next configuration to the YAML file, replacing the specified entry.

        Args:
            entry (str, optional): The entry in the YAML file to replace. Defaults to "mm".

        Returns:
            bool: True if the operation was successful, False if all configurations have been iterated over.
        """
        try:
            # Get the next parameter combination
            params = self.get_next_val()
            block_m, block_n, block_k, split_k, num_stages, num_warps, num_ctas = (
                params["block_m"],
                params["block_n"],
                params["block_k"],
                params["split_k"],
                params["num_stages"],
                params["num_warps"],
                params["num_ctas"],
            )

            # Generate the new configuration
            new_config = generate_triton_config(
                block_m, block_n, block_k, split_k, num_stages, num_warps, num_ctas
            )

            # Load the YAML file
            yaml = YAML()
            yaml_path = get_yaml_path()

            if os.path.exists(yaml_path):
                with open(yaml_path, "r") as file:
                    yaml_data = yaml.load(file)
            else:
                yaml_data = {}

            # Replace the specified entry with the new configuration
            yaml_data[entry] = yaml.load(new_config)
            # import warnings
            # warnings.warn(f"{yaml_data[entry]}", UserWarning)

            # Write the updated YAML data back to the file
            with open(yaml_path, "w") as file:
                yaml.dump(yaml_data, file)

            print(f"Updated YAML file with new configuration for entry '{entry}'.")
            return True

        except StopIteration as e:
            print(e)
            return False


class MMShapeGenerator:
    def __init__(self, start, end, step, num=0):
        """
        Initializes the MMShapeGenerator with ranges for M, N, and K.
        :param start: Start of the range (inclusive) for M, N, K.
        :param end: End of the range (exclusive) for M, N, K.
        :param step: Step size for the range of M, N, K.
        """
        self.start = start
        self.end = end
        self.step = step
        self.num = num

    def generate_fixed_step_shapes(self):
        """
        Generate all combinations of (M, N, K) within the specified ranges.
        :return: A list of [M, N, K] combinations.
        """
        # Create ranges for M, N, K
        m_range = range(self.start, self.end + 1, self.step)
        n_range = range(self.start, self.end + 1, self.step)
        k_range = range(self.start, self.end + 1, self.step)

        # Generate the Cartesian product of all combinations for M, N, K.
        shapes = list(itertools.product(m_range, n_range, k_range))

        # Convert tuples to lists for output consistency
        shapes = [list(shape) for shape in shapes]
        return shapes

    def generate_random_shapes(self, exclude_shapes=None):
        """
        Generate `num` random combinations of (M, N, K) within the specified ranges.
        :param exclude_shapes: A list of shapes to exclude from the random generation.
        :return: A list of [M, N, K] combinations.
        """
        if self.num is None:
            raise ValueError(
                "Number of random shapes must be provided for random generation."
            )

        if exclude_shapes is None:
            exclude_shapes = []

        # Convert exclude_shapes to a set of tuples for faster lookup
        exclude_set = set(tuple(shape) for shape in exclude_shapes)

        # Generate random shapes
        shapes = []
        while len(shapes) < self.num:
            m = random.randint(self.start, self.end)
            n = random.randint(self.start, self.end)
            k = random.randint(self.start, self.end)

            if m % 16 != 0:
                continue
            if n % 16 != 0:
                continue
            if k % 16 != 0:
                continue

            shape = (m, n, k)

            # Ensure the shape is not in the exclude list
            if shape not in exclude_set:
                shapes.append(list(shape))
                exclude_set.add(shape)  # Prevent duplicates

        return shapes

    def save_to_yaml(
        self, filename="configs/mm_shape.yaml", generate_fixed_shapes=True
    ):
        """
        Save the generated shapes into a YAML file.
        :param filename: Name of the YAML file to save the shapes.
        """

        if generate_fixed_shapes:
            shapes = self.generate_fixed_step_shapes()
        else:
            exclude_shapes = self.generate_fixed_step_shapes()
            shapes = self.generate_random_shapes(exclude_shapes)

        data = {"MMBenchmark": {"shapes": shapes, "shape_desc": "M, N, K"}}

        # Customize the Dumper to make sure the list is output in a flat format
        class CustomDumper(yaml.SafeDumper):
            def increase_indent(self, flow=False, indentless=False):
                return super(CustomDumper, self).increase_indent(flow, False)

        # Dump with PyYAML and ensure proper `[a, b, c]` formatting
        with open(filename, "w") as f:
            yaml.dump(
                data,
                f,
                Dumper=CustomDumper,
                default_flow_style=None,
                allow_unicode=True,
                sort_keys=False,
                indent=2,
            )

    _iterated_shapes_static = None

    def iterShapeOneByOne(
        self, filename="configs/mm_shape.yaml", generate_fixed_shapes=True
    ):
        if MMShapeGenerator._iterated_shapes_static is None:
            if generate_fixed_shapes:
                shapes = self.generate_fixed_step_shapes()
            else:
                exclude_shapes = self.generate_fixed_step_shapes()
                shapes = self.generate_random_shapes(exclude_shapes)
            MMShapeGenerator._iterated_shapes_static = shapes

        if MMShapeGenerator._iterated_shapes_static:
            # shape = MMShapeGenerator._iterated_shapes_static.pop(0)
            index = random.randrange(len(MMShapeGenerator._iterated_shapes_static))
            shape = MMShapeGenerator._iterated_shapes_static.pop(index)
            shapeM, shapeN, shapeK = shape[0], shape[1], shape[2]

            data = {"MMBenchmark": {"shapes": [shape], "shape_desc": "M, N, K"}}

            # Customize the Dumper to make sure the list is output in a flat format
            class CustomDumper(yaml.SafeDumper):
                def increase_indent(self, flow=False, indentless=False):
                    return super(CustomDumper, self).increase_indent(flow, False)

            # Dump with PyYAML and ensure proper `[a, b, c]` formatting
            with open(filename, "w") as f:
                yaml.dump(
                    data,
                    f,
                    Dumper=CustomDumper,
                    default_flow_style=None,
                    allow_unicode=True,
                    sort_keys=False,
                    indent=2,
                )

            return True, shapeM, shapeN, shapeK
        else:
            print("No shapes left to iterate.")
            return False, None, None, None


class TunedParameterFunctions:
    """
    A class to encapsulate the functions used to generate parameters for the ParameterGenerator.

    Attributes:
        block_m_func (callable): Function to generate block_m.
        block_k_func (callable): Function to generate block_k.
        block_n_func (callable): Function to generate block_n.
        split_k_func (callable): Function to generate split_k.
        num_stages_func (callable): Function to generate num_stages.
        num_warps_func (callable): Function to generate num_warps.
        num_ctas_func (callable): Function to generate num_ctas.
    """

    def __init__(
        self,
        block_m_func: callable,
        block_k_func: callable,
        block_n_func: callable,
        split_k_func: callable,
        num_stages_func: callable,
        num_warps_func: callable,
        num_ctas_func: callable,
    ):
        """
        Initializes the TunedParameterFunctions with the provided functions.

        Args:
            block_m_func (callable): Function to generate block_m.
            block_k_func (callable): Function to generate block_k.
            block_n_func (callable): Function to generate block_n.
            split_k_func (callable): Function to generate split_k.
            num_stages_func (callable): Function to generate num_stages.
            num_warps_func (callable): Function to generate num_warps.
            num_ctas_func (callable): Function to generate num_ctas.
        """
        self.block_m_func = block_m_func
        self.block_k_func = block_k_func
        self.block_n_func = block_n_func
        self.split_k_func = split_k_func
        self.num_stages_func = num_stages_func
        self.num_warps_func = num_warps_func
        self.num_ctas_func = num_ctas_func


class TunedParameterGenerator:
    """
    A class to generate parameters (`block_m`, `block_k`, `block_n`, `split_k`, `num_stages`,
    `num_warps`, `num_ctas`) based on input shapes (`shapeM`, `shapeK`, `shapeN`) and additional
    function parameters.

    This class encapsulates the logic for generating the 6 parameters using 6 function parameters.
    The generated parameters are used for further computation or optimization.

    Attributes:
        shapeM (int): The first input shape dimension.
        shapeK (int): The second input shape dimension.
        shapeN (int): The third input shape dimension.
        param_functions (TunedParameterFunctions): An instance of TunedParameterFunctions containing
                                                   the functions to generate the parameters.

    Methods:
        __init__: Initializes the TunedParameterGenerator with input shapes and function parameters.
        generate_parameters: Generates and returns the 6 parameters based on the input shapes and functions.
        write_to_yaml: Write the generated configuration to the YAML file, replacing the specified entry.
    """

    def __init__(
        self,
        shapeM: int,
        shapeK: int,
        shapeN: int,
        param_functions: TunedParameterFunctions,
    ):
        """
        Initializes the TunedParameterGenerator with input shapes and function parameters.

        Args:
            shapeM (int): The first input shape dimension.
            shapeK (int): The second input shape dimension.
            shapeN (int): The third input shape dimension.
            param_functions (TunedParameterFunctions): An instance of TunedParameterFunctions containing
                                                       the functions to generate the parameters.
        """
        self.shapeM = shapeM
        self.shapeK = shapeK
        self.shapeN = shapeN
        self.param_functions = param_functions
        self.block_m_func = self.param_functions.block_m_func
        self.block_k_func = self.param_functions.block_k_func
        self.block_n_func = self.param_functions.block_n_func
        self.split_k_func = self.param_functions.split_k_func
        self.num_stages_func = self.param_functions.num_stages_func
        self.num_warps_func = self.param_functions.num_warps_func
        self.num_ctas_func = self.param_functions.num_ctas_func

    def generate_parameters(self):
        """
        Generates and returns the 6 parameters based on the input shapes and functions.

        Returns:
            dict: A dictionary containing the generated parameters:
                - block_m (int): Generated block_m value.
                - block_k (int): Generated block_k value.
                - block_n (int): Generated block_n value.
                - split_k (int): Generated split_k value.
                - num_stages (int): Generated num_stages value.
                - num_warps (int): Generated num_warps value.
                - num_ctas (int): Generated num_ctas value.
        """
        block_m = self.block_m_func(self.shapeM, self.shapeK, self.shapeN)
        block_k = self.block_k_func(self.shapeM, self.shapeK, self.shapeN)
        block_n = self.block_n_func(self.shapeM, self.shapeK, self.shapeN)
        split_k = self.split_k_func(self.shapeM, self.shapeK, self.shapeN)
        num_stages = self.num_stages_func(self.shapeM, self.shapeK, self.shapeN)
        num_warps = self.num_warps_func(self.shapeM, self.shapeK, self.shapeN)
        num_ctas = self.num_ctas_func(self.shapeM, self.shapeK, self.shapeN)

        return {
            "block_m": block_m,
            "block_k": block_k,
            "block_n": block_n,
            "split_k": split_k,
            "num_stages": num_stages,
            "num_warps": num_warps,
            "num_ctas": num_ctas,
        }

    def write_to_yaml(self, entry="mm"):
        """
        Write the generated configuration to the YAML file, replacing the specified entry.

        Args:
            entry (str, optional): The entry in the YAML file to replace. Defaults to "mm".

        Returns:
            bool: True if the operation was successful, False if all configurations have been iterated over.
        """
        try:
            # Get the next parameter combination
            params = self.generate_parameters()
            block_m, block_n, block_k, split_k, num_stages, num_warps, num_ctas = (
                params["block_m"],
                params["block_n"],
                params["block_k"],
                params["split_k"],
                params["num_stages"],
                params["num_warps"],
                params["num_ctas"],
            )

            # Generate the new configuration
            new_config = generate_triton_config(
                block_m, block_n, block_k, split_k, num_stages, num_warps, num_ctas
            )

            # Load the YAML file
            yaml = YAML()
            yaml_path = get_yaml_path()

            if os.path.exists(yaml_path):
                with open(yaml_path, "r") as file:
                    yaml_data = yaml.load(file)
            else:
                yaml_data = {}

            # Replace the specified entry with the new configuration
            yaml_data[entry] = yaml.load(new_config)
            # import warnings
            # warnings.warn(f"{yaml_data[entry]}", UserWarning)

            # Write the updated YAML data back to the file
            with open(yaml_path, "w") as file:
                yaml.dump(yaml_data, file)

            print(f"Updated YAML file with new configuration for entry '{entry}'.")
            return True

        except StopIteration as e:
            print(e)
            return False


def archive_file_with_timestamp(file_path, archive_dir="archive"):
    """
    Archive a file by renaming it with a timestamp and moving it to an archive directory
    located in the same directory as the original file.

    Args:
        file_path (str): The path of the file to be archived.
        archive_dir (str, optional): The directory where the archived file will be moved.
                                     Defaults to "archive".

    Returns:
        str: The new file path if the file was archived, otherwise None.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist, no need to archive.")
        return None

    # Get the directory of the file
    file_dir = os.path.dirname(file_path)

    # Construct the full path for the archive directory
    full_archive_dir = os.path.join(file_dir, archive_dir)

    # Create the archive directory if it doesn't exist
    os.makedirs(full_archive_dir, exist_ok=True)

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract the file name and extension
    file_name, file_extension = os.path.splitext(os.path.basename(file_path))

    # Construct the new file name with a timestamp
    new_file_name = f"{file_name}_{timestamp}{file_extension}"

    # Construct the full path for the archived file
    new_file_path = os.path.join(full_archive_dir, new_file_name)

    # Rename (move) the file to the archive directory
    os.rename(file_path, new_file_path)
    print(f"Archived {file_path} to {new_file_path}")

    return new_file_path


def run_perf_pytest(
    operation,
    shape_file,
    level="core",
    warmup=5,
    iter=5,
    dtypes="float16",
    log="log",
    verbose=False,
):
    """
    Run a performance test using pytest with the specified parameters.

    Args:
        operation (str): The operation to test (e.g., "mm" for matrix multiplication).
        level (str, optional): The test level. Defaults to "core".
        warmup (int, optional): The number of warmup iterations. Defaults to 1.
        iter (int, optional): The number of test iterations. Defaults to 5.
        dtypes (str, optional): The data types to test. Defaults to "float16".
        shape_file (str, optional): The path to the shape configuration file. Defaults to "configs/mm_shape.yaml".
        log (str, optional): The log file name. Defaults to "log".
    """
    # Construct the command as a list of arguments
    if verbose:
        cmd = [
            "pytest",
            "-s",
            f"test_{operation}_perf.py",
            "--level",
            level,
            "--warmup",
            str(warmup),
            "--iter",
            str(iter),
            "--dtypes",
            dtypes,
            "--shape_file",
            shape_file,
            "--record",
            log,
        ]
    else:
        cmd = [
            "pytest",
            f"test_{operation}_perf.py",
            "--level",
            level,
            "--warmup",
            str(warmup),
            "--iter",
            str(iter),
            "--dtypes",
            dtypes,
            "--shape_file",
            shape_file,
            "--record",
            log,
        ]

    # Run the command
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        print("Performance test completed successfully.")
    else:
        print("Performance test failed.")

    filename = "result-{}.log".format("_".join(cmd))
    filename = filename.replace("pytest_-", "").replace(".py_", "")
    filename = filename.replace("_-", "-").replace("/", "_")
    return filename
