import ast
import gc
import glob
import importlib
import inspect
import itertools
import logging
import os
import random
import statistics
import subprocess
import sys
import textwrap
import time
import warnings
from datetime import datetime
from typing import Any, Generator, List, Optional, Tuple

import pandas as pd
import pytest
import torch
import yaml
from ruamel.yaml import YAML
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../src"))
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


# Example Format of mm op: {(dtype, M, K, N): optimal latency}
optimal_latency_of_native_flaggems = {}
# Example Format of mm op: [(dtype, M, K, N, blk_m, ...)]
have_itered_shape_config_pairs = []
# Store the local key for this test.
localkey_during_this_test = None
# Store the data type of this test.
data_type = None


def read_native_flaggems_from_trainset(
    tarinSetPath="./train-set",
    datatype="float16",
    config=None,
):
    global optimal_latency_of_native_flaggems
    global have_itered_shape_config_pairs
    global localkey_during_this_test
    global data_type

    if config is None:
        config = {
            "dtype_col": "dtype",
            "shape_cols": ["shape_detail_M", "shape_detail_K", "shape_detail_N"],
            "config_cols": [
                "BLOCK_M",
                "BLOCK_N",
                "BLOCK_K",
                "SPLIT_K",
                "num_stages",
                "num_warps",
            ],
            "latency_col": "latency",
        }

        warning_message = textwrap.dedent(
            f"""
            `config` in `read_native_flaggems_from_trainset` has not been specified.
            Using default `config` for matrix multiplication (mm) operation instead:
            {config}
        """
        ).strip()
        warnings.warn(warning_message, UserWarning)

    # TODO: Change var name from "xlsx" to "csv"
    xlsx_files = glob.glob(os.path.join(tarinSetPath, "*.csv"))

    if (
        len(optimal_latency_of_native_flaggems) == 0
        or len(have_itered_shape_config_pairs) == 0
        or localkey_during_this_test is None
        or data_type is None
    ) and (not len(xlsx_files) == 0):
        # for xlsxpath in xlsx_files:
        for xlsxpath in tqdm(xlsx_files, desc="Processing Excel files"):
            data = pd.read_csv(xlsxpath)

            for index, row in tqdm(
                data.iterrows(),
                desc=f"    Processing rows in {os.path.basename(xlsxpath)}",
                leave=False,
                total=len(data),
            ):
                dtype = row[config["dtype_col"]]
                if datatype in dtype:
                    shape_cols = [int(row[col]) for col in config["shape_cols"]]
                    config_cols = [int(row[col]) for col in config["config_cols"]]
                    latency = row[config["latency_col"]]
                    continue_flag = False
                    if (
                        isinstance(latency, str)
                        and latency.startswith("[")
                        and latency.endswith("]")
                    ):
                        try:
                            latency_list = ast.literal_eval(latency)
                            if isinstance(latency_list, list):
                                latency = statistics.mean(latency_list)
                            else:
                                continue_flag = True
                        except:
                            continue_flag = True
                    else:
                        try:
                            latency = float(latency)
                        except ValueError:
                            continue_flag = True
                    localkey = tuple([dtype] + shape_cols)
                    localkey_during_this_test = localkey
                    data_type = dtype
                    have_itered_shape_config_pairs.append(
                        tuple([dtype] + shape_cols + config_cols)
                    )
                    if not continue_flag:
                        if localkey not in optimal_latency_of_native_flaggems:
                            optimal_latency_of_native_flaggems[localkey] = latency
                        else:
                            if latency < optimal_latency_of_native_flaggems[localkey]:
                                optimal_latency_of_native_flaggems[localkey] = latency


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
    raise RuntimeError(
        "`remove_triton_cache` is deprecated and will be removed in a future version. "
        "This is because we have found that Triton's cache only stores the compilation "
        "information of the operation, and does not store the configuration of the "
        "optimal latency of each shape. So continuing to use this function will only "
        "increase the cost of testing.",
    )

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


def read_config_from_yaml(op_name):
    """
    Read the configuration for the specified `op_name` from the YAML file and return it as a dictionary.

    Args:
        op_name (str): The operation name (e.g., "mm", "bmm") to read from the YAML file.

    Returns:
        dict: A dictionary containing the configuration for the specified `op_name`.
    """
    return RuntimeFlagGemsConfigReader.read_config_from_yaml(op_name)


def generate_config_from_dict(config_dict):
    """
    Generate a YAML configuration string from a dictionary, using the structure of the original YAML file.

    Args:
        config_dict (dict): A dictionary containing the configuration fields.

    Returns:
        str: A YAML-formatted string representing the configuration.
    """
    return RuntimeFlagGemsConfigReader.generate_config_from_dict(config_dict)


def write_config_to_yaml(config_dict):
    """
    Write the updated configuration back to the YAML file. So that users can build their
    own `config_dict` and the format of `config_dict` can gotten using `read_config_from_yaml`.

    Args:
        config_dict (dict): A dictionary containing the configuration fields to update.
    """
    return RuntimeFlagGemsConfigReader.write_config_to_yaml(config_dict)


def write_shapes_to_yaml(shapes, excel_config, output_path="configs/shape.yaml"):
    """
    Write the provided shapes to a YAML file using a template-based structure. This function
    is a wrapper around `ShapeYAMLWriter.write_shapes_to_yaml`, providing a simplified interface
    for writing shape configurations to a YAML file.

    Args:
        shapes (list of dict): A list of shape dictionaries, each containing keys
                               defined in `excel_config["shape_cols"]`.
        excel_config (dict): A dictionary containing configuration details, including
                             `bench_name`, `shape_cols`, and `shape_desc`.
        output_path (str): The path to the output YAML file. Defaults to "shapes.yaml".
    """
    return ShapeYAMLWriter.write_shapes_to_yaml(shapes, excel_config, output_path)


class ShapeYAMLWriter:
    """
    A class to write shape configurations to a YAML file using a template-based approach.
    The YAML file structure is dynamically generated based on the provided template and input data.
    All methods are classmethods to avoid the need for instantiation.
    """

    @classmethod
    def write_shapes_to_yaml(
        cls, shapes, excel_config, output_path="configs/shape.yaml"
    ):
        """
        Write the provided shapes to a YAML file using a template-based structure.

        Args:
            shapes (list of dict): A list of shape dictionaries, each containing keys
                                   defined in `excel_config["shape_cols"]`.
            excel_config (dict): A dictionary containing configuration details, including
                                 `bench_name`, `shape_cols`, and `shape_desc`.
            output_path (str): The path to the output YAML file. Defaults to "shapes.yaml".

        Returns:
            None
        """
        # Extract benchmark name from excel_config
        bench_name = excel_config.get("bench_name", None)
        if bench_name is None:
            raise ValueError("`bench_name` must be provided in `excel_config`.")

        # Extract shape columns from excel_config
        shape_cols = excel_config.get("shape_cols", [])
        if not shape_cols:
            raise ValueError("`shape_cols` must be provided in `excel_config`.")

        # Prepare the YAML structure based on the template
        yaml_data = {
            bench_name: {
                "shapes": [
                    [
                        shape[col] for col in shape_cols
                    ]  # Dynamically extract shape values
                    for shape in shapes
                ]
            }
        }

        # Add shape_desc if provided in excel_config
        shape_desc = excel_config.get("shape_desc", None)
        if shape_desc is None:
            raise ValueError("`shape_desc` must be provided in `excel_config`.")

        yaml_data[bench_name]["shape_desc"] = ", ".join(shape_desc)

        # Write the YAML data to the file
        with open(output_path, "w") as file:
            yaml.dump(yaml_data, file, default_flow_style=None)

        print(f"Shapes have been written to {output_path}.")


class RuntimeFlagGemsConfigReader:
    """
    Read the configuration for the specified `op_name` from the runtime changed YAML file
    and return it as a dictionary. This returned dict can be as a key in Excel files.
    """

    # The cache of the config format.
    _cache_config_format = None
    _cache_op_name = None

    @classmethod
    def read_config_from_yaml(cls, op_name):
        """
        Read the configuration for the specified `op_name` from the YAML file and return
        it as a flattened dictionary. Filters out non-numeric values in `param_map`,
        selects the first value for list-type fields, and removes all prefixes from key
        names (e.g., "META.TILE_M" becomes "TILE_M").

        Args:
            op_name (str): The operation name (e.g., "mm", "bmm", "attention") to read
                           from the YAML file.

        Returns:
            dict: A flattened dictionary containing all configuration fields for the
                  specified `op_name`. For list-type fields, only the first value is
                  selected, and key names are simplified.

        Raises:
            ValueError: If the YAML file does not contain the specified `op_name` or has
                        an invalid structure.
        """
        yaml_path = cls._get_yaml_path()
        print(yaml_path)

        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

            # Check if the specified `op_name` exists in the YAML file
            if op_name not in config:
                raise ValueError(
                    f"Operation '{op_name}' not found in the YAML file: {yaml_path}"
                )

            # Get the list of configurations for the specified `op_name`
            op_configs = config[op_name]
            if not isinstance(op_configs, list) or len(op_configs) == 0:
                raise ValueError(
                    f"Invalid structure for operation '{op_name}' in the YAML file: {yaml_path}"
                )

            # Extract the first configuration
            first_config = op_configs[0]

            # Cache the config format
            if cls._cache_config_format is None:
                cls._cache_config_format = first_config
                cls._cache_op_name = op_name

            # Flatten the configuration into a dictionary, filtering non-numeric values in `param_map`
            result_dict = cls._flatten_config(first_config)
            result_dict = {"autotune_configs": result_dict}
            return result_dict

    @classmethod
    def generate_config_from_dict(cls, config_dict):
        """
        Generate a YAML configuration string from a dictionary, using the structure of the original YAML file.

        Args:
            config_dict (dict): A dictionary containing the configuration fields.

        Returns:
            str: A YAML-formatted string representing the configuration.
        """
        if cls._cache_config_format is None:
            raise ValueError(
                "No cached config format found. Call `read_config_from_yaml` first."
            )

        # Update the configuration with the provided dictionary
        updated_config = cls._update_config(cls._cache_config_format, config_dict)

        # Convert the updated configuration to a YAML string
        yaml = YAML()
        yaml.preserve_quotes = True  # Preserve quotes in the original YAML file

        from io import StringIO

        string_stream = StringIO()
        yaml.dump({cls._cache_op_name: [updated_config]}, string_stream)
        yaml_config = string_stream.getvalue()

        return yaml_config

    @classmethod
    def write_config_to_yaml(cls, config_dict):
        """
        Write the updated configuration back to the YAML file.

        Args:
            config_dict (dict): A dictionary containing the configuration fields to update.

        Returns:
            None
        """
        if cls._cache_config_format is None:
            raise ValueError(
                "No cached config format found. Call `read_config_from_yaml` first."
            )

        # Update the configuration with the provided dictionary
        updated_config = cls._update_config(cls._cache_config_format, config_dict)

        # Initialize YAML parser
        yaml = YAML()
        yaml.preserve_quotes = True  # Preserve quotes in the original YAML file
        yaml_path = cls._get_yaml_path()

        # Load existing YAML data or initialize an empty dictionary
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as file:
                yaml_data = yaml.load(file)
        else:
            yaml_data = {}

        # Replace or add the specified entry with the new configuration
        yaml_data[cls._cache_op_name] = [updated_config]

        # Write the updated YAML data back to the file
        with open(yaml_path, "w") as file:
            yaml.dump(yaml_data, file)

        print(
            f"Updated Configuration for '{cls._cache_op_name}' has been written to {yaml_path}"
        )

    @classmethod
    def _update_config(cls, config, config_dict):
        """
        Update a configuration dictionary with values from another dictionary, preserving
        the list structures.

        Args:
            config (dict): The original configuration dictionary.
            config_dict (dict): A dictionary containing the new values.

        Returns:
            dict: The updated configuration dictionary.
        """
        for key, value in config_dict.items():
            if key in config:
                # If the original value is a list, ensure the new value is also a list
                if isinstance(config[key], list):
                    config[key] = [value] if not isinstance(value, list) else value
                else:
                    config[key] = value
            elif isinstance(config, dict):
                for sub_key, sub_value in config.items():
                    if isinstance(sub_value, dict):
                        cls._update_config(sub_value, config_dict)
        return config

    @classmethod
    def _flatten_config(cls, config, parent_key="", sep="."):
        """
        Flatten a nested dictionary into a single-level dictionary.
        Filters out non-numeric values in `param_map`, selects the first value for list-type
        fields, and removes all prefixes from key names.

        Args:
            config (dict): The nested dictionary to flatten.
            parent_key (str): The base key for nested keys (used internally for recursion).
            sep (str): The separator used to join nested keys.

        Returns:
            dict: A flattened dictionary with simplified key names.
        """
        items = {}
        for key, value in config.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                if key == "param_map":
                    # Recursively filter out non-numeric values in `param_map`
                    filtered_param_map = cls._filter_numeric_values(value)
                    items.update(filtered_param_map)
                else:
                    items.update(cls._flatten_config(value, new_key, sep=sep))
            elif isinstance(value, list) and len(value) > 0:
                # For list-type fields, select the first value
                final_key = new_key.split(sep)[-1]  # Remove all prefixes
                items[final_key] = value[0]
            else:
                # Add the key-value pair to the result, removing all prefixes
                final_key = new_key.split(sep)[-1]  # Remove all prefixes
                items[final_key] = value
        return items

    @classmethod
    def _filter_numeric_values(cls, config):
        """
        Recursively filter out non-numeric values in a dictionary.

        Args:
            config (dict): The dictionary to filter.

        Returns:
            dict: A dictionary containing only numeric values.
        """
        result = {}
        for key, value in config.items():
            if isinstance(value, dict):
                # Recursively filter nested dictionaries
                nested_result = cls._filter_numeric_values(value)
                if nested_result:
                    result.update(nested_result)
            elif isinstance(value, (int, float)):
                # Keep numeric values
                result[key] = value
        return result

    @classmethod
    def _get_yaml_path(cls):
        """
        Get the path to the YAML file.

        Returns:
            str: The path to the YAML file.
        """
        return get_yaml_path()


class NativeFlagGemsConfigSaver:
    """
    This class load and parse the native YAML file, and Write the configuration of the
    automatically detected entry to the output YAML file. The output YAML path can be
    got from `get_yaml_path()`.

    In fact, this class intends to restore the default config of native FlagGems, if we
    want to test the performance of native FlagGems. The `nativeYAML` is actually the
    copy of the original config of native FlagGems for each operation.
    """

    # Class property, which is used to cache YAML data.
    _yaml_cache = None
    _yaml_file_path = None
    _yaml_entry = None

    @classmethod
    def _load_yaml(cls, nativeYAML):
        """
        Load and parse the YAML file, or return cached data if available.
        Automatically detects the entry in the YAML file and ensures there is only one.

        Args:
            nativeYAML (str): Path to the YAML file. Defaults to "native.yaml".

        Returns:
            dict: Parsed YAML content as a dictionary.

        Raises:
            ValueError: If the YAML file contains zero or more than one entry.
        """
        if (
            cls._yaml_cache is None
            or cls._yaml_entry is None
            or cls._yaml_file_path != nativeYAML
        ):
            with open(nativeYAML, "r") as file:
                cls._yaml_cache = yaml.safe_load(file)
                cls._yaml_file_path = nativeYAML

            # Auto-detect the identical entry of nativeYAML
            entries = list(cls._yaml_cache.keys())
            if len(entries) == 0:
                raise ValueError(f"No entry found in the YAML file: {nativeYAML}")
            elif len(entries) > 1:
                raise ValueError(
                    f"Multiple entries found in the YAML file: {entries}. "
                    "Only one entry is allowed."
                )
            else:
                cls._yaml_entry = entries[0]

    @classmethod
    def refresh_cache(cls):
        """
        Refresh the cached YAML data by reloading it from the file.
        """
        cls._yaml_cache = None
        cls._yaml_entry = None
        if cls._yaml_file_path:
            cls._load_yaml(cls._yaml_file_path)

    @classmethod
    def write_to_yaml(cls, nativeYAML="native.yaml"):
        """
        Write the configuration of the automatically detected entry to the output YAML file.

        Args:
            nativeYAML (str): Path to the input YAML file. Defaults to "native.yaml".

        Raises:
            ValueError: If the YAML file contains zero or more than one entry.
        """
        # Load the input YAML file
        cls._load_yaml(nativeYAML)

        if cls._yaml_entry not in cls._yaml_cache:
            raise ValueError(
                f"Entry '{cls._yaml_entry}' not found in the '{nativeYAML}' file."
            )

        entry_config = cls._yaml_cache[cls._yaml_entry]

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
        yaml_data[cls._yaml_entry] = entry_config

        # Write the updated YAML data back to the file
        with open(yaml_path, "w") as file:
            yaml.dump(yaml_data, file)

        print(
            f"Native Configuration for '{cls._yaml_entry}' has been written to {yaml_path}"
        )


# This is because that the opname (called A) in `benchmark/test*` and the opname
# (called B) in `tune_configs.yaml` may not equal, so this helps us transfer the
# A to B.
opname_benchmark2tuneconfig = {
    "mm": "mm",
    "scaled_dot_product_attention": "attention",
}


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

    def run(self):
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
            res_read_config_from_yaml = None
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
                        if self.op_name in opname_benchmark2tuneconfig.keys():
                            res_read_config_from_yaml = read_config_from_yaml(
                                opname_benchmark2tuneconfig[self.op_name]
                            )
                        else:
                            res_read_config_from_yaml = read_config_from_yaml(
                                self.op_name
                            )
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
                            torch.compile(
                                self.torch_op,
                                mode="max-autotune",
                                dynamic=False,
                                fullgraph=False,
                                backend="inductor",
                            ),
                            *args,
                            **kwargs,
                        )
                    if "latency_native_flaggems" in self.to_bench_metrics:
                        # remove_triton_cache()
                        # Here, the written configs has not impact, cause the config has been
                        # loaded when the pytest started.
                        NativeFlagGemsConfigSaver.write_to_yaml()
                        # So, here we have written the original yaml to config files and next
                        # we also need to reset the `config_loader` instance and reload the
                        # operations that we will test.
                        flag_gems.runtime.config_loader = (
                            flag_gems.runtime.ConfigLoader.reset_instance()
                        )
                        if self.op_name in opname_benchmark2tuneconfig.keys():
                            importlib.reload(
                                sys.modules[
                                    "flag_gems.ops."
                                    + opname_benchmark2tuneconfig[self.op_name]
                                ]
                            )
                        else:
                            importlib.reload(
                                sys.modules["flag_gems.ops." + self.op_name]
                            )
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
                    if "speedup_vs_native_flaggems_trainset" in self.to_bench_metrics:
                        # TODO: Here I have wanted to add the function which read the latency
                        # of native_flaggems from the trainset, but it has some issues, one
                        # of them is that maybe the latency from the trainset was gotten for
                        # the warmup and repeats runs being not equal to the other runtime
                        # tests, such as our method, torch, torch.compile. So, this should be
                        # ascertained whether it is feasible.

                        # BUG or not BUG: When we use trainsets to filter out repeated shape
                        # and config onfigurations, this `speedup_vs_native_flaggems_trainset`
                        # may always be null, for the reason that all tested combinations are
                        # not likely to occur in trainsets.

                        # read_native_flaggems_from_trainset(datatype=str(dtype))

                        # Example Format of mm op: {(dtype, M, K, N): optimal latency}
                        # optimal_latency_of_native_flaggems = {}
                        # Example Format of mm op: [(dtype, M, K, N, blk_m, ...)]
                        # have_itered_shape_config_pairs = []

                        localkey = localkey_during_this_test
                        if localkey in optimal_latency_of_native_flaggems.keys():
                            latencybase = optimal_latency_of_native_flaggems[localkey]

                            if self.return_all_times:
                                metric.speedup_vs_native_flaggems_trainset = (
                                    statistics.mean(latencybase) / metric.latency
                                )
                            else:
                                metric.speedup_vs_native_flaggems_trainset = (
                                    latencybase / metric.latency
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


def get_yaml_path(default_yaml_path=None):
    """
    This func returns the absolute path of `tune_configs.yaml`, please note that
    this is not in the package path of miniconda, but in the source code path.

    Args:
        default_yaml_path (object or str, optional): An object with a `yaml_path`
            attribute or a string representing the path. Defaults to None.

    Returns:
        str: The absolute path to `tune_configs.yaml`.

    Raises:
        FileNotFoundError: If the file does not exist.
    """

    # TODO: fix this using more generic configuration.
    if default_yaml_path is None:
        yaml_path = os.path.abspath(
            "../../src/flag_gems/runtime/backend/_nvidia/tune_configs.yaml"
        )
    else:
        yaml_path = default_yaml_path

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"The file `{yaml_path}` does not exist.")

    return yaml_path


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


# ===----------------------------------------------------------------------===
# User Specified Helper Function Definitions
# ===----------------------------------------------------------------------===


class ShapeGenerator:
    """
    A template class to generate shape parameter combinations based on user-defined
    functions.
    """

    def __init__(self, excel_config, shape_generators):
        """
        Initialize the ShapeGenerator.

        Args:
            excel_config (dict): The configuration dictionary containing "shape_cols".
            shape_generators (tuple): A tuple of user-defined generator functions.
                                      Function names should follow the format `gen_<field_name>`.
        """
        self.shape_cols = excel_config["shape_cols"]
        self.shape_generators = shape_generators
        self._validate_generators()

    def _validate_generators(self):
        """
        Validate that the provided generator functions match the fields in "shape_cols".
        """
        # Extract field names from generator function names
        generator_fields = [gen.__name__[4:] for gen in self.shape_generators]

        # Check if all required fields are covered
        missing_fields = set(self.shape_cols) - set(generator_fields)
        if missing_fields:
            raise ValueError(
                f"Missing generator functions for fields: {missing_fields}. "
                f"Expected functions named 'gen_<field_name>'."
            )

        # Check if there are any extra fields
        extra_fields = set(generator_fields) - set(self.shape_cols)
        if extra_fields:
            raise ValueError(
                f"Extra generator functions for fields: {extra_fields}. "
                f"These fields are not in 'shape_cols'."
            )

    def generate(self):
        """
        Generate all possible combinations of shape parameters.

        Returns:
            list: A list of dictionaries, where each dictionary represents a combination
                  of shape parameters.
        """
        from itertools import product

        # Map generator functions to their corresponding fields
        generator_map = {gen.__name__[4:]: gen for gen in self.shape_generators}

        # Generate values for each key
        values = {}
        for key in self.shape_cols:
            values[key] = generator_map[key]()

        # Generate all combinations
        combinations = []
        for combination in product(*values.values()):
            param_dict = {
                key: value for key, value in zip(self.shape_cols, combination)
            }
            combinations.append(param_dict)

        return combinations


class ConfigGenerator:
    """
    A template class to generate configuration parameter combinations based on user-defined functions.
    """

    def __init__(self, excel_config, config_generators):
        """
        Initialize the ConfigGenerator.

        Args:
            excel_config (dict): The configuration dictionary containing "config_cols".
            config_generators (tuple): A tuple of user-defined generator functions.
                                      Function names should follow the format `gen_<field_name>`.
        """
        self.config_cols = excel_config["config_cols"]
        self.config_generators = config_generators
        self._validate_generators()

    def _validate_generators(self):
        """
        Validate that the provided generator functions match the fields in "config_cols".
        """
        # Extract field names from generator function names
        generator_fields = [gen.__name__[4:] for gen in self.config_generators]

        # Check if all required fields are covered
        missing_fields = set(self.config_cols) - set(generator_fields)
        if missing_fields:
            raise ValueError(
                f"Missing generator functions for fields: {missing_fields}. "
                f"Expected functions named 'gen_<field_name>'."
            )

        # Check if there are any extra fields
        extra_fields = set(generator_fields) - set(self.config_cols)
        if extra_fields:
            raise ValueError(
                f"Extra generator functions for fields: {extra_fields}. "
                f"These fields are not in 'config_cols'."
            )

    def generate(self):
        """
        Generate all possible combinations of configuration parameters.

        Returns:
            list: A list of dictionaries, where each dictionary represents a combination
                  of configuration parameters.
        """
        from itertools import product

        # Map generator functions to their corresponding fields
        generator_map = {gen.__name__[4:]: gen for gen in self.config_generators}

        # Generate values for each key
        values = {}
        for key in self.config_cols:
            values[key] = generator_map[key]()

        # Generate all combinations
        combinations = []
        for combination in product(*values.values()):
            param_dict = {
                key: value for key, value in zip(self.config_cols, combination)
            }
            combinations.append(param_dict)

        return combinations


class ParameterMismatchError(Exception):
    """
    Custom exception for parameter mismatches.
    """

    def __init__(self, message, func_params, expected_params, mismatch_type):
        super().__init__(message)
        self.func_params = func_params
        self.expected_params = expected_params
        self.mismatch_type = mismatch_type

    def __str__(self):
        return (
            f"{self.mismatch_type}:\n"
            f"  Function parameters: {self.func_params}\n"
            f"  Expected parameters: {self.expected_params}"
        )


class TunedConfigGenerator:
    """
    A template class to generate tuned configuration parameter combinations based on user-defined
    functions.
    """

    def __init__(self, excel_config, config_generators, shapegen: ShapeGenerator):
        """
        Initialize the TunedConfigGenerator.

        Args:
            excel_config (dict): The configuration dictionary containing "config_cols".
            config_generators (tuple): A tuple of user-defined generator functions.
                                      Function names should follow the format `gen_<field_name>`.
            shapegen (ShapeGenerator): The `ShapeGenerator` instance that the configs generated for.
        """
        self.config_cols = excel_config["config_cols"]
        self.shape_cols = excel_config["shape_cols"]
        self.config_generators = config_generators
        self.shapes_list = shapegen.generate()
        self._validate_generators()

    def _validate_generators(self):
        """
        Validate that the provided generator functions match the fields in "config_cols".
        """

        # Extract field names from generator function names
        generator_fields = [gen.__name__[4:] for gen in self.config_generators]

        # Check if all required fields are covered
        missing_fields = set(self.config_cols) - set(generator_fields)
        if missing_fields:
            raise ValueError(
                f"Missing generator functions for fields: {missing_fields}. "
                f"Expected functions named 'gen_<field_name>'."
            )

        # Check if there are any extra fields
        extra_fields = set(generator_fields) - set(self.config_cols)
        if extra_fields:
            raise ValueError(
                f"Extra generator functions for fields: {extra_fields}. "
                f"These fields are not in 'config_cols'."
            )

        # Check if each generation function has the parameters that contails all of
        # items in `slef.shape_cols`.
        for func in self.config_generators:
            self._check_parameters_match(func, self.shape_cols)

    def _check_parameters_match(self, func, param_names):
        """
        Check if the parameter list of a function matches exactly with a given list of parameter
        names. If not, raise a detailed error indicating the type of mismatch.

        Args:
            func (callable): The function to check.
            param_names (list): A list of expected parameter names.

        Raises:
            ParameterMismatchError: If the parameter list does not match `param_names` exactly.
        """
        # Get the function's signature
        sig = inspect.signature(func)

        # Extract the parameter names from the signature
        func_params = list(sig.parameters.keys())

        # Check for exact match (order and names)
        exact_match = func_params == param_names
        if exact_match:
            return  # No error if exact match

        # Check for name match (ignoring order)
        name_match = set(func_params) == set(param_names)

        # Determine the type of mismatch
        if not name_match:
            raise ParameterMismatchError(
                "Parameter names do not match.",
                func_params,
                param_names,
                "Name mismatch",
            )
        else:
            raise ParameterMismatchError(
                "Parameter names match, but the order is incorrect.",
                func_params,
                param_names,
                "Order mismatch",
            )

    def generate(self):
        """
        Generate a list of tuples, where each item in this list has two items, and the first
        one is the dictionary contains a shape detail, and the second one is its corresponding
        configuration parameters. And the shape detail and its corresponding config parameters
        are all dicts.

        Returns:
            list: A list of tuples, where each tuple has two items:
                  - "shape": The shape ditails (e.g., {"shape_detail_M": 2048, ...}).
                  - "config": The generated configuration dictionary (e.g., {"BLOCK_M": 512, ...}).
        """
        result = []
        for shape in self.shapes_list:
            config = {}
            for gen_func in self.config_generators:
                # Extract the field name from the generator function name
                field_name = gen_func.__name__[4:]
                # Call the generator function with the shape parameters
                config[field_name] = gen_func(**shape)
            # Append the shape and config to the result
            result.append((shape, config))
        return result


# Convert each shape-config combination into a tuple format for comparison
def convert_to_tuple(shape_config_pair, excel_config, dtype):
    shape, config = shape_config_pair
    return tuple(
        [dtype]
        + [shape[key] for key in excel_config["shape_cols"]]
        + [config[key] for key in excel_config["config_cols"]]
    )


def run_perf_pytest(
    operation,
    shape_file,
    level="core",
    warmup=3,
    iter=3,
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
        shape_file (str, optional): The path to the shape configuration file. Defaults to "configs/shape.yaml".
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


def print_centered_label(label="START", color="green"):
    """
    Print a full-width colored line with a centered label.

    Args:
        label (str): The label to center (e.g., "START" or "END").
        color (str): The color of the line and label. Options: "green", "red", etc.
    """
    # Get terminal width
    terminal_width = os.get_terminal_size().columns

    # Define color codes
    colors = {
        "green": "\033[32m",
        "red": "\033[31m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
    }
    reset = "\033[0m"

    # Get the color code
    color_code = colors.get(color, "\033[32m")  # Default to green if color is invalid

    # Calculate the position to center the label
    label_length = len(label)
    padding = (terminal_width - label_length) // 2

    # Construct the line
    line = (
        color_code
        + "=" * padding  # Left part of the line
        + label  # Centered label
        + "=" * (terminal_width - padding - label_length)  # Right part of the line
        + reset  # Reset color
    )

    # Print the line
    print(line, end="\n\n")


stringDtype2TorchDtypeDict = {
    "float16": "torch.float16",
    "float32": "torch.float32",
    "bfloat16": "torch.bfloat16",
    "int16": "torch.int16",
    "int32": "torch.int32",
    "bool": "torch.bool",
    "cfloat": "torch.cfloat",
}


def stringDtype2TorchDtype(stringDtype: str) -> str:
    return stringDtype2TorchDtypeDict[stringDtype]
