import importlib
import itertools
import os
import subprocess
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import attri_util
from performance_utils import (
    ConfigGenerator,
    ShapeGenerator,
    archive_file_with_timestamp,
    convert_to_tuple,
    have_itered_shape_config_pairs,
    print_centered_label,
    read_config_from_yaml,
    read_native_flaggems_from_trainset,
    run_perf_pytest,
    stringDtype2TorchDtype,
    write_config_to_yaml,
    write_shapes_to_yaml,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../src"))
sys.path.append(src_dir)

# flake8: noqa: E402
import flag_gems

# ===---------------------------------------------------------------------------------===
# User-Specified Parameters
# ===---------------------------------------------------------------------------------===

pytest_operation_name = "conv2d"
# Optional["float16", "bfloat16"]
pytest_data_type = "float16"

pytest_verbose = True
pytest_warmup_runs = 3
pytest_iter_runs = 3

filter_out_repeat_comb = True

# Just don't edit this.
pytest_shape_file = "configs/shape.yaml"

# Just don't edit this.
print_shape_config_combinations = False
print_grouped_shape_config_combinations = False


# ===---------------------------------------------------------------------------------===
# Configuration Dictionary for Reading Native Flaggems from Training Set
# ===---------------------------------------------------------------------------------===
# This section defines a configuration dictionary (`excel_config`) that specifies the
# column names and structure of the Excel files containing native flaggems performance.
# The dictionary is used by the `read_native_flaggems_from_trainset` function to read and
# process the training set data. And the main purpose of this function is to prevent the
# performance of FlagGems that have already been tested from being retested, and also to
# prevent combinations of shapes and configs that have already been traversed from being
# re-traversed.
# ===---------------------------------------------------------------------------------===

# NOTE: These column names must exactly match the names in the XLSX file in the tarinset
# directory. These columns of "dtype_col" and "shape_cols" will be used as the key to
# perform the functionality introduced above.
excel_config = {
    # Data type.
    "dtype_col": "dtype",
    # Shape parameters.
    "shape_cols": [
        "shape_detail_batch",
        "shape_detail_input_c",
        "shape_detail_input_h",
        "shape_detail_input_w",
        "shape_detail_out_c",
        "shape_detail_kernel_h",
        "shape_detail_kernel_w",
        "shape_detail_stride",
        "shape_detail_padding",
        "shape_detail_groups",
    ],
    # Auto-tune configs.
    "config_cols": [
        "BLOCK_NI_HO_WO",
        "BLOCK_CI",
        "BLOCK_CO",
        "num_warps",
        "num_stages",
    ],
    # Performance.
    "latency_col": "latency",
    # Benchmark name of shape yaml.
    "bench_name": "ConvBenchmark",
    # Shape description of shape yaml. It should correspond one-to-one with "shape_cols".
    "shape_desc": [],
}
if filter_out_repeat_comb:
    read_native_flaggems_from_trainset(
        tarinSetPath="./train-set", datatype=pytest_data_type, config=excel_config
    )
config_format = read_config_from_yaml(pytest_operation_name)
print(f"Using config format of {config_format} to write configs.")


# ===---------------------------------------------------------------------------------===
# Result File Management and Archiving
# ===---------------------------------------------------------------------------------===
# This section handles the creation and archiving of result files for a given operation_name.
# The result file is named based on the operation_name and stored in the "results/" directory.
# After generating the result file, it is archived with a timestamp to preserve historical
# results and avoid overwriting previous data. This ensures that each run's results are
# uniquely stored and can be referenced later for comparison or analysis.
#
# The `archive_file_with_timestamp` function is responsible for adding a timestamp to the
# result file and moving it to an archive directory. This helps maintain a clean and
# organized record of all performance testing results.
# ===---------------------------------------------------------------------------------===

# Define the path to the result file based on the operation_name name
result_file = "results/" + pytest_operation_name + "-result.txt"

# Archive the result file with a timestamp to preserve historical results
archive_file_with_timestamp(result_file)


# ===---------------------------------------------------------------------------------===
# Shape and Configuration Parameter Generator Functions
# ===---------------------------------------------------------------------------------===
# This section contains functions that generate various parameters for shape details and
# configuration options. These parameters are used to create different configurations for
# performance testing. Each function generates a list of possible values for a specific
# parameter, which are then combined to form complete configuration sets. The purpose of
# these generators is to systematically explore the parameter space and identify optimal
# configurations for the given task.
# ===---------------------------------------------------------------------------------===


# NOTE: The function name must start with "gen_", and the second half of the name must
# correspond to the name in "Shape parameters" and "Auto-tune configs" in excel_config.
def gen_shape_detail_batch():
    return [1, 16, 32, 64, 128, 256]


def gen_shape_detail_input_c():
    # return [3, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416]
    return [3, 64, 96, 128]


def gen_shape_detail_input_h():
    return [7, 14, 28, 56, 112, 224]


def gen_shape_detail_input_w():
    return [7, 14, 28, 56, 112, 224]


def gen_shape_detail_out_c():
    return [64, 96, 128]


def gen_shape_detail_kernel_h():
    return [2, 3, 4, 8]


def gen_shape_detail_kernel_w():
    return [2, 3, 4, 8]


def gen_shape_detail_stride():
    return [1, 2, 3]


def gen_shape_detail_padding():
    return [1, 4, 8, 16, 32, 64]


def gen_shape_detail_groups():
    return [1, 4, 8, 16, 32, 64]


def constraint_input_size_based_on_input_c(**kwargs):
    detail_input_c = kwargs["shape_detail_input_c"]
    input_h = kwargs["shape_detail_input_h"]
    input_w = kwargs["shape_detail_input_w"]

    if detail_input_c == 3:
        return input_h >= 112 and input_w >= 112
    elif detail_input_c == 64:
        return input_h <= 112 and input_w <= 112
    elif detail_input_c == 96:
        return input_h <= 56 and input_w <= 56
    elif detail_input_c == 128:
        return input_h <= 28 and input_w <= 28
    else:
        return True


def constraint_kernel_size_ge_stride(**kwargs):
    kernel_h = kwargs["shape_detail_kernel_h"]
    kernel_w = kwargs["shape_detail_kernel_w"]
    stride = kwargs["shape_detail_stride"]
    return kernel_h >= stride and kernel_w >= stride


def constraint_kernel_size_gt_padding(**kwargs):
    kernel_h = kwargs["shape_detail_kernel_h"]
    kernel_w = kwargs["shape_detail_kernel_w"]
    padding = kwargs["shape_detail_padding"]
    return kernel_h > padding and kernel_w > padding


def constraint_kernel_size_less_than_input(**kwargs):
    return (
        kwargs["shape_detail_kernel_h"] < kwargs["shape_detail_input_h"]
        and kwargs["shape_detail_kernel_w"] < kwargs["shape_detail_input_w"]
    )


def constraint_groups_divide_input_c(**kwargs):
    return kwargs["shape_detail_input_c"] % kwargs["shape_detail_groups"] == 0


def constraint_input_c_divisible_by_groups(**kwargs):
    return kwargs["shape_detail_input_c"] % kwargs["shape_detail_groups"] == 0


def constraint_output_c_divisible_by_groups(**kwargs):
    return kwargs["shape_detail_out_c"] % kwargs["shape_detail_groups"] == 0


def constraint_output_size_positive(**kwargs):
    input_h = kwargs["shape_detail_input_h"]
    input_w = kwargs["shape_detail_input_w"]
    kernel_h = kwargs["shape_detail_kernel_h"]
    kernel_w = kwargs["shape_detail_kernel_w"]
    stride = kwargs["shape_detail_stride"]
    padding = kwargs["shape_detail_padding"]

    output_h = (input_h + 2 * padding - kernel_h) // stride + 1
    output_w = (input_w + 2 * padding - kernel_w) // stride + 1

    return output_h > 0 and output_w > 0


def constraint_groups_le_input_and_output_c(**kwargs):
    return (
        kwargs["shape_detail_groups"] <= kwargs["shape_detail_input_c"]
        and kwargs["shape_detail_groups"] <= kwargs["shape_detail_out_c"]
    )


def constraint_padding_lt_input_size(**kwargs):
    return (
        kwargs["shape_detail_padding"] < kwargs["shape_detail_input_h"]
        and kwargs["shape_detail_padding"] < kwargs["shape_detail_input_w"]
    )


def gen_BLOCK_NI_HO_WO():
    return [32, 64, 128, 256]


def gen_BLOCK_CI():
    return [16, 32, 64, 128]


def gen_BLOCK_CO():
    return [16, 32, 64, 128]


def gen_num_warps():
    return [1, 2, 4]


def gen_num_stages():
    return [1, 2, 3, 4, 5]


# ===---------------------------------------------------------------------------------===
# Shape and Configuration Generators for Performance Testing
# ===---------------------------------------------------------------------------------===
# This section initializes the `ShapeGenerator` and `ConfigGenerator` classes, which are
# responsible for generating combinations of shapes and configurations for performance
# testing. These generators use the parameter generation functions defined earlier to
# create a comprehensive set of test cases. The `excel_config` dictionary is passed to
# both generators to ensure consistency with the training set data and to avoid retesting
# previously evaluated combinations.
# ===---------------------------------------------------------------------------------===

shapegen = ShapeGenerator(
    excel_config,
    (
        gen_shape_detail_batch,
        gen_shape_detail_input_c,
        gen_shape_detail_input_h,
        gen_shape_detail_input_w,
        gen_shape_detail_out_c,
        gen_shape_detail_kernel_h,
        gen_shape_detail_kernel_w,
        gen_shape_detail_stride,
        gen_shape_detail_padding,
        gen_shape_detail_groups,
    ),
    (
        constraint_input_size_based_on_input_c,
        constraint_kernel_size_ge_stride,
        constraint_kernel_size_gt_padding,
        constraint_kernel_size_less_than_input,
        constraint_groups_divide_input_c,
        constraint_input_c_divisible_by_groups,
        constraint_output_size_positive,
        constraint_groups_le_input_and_output_c,
        constraint_padding_lt_input_size,
        constraint_output_c_divisible_by_groups,
    ),
)
configgen = ConfigGenerator(
    excel_config,
    (gen_BLOCK_NI_HO_WO, gen_BLOCK_CI, gen_BLOCK_CO, gen_num_warps, gen_num_stages),
)

shape_config_combinations = list(
    itertools.product(shapegen.generate(), configgen.generate())
)


# ===---------------------------------------------------------------------------------===
# Filter Out Already Iterated Shape-Config Combinations
# ===---------------------------------------------------------------------------------===
# This section filters out shape-config combinations that have already been traversed and
# stored in `have_itered_shape_config_pairs`.
# ===---------------------------------------------------------------------------------===

# Filter out combinations that have already been traversed
if filter_out_repeat_comb:
    shape_config_combinations = [
        pair
        for pair in shape_config_combinations
        if convert_to_tuple(
            pair, excel_config, stringDtype2TorchDtype(pytest_data_type)
        )
        not in have_itered_shape_config_pairs
    ]

if print_shape_config_combinations:
    for shape, config in shape_config_combinations:
        print(shape, config)


# ===---------------------------------------------------------------------------------===
# Group Shape-Config Combinations by Config
# ===---------------------------------------------------------------------------------===
# This section groups shape-config combinations by their configuration (`config_cols`).
# Each unique configuration will correspond to a list of shapes (`shape_cols`), allowing
# for efficient batch processing of shapes with the same configuration.
# ===---------------------------------------------------------------------------------===


# Helper function to extract config from a shape-config pair
def extract_config(pair, excel_config):
    shape, config = pair
    return {key: config[key] for key in excel_config["config_cols"]}


# Group shape-config combinations by config
grouped_shape_config_combinations = defaultdict(list)
for pair in shape_config_combinations:
    config_key = extract_config(pair, excel_config)
    grouped_shape_config_combinations[tuple(config_key.items())].append(pair)


# ===---------------------------------------------------------------------------------===
# Run Performance Tests
# ===---------------------------------------------------------------------------------===
# This script processes each unique configuration and its associated shapes, writes them
# to YAML files, runs performance tests, and appends the results to a main result file.
# The "START" and "END" labels help visually separate the output for each configuration.
# ===---------------------------------------------------------------------------------===

# Iterate over each unique configuration
for config_key in grouped_shape_config_combinations:
    print_centered_label(" START ", color="green")

    # Convert the config_key (tuple of items) back to a dictionary
    config_dict = dict(config_key)
    print(f"Config: {config_dict}")
    # Write the configuration to YAML
    write_config_to_yaml(config_dict)

    # Reload the configuration and the operation
    flag_gems.runtime.config_loader = flag_gems.runtime.ConfigLoader.reset_instance()
    importlib.reload(sys.modules["flag_gems.ops." + pytest_operation_name])

    # Write the shapes to YAML
    write_shapes_to_yaml(
        [shape_pair[0] for shape_pair in grouped_shape_config_combinations[config_key]],
        excel_config,
        output_path=pytest_shape_file,
    )

    if print_grouped_shape_config_combinations:
        # Iterate over each shape pair for the current configuration
        for shape_pair in grouped_shape_config_combinations[config_key]:
            shape = shape_pair[0]  # Extract the shape dictionary
            print(f"Shape: {shape}")

    # Run pytest
    output_file = run_perf_pytest(
        operation=pytest_operation_name,
        shape_file=pytest_shape_file,
        warmup=pytest_warmup_runs,
        iter=pytest_iter_runs,
        dtypes=pytest_data_type,
        verbose=pytest_verbose,
    )
    subprocess.run("cat " + output_file + " >> " + result_file, shell=True, check=False)

    print_centered_label(" END ", color="green")
