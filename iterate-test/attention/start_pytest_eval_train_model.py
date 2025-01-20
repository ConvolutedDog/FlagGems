import importlib
import subprocess
import sys
from collections import defaultdict

from performance_utils import (
    ShapeGenerator,
    TunedConfigGenerator,
    archive_file_with_timestamp,
    print_centered_label,
    read_config_from_yaml,
    run_perf_pytest,
    write_config_to_yaml,
    write_shapes_to_yaml,
)

# flake8: noqa: E402
import flag_gems

# ===---------------------------------------------------------------------------------===
# User-Specified Parameters
# ===---------------------------------------------------------------------------------===

pytest_operation_name = "mm"
# Optional["float16", "float32", "bfloat16", "int16", "int32", "bool", "cfloat"]
pytest_data_type = "float16"

pytest_verbose = True
pytest_warmup_runs = 3
pytest_iter_runs = 3

# Just don't edit this.
pytest_shape_file = "configs/shape.yaml"

# Just don't edit this.
print_shape_config_combinations = True
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
    "shape_cols": ["shape_detail_M", "shape_detail_N", "shape_detail_K"],
    # Auto-tune configs.
    "config_cols": [
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "SPLIT_K",
        "num_stages",
        "num_warps",
    ],
    # Performance.
    "latency_col": "latency",
    # Benchmark name of shape yaml.
    "bench_name": "MMBenchmark",
    # Shape description of shape yaml. It should correspond one-to-one with "shape_cols".
    "shape_desc": ["M", "N", "K"],
}
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
# Shape Generator Functions
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
def gen_shape_detail_M():
    return [2048, 1024]


def gen_shape_detail_K():
    return [2048, 512]


def gen_shape_detail_N():
    return [2048, 4096]


# ===---------------------------------------------------------------------------------===
# Configuration Parameter Generation Functions
# ===---------------------------------------------------------------------------------===
# This section contains functions to generate configuration parameters (e.g., block sizes,
# split factors, number of stages, etc.) based on input shapes (M, K, N). These functions
# use predefined formulas to calculate optimal values for each parameter.
# ===---------------------------------------------------------------------------------===


def discretize_to_multiple(x, multiple):
    return round(x / multiple) * multiple


# Define functions to generate parameters
def gen_BLOCK_M(shape_detail_M, shape_detail_N, shape_detail_K):
    """
    block m: 5.02754660e-5  1.24650843e-05 5.70746029e-05 0.951953125 5
    """
    res = (
        5.02754660e-5 * shape_detail_M
        + 1.24650843e-05 * shape_detail_K
        + 5.70746029e-05 * shape_detail_N
        + 0.951953125
    )
    return 2 ** (round(res + 5))


def gen_BLOCK_K(shape_detail_M, shape_detail_N, shape_detail_K):
    """
    bolck k: -4.32182761e-05 1.28128949e-05 -4.64046703e-05 0.47832031249999984 5
    """
    res = (
        -4.32182761e-05 * shape_detail_M
        + 1.28128949e-05 * shape_detail_K
        + -4.64046703e-05 * shape_detail_N
        + 0.47832031249999984
    )
    return 2 ** (round(res + 5))


def gen_BLOCK_N(shape_detail_M, shape_detail_N, shape_detail_K):
    """
    block n: 4.52714808e-05 -7.57329604e-06 3.63630407e-05 0.52656250 6
    """
    res = (
        4.52714808e-05 * shape_detail_M
        + -7.57329604e-06 * shape_detail_K
        + 3.63630407e-05 * shape_detail_N
        + 0.52656250
    )
    return 2 ** (round(res + 6))


def gen_SPLIT_K(shape_detail_M, shape_detail_N, shape_detail_K):
    return 1


def gen_num_stages(shape_detail_M, shape_detail_N, shape_detail_K):
    """
    num stages: -1.34748571e-05 -7.30402329e-06 -6.40644747e-06 1.0521484374999996 1
    """
    res = (
        -1.34748571e-05 * shape_detail_M
        + -7.30402329e-06 * shape_detail_K
        + -6.40644747e-06 * shape_detail_N
        + 1.0521484374999996
    )
    return 2 ** (round(res + 1))


def gen_num_warps(shape_detail_M, shape_detail_N, shape_detail_K):
    """
    num warps: -1.61563649e-05 2.72414264e-05 -2.95751235e-05 1.27919921875 1
    """
    res = (
        -1.61563649e-05 * shape_detail_M
        + 2.72414264e-05 * shape_detail_K
        + -2.95751235e-05 * shape_detail_N
        + 1.27919921875
    )
    return 2 ** (round(res + 1))


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
    excel_config, (gen_shape_detail_M, gen_shape_detail_K, gen_shape_detail_N)
)

# print(shapegen.generate())
# for kv in shapegen.generate():
#     print(kv)

tunedconfiggen = TunedConfigGenerator(
    excel_config,
    (gen_BLOCK_M, gen_BLOCK_K, gen_BLOCK_N, gen_SPLIT_K, gen_num_stages, gen_num_warps),
    shapegen,
)

shape_config_combinations = tunedconfiggen.generate()

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
