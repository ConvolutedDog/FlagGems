import importlib
import os
import subprocess
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

pytest_operation_name = "attention"
# Optional["float16", "bfloat16"]
pytest_data_type = "float16"

pytest_verbose = True
pytest_warmup_runs = 10
pytest_iter_runs = 30

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
    "shape_cols": [
        "shape_detail_B",
        "shape_detail_H",
        "shape_detail_L",
        "shape_detail_D",
    ],
    # Auto-tune configs.
    "config_cols": [
        "block_m",
        "block_n",
        "pre_load_v",
        "warps",
        "stages",
    ],
    # Performance.
    "latency_col": "latency",
    # Benchmark name of shape yaml.
    "bench_name": "AttentionBenchmark",
    # Shape description of shape yaml. It should correspond one-to-one with "shape_cols".
    "shape_desc": [],
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
def gen_shape_detail_B():
    return [1, 4, 8, 16, 32, 64]
    # return [24, 40, 48, 56]


def gen_shape_detail_H():
    return [8, 16, 32, 64]
    # return [24, 40, 48, 56]


def gen_shape_detail_L():
    return [512, 1024, 2048]
    # return [512, 1024]


def gen_shape_detail_D():
    return [32, 64, 128]
    # return [32, 64]


# ===---------------------------------------------------------------------------------===
# Configuration Parameter Generation Functions
# ===---------------------------------------------------------------------------------===
# This section contains functions to generate configuration parameters (e.g., block sizes,
# split factors, number of stages, etc.) based on input shapes (M, K, N). These functions
# use predefined formulas to calculate optimal values for each parameter.
# ===---------------------------------------------------------------------------------===


# Define functions to generate parameters
def gen_block_m(shape_detail_B, shape_detail_H, shape_detail_L, shape_detail_D):
    # res = 0.0066405717512804*shape_detail_B + 0.00734747023809523*shape_detail_D + 0.0034621578099839*shape_detail_H + 0.000151134672619047*shape_detail_L
    # return 2 ** (round(res + 5.2508455764377))
    res = (
        0.00683199967292153 * shape_detail_B
        + 0.00936330104811068 * shape_detail_D
        + 0.00448095972738805 * shape_detail_H
        + 0.000148717188119298 * shape_detail_L
    )
    return 2 ** (round(res + 5.02875534230089))


def gen_block_n(shape_detail_B, shape_detail_H, shape_detail_L, shape_detail_D):
    # res = -0.0086496205851233*shape_detail_B + -0.00179811507936508*shape_detail_D + -0.007085346215781*shape_detail_H + 0.000218951512896825*shape_detail_L
    # return 2 ** (round(res + 5.33952007458942))
    res = (
        -0.0188492809138742 * shape_detail_B
        + -0.000773963510128283 * shape_detail_D
        + -0.00599468239382565 * shape_detail_H
        + 0.000286142116806643**shape_detail_L
    )
    return 2 ** (round(res + 5.3864909959795))


def gen_pre_load_v(shape_detail_B, shape_detail_H, shape_detail_L, shape_detail_D):
    # res = -0.00223747121802199*shape_detail_B + -0.000453374954055956*shape_detail_D + -0.000317914402565846*shape_detail_H + -1.13343738513988e-6*shape_detail_L
    # return 2 ** (round(res + 1.41904481901093)) - 1
    res = (
        -0.00148424365317821 * shape_detail_B
        + -0.000116212846695584 * shape_detail_D
        + 0.00542074895974272 * shape_detail_H
        + 4.6115370774636e-5**shape_detail_L
    )
    return 2 ** (round(res + 0.623104826362066)) - 1


def gen_warps(shape_detail_B, shape_detail_H, shape_detail_L, shape_detail_D):
    # res = -0.00277177335072994*shape_detail_B + 0.00403025793650793*shape_detail_D + -0.00181159420289855*shape_detail_H + 6.58792162698411e-5*shape_detail_L
    # return 2 ** (round(res + 1.8528338449679))
    res = (
        -0.00459004156423515 * shape_detail_B
        + 0.00610042249850166 * shape_detail_D
        + -0.00125747752342906 * shape_detail_H
        + 0.000211103162525124 * shape_detail_L
    )
    return 2 ** (round(res + 1.53158201642767))


def gen_stages(shape_detail_B, shape_detail_H, shape_detail_L, shape_detail_D):
    # res = -0.000644258794051397*shape_detail_B + -0.00108506944444444*shape_detail_D + -0.00159017713365539*shape_detail_H + -6.78168402777779e-5*shape_detail_L
    # return 2 ** (round(res + 3.13520144629314))
    res = (
        -0.000330363298158426 * shape_detail_B
        + -0.000494254736256258 * shape_detail_D
        + 0.000679446965320056 * shape_detail_H
        + -2.10789867187994e-5 * shape_detail_L
    )
    return round(res + 3.0216598676481)


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
    (gen_shape_detail_B, gen_shape_detail_H, gen_shape_detail_L, gen_shape_detail_D),
)

# print(shapegen.generate())
# for kv in shapegen.generate():
#     print(kv)

tunedconfiggen = TunedConfigGenerator(
    excel_config,
    (gen_block_m, gen_block_n, gen_pre_load_v, gen_warps, gen_stages),
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
