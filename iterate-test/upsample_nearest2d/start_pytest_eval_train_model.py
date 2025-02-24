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
    get_gpu_name,
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

pytest_operation_name = "upsample_nearest2d"
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
    "shape_cols": [
        "shape_detail_N",
        "shape_detail_C",
        "shape_detail_H",
        "shape_detail_W",
        "form_detail_Ho",
        "form_detail_Wo",
    ],
    # Auto-tune configs.
    "config_cols": ["block_n", "warps"],
    # Performance.
    "latency_col": "latency",
    # Benchmark name of shape yaml.
    "bench_name": "UPSAMPLENEAREST2DBenchmark",
    # Shape description of shape yaml. It should correspond one-to-one with "shape_cols".
    "shape_desc": ["N", "C", "H", "W"],
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
def gen_shape_detail_N():
    # return list(range(512, 8192 + 1, 512))
    return [
        1,
    ]


def gen_shape_detail_C():
    # return list(range(512, 524288 + 1, 512))
    return [
        3,
    ]


def gen_shape_detail_H():
    # return list(range(512, 524288 + 1, 512))
    return [7, 13, 27, 56, 112, 224]


def gen_shape_detail_W():
    # return list(range(512, 8192 + 1, 512))
    return [7, 13, 27, 56, 112, 224]


scale_factor_h, scale_factor_w = 2, 2


def gen_form_detail_Ho():
    return list(map(lambda i: i * scale_factor_h, gen_shape_detail_H()))


def gen_form_detail_Wo():
    return list(map(lambda i: i * scale_factor_w, gen_shape_detail_W()))


def constraint_HW_equal(**kwargs):
    input_h = kwargs["shape_detail_H"]
    input_w = kwargs["shape_detail_W"]

    form_detail_Ho = kwargs["form_detail_Ho"]
    form_detail_Wo = kwargs["form_detail_Wo"]

    return (
        input_h == input_w
        and form_detail_Ho == 2 * input_h
        and form_detail_Wo == 2 * input_w
    )


# ===---------------------------------------------------------------------------------===
# Configuration Parameter Generation Functions
# ===---------------------------------------------------------------------------------===
# This section contains functions to generate configuration parameters (e.g., block sizes,
# split factors, number of stages, etc.) based on input shapes (M, K, N). These functions
# use predefined formulas to calculate optimal values for each parameter.
# ===---------------------------------------------------------------------------------===

current_gpu_name = get_gpu_name()


# Define functions to generate parameters
def gen_block_n(
    shape_detail_N,
    shape_detail_C,
    shape_detail_H,
    shape_detail_W,
    form_detail_Ho,
    form_detail_Wo,
):
    if current_gpu_name == "NVIDIA GeForce RTX 4090":
        res = (
            -0.00660090172074113 * shape_detail_C
            + 1.70268651097824 * shape_detail_H
            - 1.53809040762666 * shape_detail_N
            + 1.70268651097824 * shape_detail_W
            + 262.167259759111
        )
        candidates = [156, 512, 1024, 2048, 4096]
        closest_value = min(candidates, key=lambda x: abs(x - res))
        return closest_value
    elif current_gpu_name == "NVIDIA H100 80GB HBM3":
        res = (
            -0.029749987219103 * shape_detail_C
            + 1.22629818764949 * shape_detail_H
            - 2.78031966166532 * shape_detail_N
            + 1.22629818764949 * shape_detail_W
            + 329.45064476148
        )
        candidates = [156, 512, 1024, 2048, 4096]
        closest_value = min(candidates, key=lambda x: abs(x - res))
        return closest_value
    elif current_gpu_name == "Quadro GV100":
        res = (
            -0.0263468967131644 * shape_detail_C
            + 2.45369420195092 * shape_detail_H
            - 0.798437523656681 * shape_detail_N
            + 2.45369420195092 * shape_detail_W
            + 263.903146264578
        )
        candidates = [156, 512, 1024, 2048, 4096]
        closest_value = min(candidates, key=lambda x: abs(x - res))
        return closest_value


def gen_warps(
    shape_detail_N,
    shape_detail_C,
    shape_detail_H,
    shape_detail_W,
    form_detail_Ho,
    form_detail_Wo,
):
    if current_gpu_name == "NVIDIA GeForce RTX 4090":
        res = (
            -0.000733199780908817 * shape_detail_C
            - 0.0202793134282105 * shape_detail_H
            - 0.0824917864010052 * shape_detail_N
            - 0.0202793134282105 * shape_detail_W
            + 7.86860335102747
        )
        candidates = [2, 4, 8, 16]
        closest_value = min(candidates, key=lambda x: abs(x - res))
        return closest_value
    elif current_gpu_name == "NVIDIA H100 80GB HBM3":
        res = (
            -0.00105658030375112 * shape_detail_C
            - 0.0198305135122421 * shape_detail_H
            - 0.0720323800007066 * shape_detail_N
            - 0.019830513512242 * shape_detail_W
            + 7.55422547727122
        )
        candidates = [2, 4, 8, 16]
        closest_value = min(candidates, key=lambda x: abs(x - res))
        return closest_value
    elif current_gpu_name == "Quadro GV100":
        res = (
            -0.00121968604182412 * shape_detail_C
            - 0.0146597102269414 * shape_detail_H
            - 0.0439749277051885 * shape_detail_N
            - 0.0146597102269413 * shape_detail_W
            + 7.02980543990727
        )
        candidates = [2, 4, 8, 16]
        closest_value = min(candidates, key=lambda x: abs(x - res))
        return closest_value


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
        gen_shape_detail_N,
        gen_shape_detail_C,
        gen_shape_detail_H,
        gen_shape_detail_W,
        gen_form_detail_Ho,
        gen_form_detail_Wo,
    ),
    [constraint_HW_equal],
)

# print(shapegen.generate())
# for kv in shapegen.generate():
#     print(kv)

tunedconfiggen = TunedConfigGenerator(
    excel_config,
    (gen_block_n, gen_warps),
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
