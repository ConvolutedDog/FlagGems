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

pytest_operation_name = "mm"
# Optional["float16", "float32", "bfloat16", "int16", "int32", "bool", "cfloat"]
pytest_data_type = "float16"

pytest_verbose = True
pytest_warmup_runs = 1000
pytest_iter_runs = 3000

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
    return list(range(512, 8192 + 1, 512))


def gen_shape_detail_K():
    return list(range(512, 8192 + 1, 512))


def gen_shape_detail_N():
    return list(range(512, 8192 + 1, 512))


def constraint_MNK_equal(**kwargs):
    M = kwargs["shape_detail_M"]
    N = kwargs["shape_detail_N"]
    K = kwargs["shape_detail_K"]
    return M == N and N == K


# ===---------------------------------------------------------------------------------===
# Configuration Parameter Generation Functions
# ===---------------------------------------------------------------------------------===
# This section contains functions to generate configuration parameters (e.g., block sizes,
# split factors, number of stages, etc.) based on input shapes (M, K, N). These functions
# use predefined formulas to calculate optimal values for each parameter.
# ===---------------------------------------------------------------------------------===

current_gpu_name = get_gpu_name()

import numpy as np


def gen_BLOCK_M(shape_detail_M, shape_detail_N, shape_detail_K):
    if current_gpu_name == "NVIDIA GeForce RTX 4090":
        # res = (
        #     0.00331313189338235 * shape_detail_K
        #     + 0.00383588005514706 * shape_detail_M
        #     + 0.00230210248161765 * shape_detail_N
        #     + 54.0875
        # )
        # trainsets = [32, 64, 128, 256]
        # closest_value = min(trainsets, key=lambda x: abs(x - res))
        # return closest_value # 2015-02-21
        res = (
            -16.4416633774714 * np.log(shape_detail_K)
            + 8.84015266594844 * np.log(shape_detail_M)
            - 14.0015276484632 * np.log(shape_detail_N)
            + 0.00830391194531081 * shape_detail_K
            + 0.0011524984186108 * shape_detail_M
            + 0.0065521922249836 * shape_detail_N
            + 201.728913362594
        )
        candidates = [32, 64, 128, 256]
        closest_value = min(candidates, key=lambda x: abs(x - res))
        return closest_value  # 2015-02-27
    elif current_gpu_name == "NVIDIA H100 80GB HBM3":
        pass
    elif current_gpu_name == "Quadro GV100":
        pass


def gen_BLOCK_N(shape_detail_M, shape_detail_N, shape_detail_K):
    if current_gpu_name == "NVIDIA GeForce RTX 4090":
        # res = (
        #     -0.00102467256433823 * shape_detail_K
        #     + 0.00125732421875 * shape_detail_M
        #     + 0.00430908203124999 * shape_detail_N
        #     + 110.90625
        # )
        # trainsets = [32, 64, 128, 256]
        # closest_value = min(trainsets, key=lambda x: abs(x - res))
        # return closest_value # 2015-02-21
        res = (
            13.2924243121098 * np.log(shape_detail_K)
            + 1.18067698516382 * np.log(shape_detail_M)
            + 13.5143616845211 * np.log(shape_detail_N)
            - 0.00505951773636297 * shape_detail_K
            + 0.000898935957778755 * shape_detail_M
            + 0.000206868942391525 * shape_detail_N
            - 80.3681680799561
        )
        candidates = [32, 64, 128, 256]
        closest_value = min(candidates, key=lambda x: abs(x - res))
        return closest_value  # 2015-02-27
    elif current_gpu_name == "NVIDIA H100 80GB HBM3":
        pass
    elif current_gpu_name == "Quadro GV100":
        pass


def gen_BLOCK_K(shape_detail_M, shape_detail_N, shape_detail_K):
    if current_gpu_name == "NVIDIA GeForce RTX 4090":
        # res = (
        #     0.000849106732536766 * shape_detail_K
        #     + -0.00174524643841912 * shape_detail_M
        #     + -0.00165477079503677 * shape_detail_N
        #     + 49.46875
        # )
        # trainsets = [32, 64, 128]
        # closest_value = min(trainsets, key=lambda x: abs(x - res))
        # return closest_value # 2015-02-21
        res = (
            0.924693121302579 * np.log(shape_detail_K)
            - 17.0253564082307 * np.log(shape_detail_M)
            - 18.6519896087864 * np.log(shape_detail_N)
            + 0.000568421021000007 * shape_detail_K
            + 0.00342271054916868 * shape_detail_M
            + 0.00400694210658514 * shape_detail_N
            + 286.978432949335
        )
        candidates = [32, 64, 128]
        closest_value = min(candidates, key=lambda x: abs(x - res))
        return closest_value  # 2015-02-27
    elif current_gpu_name == "NVIDIA H100 80GB HBM3":
        pass
    elif current_gpu_name == "Quadro GV100":
        pass


def gen_SPLIT_K(shape_detail_M, shape_detail_N, shape_detail_K):
    res = 1
    return res


def gen_num_stages(shape_detail_M, shape_detail_N, shape_detail_K):
    if current_gpu_name == "NVIDIA GeForce RTX 4090":
        # res = (
        #     2.89468204273897e-5 * shape_detail_K
        #     + -1.64480770335478e-5 * shape_detail_M
        #     + -4.78183521943934e-5 * shape_detail_N
        #     + 3.699609375
        # )
        # trainsets = [2, 3, 4, 5]
        # closest_value = min(trainsets, key=lambda x: abs(x - res))
        # return closest_value # 2015-02-21
        res = (
            0.140726590493495 * np.log(shape_detail_K)
            - 0.119341599811431 * np.log(shape_detail_M)
            - 0.222357270090079 * np.log(shape_detail_N)
            - 1.37699926155773e-5 * shape_detail_K
            + 1.97774350712814e-5 * shape_detail_M
            + 1.96770223604287e-5 * shape_detail_N
            + 5.07311206865169
        )
        candidates = [2, 3, 4, 5]
        closest_value = min(candidates, key=lambda x: abs(x - res))
        return closest_value  # 2015-02-27
    elif current_gpu_name == "NVIDIA H100 80GB HBM3":
        pass
    elif current_gpu_name == "Quadro GV100":
        pass


def gen_num_warps(shape_detail_M, shape_detail_N, shape_detail_K):
    """
    def gen_num_warps(shape_detail_M, shape_detail_N, shape_detail_K):
            res = 0.000163044649011948*shape_detail_K - 0.000185125014361214*shape_detail_M - 0.000130238252527574*shape_detail_N + 5.8279296875
    """

    if current_gpu_name == "NVIDIA GeForce RTX 4090":
        # res = (
        #     0.000163044649011948 * shape_detail_K
        #     + -0.000185125014361214 * shape_detail_M
        #     + -0.000130238252527574 * shape_detail_N
        #     + 5.8279296875
        # )
        # trainsets = [2, 4]
        # closest_value = min(trainsets, key=lambda x: abs(x - res))
        # return closest_value # 2015-02-21
        res = (
            0.398926553727418 * np.log(shape_detail_K)
            + 0.122707432836652 * np.log(shape_detail_M)
            + 0.312073825010493 * np.log(shape_detail_N)
            + 4.19526012781772e-5 * shape_detail_K
            - 0.000222372207298131 * shape_detail_M
            - 0.000224966613328673 * shape_detail_N
            + 0.130129332699289
        )
        candidates = [2, 4]
        closest_value = min(candidates, key=lambda x: abs(x - res))
        return closest_value  # 2015-02-27
    elif current_gpu_name == "NVIDIA H100 80GB HBM3":
        pass
    elif current_gpu_name == "Quadro GV100":
        pass


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
    (gen_shape_detail_M, gen_shape_detail_K, gen_shape_detail_N),
    [
        constraint_MNK_equal,
    ],
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
