import yaml
import itertools
import subprocess

from performance_utils import get_yaml_path, remove_triton_cache
from conftest import Config

# def get_yaml_path():
#     """This func returns the absolute path of `tune_configs.yaml`, please note that
#     this is not in the source code path, but in the package path of miniconda."""
#     # conda_env_path = os.environ['CONDA_PREFIX']
#     # relative_path = "lib/python3.12/site-packages/flag_gems/runtime/backend/_nvidia/tune_configs.yaml"
#     # YAMLPATH = os.path.join(conda_env_path, relative_path)
#     # return YAMLPATH
#     return "/home/yangjianchao/Github/FlagGems-IterateBench/src/flag_gems/runtime/backend/_nvidia/tune_configs.yaml"

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
# paramIter = ParameterIterator(block_m_range=(16, 256, 16, False), # 8
#                               block_n_range=(16, 256, 16, False), # 8
#                               block_k_range=(16, 256, 16, False), # 4
#                               split_k_range=(1, 1, 2, False), # 1
#                               num_stages_range=(2, 3, 1, False), # 2
#                               num_warps_range=(2, 8, 2, False), # 3
#                               num_ctas_range=(1, 1, 2, False)) # 1  # If donnot write num_ctas, should set here to be 1 choice, and 
#                                                                    # make comment on generate_triton_config's config string.


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
        import warnings
        warnings.warn("Write!!!", UserWarning)
        return True
    except StopIteration as e:
        print(e)
        return False


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


generator = MMShapeGenerator(start=1024, end=8192, step=1024)
generator.save_to_yaml("mm_shape.yaml")


cmd = "pytest test_mm_perf.py --level core --warmup 1 --iter 5 --dtypes float16 --shape_file mm_shape.yaml --record log"
output_file = "result_test_mm_perf--level_core--warmup_1--iter_5--dtypes_float16--shape_file_mm_shape.yaml--record_log.log"
result_file = "mm-result.txt"
subprocess.run("rm " + result_file, shell=True, check=False)


# f_in = open(output_file, "r")
f_out = open(result_file, "a")

while(write_next_yaml()):
    remove_triton_cache()
    subprocess.run(cmd, shell=True, check=False)
    subprocess.run("cat " + output_file + " >> " + result_file, shell=True, check=False)