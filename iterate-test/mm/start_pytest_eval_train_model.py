import subprocess

from performance_utils import (
    MMShapeGenerator,
    TunedParameterFunctions,
    TunedParameterGenerator,
    archive_file_with_timestamp,
    remove_triton_cache,
    run_perf_pytest,
)

operation = "mm"


result_file = "results/" + operation + "-result.txt"
archive_file_with_timestamp(result_file)


# FIXED
# generator = MMShapeGenerator(start=512, end=8192, step=512)

# RANDOM
generator = MMShapeGenerator(start=512, end=8192, step=512, num=32)


# Define functions to generate parameters
def block_m_func(shapeM, shapeK, shapeN):
    """
    block m: 5.02754660e-5  1.24650843e-05 5.70746029e-05 0.951953125 5
    """
    res = (
        5.02754660e-5 * shapeM
        + 1.24650843e-05 * shapeK
        + 5.70746029e-05 * shapeN
        + 0.951953125
    )
    return 2 ** (round(res + 5))


def block_k_func(shapeM, shapeK, shapeN):
    """
    bolck k: -4.32182761e-05 1.28128949e-05 -4.64046703e-05 0.47832031249999984 5
    """
    res = (
        -4.32182761e-05 * shapeM
        + 1.28128949e-05 * shapeK
        + -4.64046703e-05 * shapeN
        + 0.47832031249999984
    )
    return 2 ** (round(res + 5))


def block_n_func(shapeM, shapeK, shapeN):
    """
    block n: 4.52714808e-05 -7.57329604e-06 3.63630407e-05 0.52656250 6
    """
    res = (
        4.52714808e-05 * shapeM
        + -7.57329604e-06 * shapeK
        + 3.63630407e-05 * shapeN
        + 0.52656250
    )
    return 2 ** (round(res + 6))


def split_k_func(shapeM, shapeK, shapeN):
    return 1


def num_stages_func(shapeM, shapeK, shapeN):
    """
    num stages: -1.34748571e-05 -7.30402329e-06 -6.40644747e-06 1.0521484374999996 1
    """
    res = (
        -1.34748571e-05 * shapeM
        + -7.30402329e-06 * shapeK
        + -6.40644747e-06 * shapeN
        + 1.0521484374999996
    )
    return 2 ** (round(res + 1))


def num_warps_func(shapeM, shapeK, shapeN):
    """
    num warps: -1.61563649e-05 2.72414264e-05 -2.95751235e-05 1.27919921875 1
    """
    res = (
        -1.61563649e-05 * shapeM
        + 2.72414264e-05 * shapeK
        + -2.95751235e-05 * shapeN
        + 1.27919921875
    )
    return 2 ** (round(res + 1))


def num_ctas_func(shapeM, shapeK, shapeN):
    return None


# Create an instance of TunedParameterFunctions
tunedParamFunctions = TunedParameterFunctions(
    block_m_func=block_m_func,
    block_k_func=block_k_func,
    block_n_func=block_n_func,
    split_k_func=split_k_func,
    num_stages_func=num_stages_func,
    num_warps_func=num_warps_func,
    num_ctas_func=num_ctas_func,
)

iterShape = generator.iterShapeOneByOne(
    filename="configs/mm_shape.yaml", generate_fixed_shapes=False
)

while iterShape[0]:
    shapeM, shapeN, shapeK = iterShape[1], iterShape[2], iterShape[3]
    remove_triton_cache()

    # Initialize the TunedParameterGenerator
    tunedParaGenerator = TunedParameterGenerator(
        shapeM=shapeM,
        shapeK=shapeK,
        shapeN=shapeN,
        param_functions=tunedParamFunctions,
    )
    tunedParaGenerator.write_to_yaml(entry="mm")

    output_file = run_perf_pytest(operation, "configs/mm_shape.yaml", verbose=True)
    subprocess.run("cat " + output_file + " >> " + result_file, shell=True, check=False)

    iterShape = generator.iterShapeOneByOne(
        filename="configs/mm_shape.yaml", generate_fixed_shapes=False
    )
