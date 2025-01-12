import subprocess

from performance_utils import remove_triton_cache, \
                              ParameterIterator, \
                              MMShapeGenerator, \
                              archive_file_with_timestamp, \
                              run_perf_pytest


operation = "mm"


result_file = "results/" + operation + "-result.txt"
archive_file_with_timestamp(result_file)


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


### FIXED
generator = MMShapeGenerator(start=512, end=8192, step=512)
generator.save_to_yaml("configs/mm_shape.yaml")

### RANDOM
# generator = MMShapeGenerator(start=512, end=8192, step=512, num=32)
# generator.save_to_yaml("configs/mm_shape.yaml", generate_fixed_shapes=False)

while(paramIter.write_next_yaml(operation)):
    remove_triton_cache()
    output_file = run_perf_pytest(operation, "configs/mm_shape.yaml", verbose=True)
    subprocess.run("cat " + output_file + " >> " + result_file, shell=True, check=False)
