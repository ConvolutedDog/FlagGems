import json

import pandas as pd

# ===================================
input_log = "results/conv2d-result.txt"
output_excel = (
    "results/conv2d-1_16_32_64_128_256-3_64_96_128-7_14_28_56_112_224-"
    + "7_14_28_56_112_224-64_96_128-2_3_4_8-2_3_4_8-1_2_3-1_4_8_16_32_64-1_4_8_16_32_64-"
    + "32_64_128_256-16_32_64_128-16_32_64_128-1_2_4-1_2_3_4_5-fixedshape-allcover-H100.csv"
)
# ===================================

f = open(input_log, "r")
json_data = f.readlines()
f.close()

df = pd.DataFrame(
    columns=[
        "op_name",
        "dtype",
        "mode",
        "level",
        "BLOCK_NI_HO_WO",
        "BLOCK_CI",
        "BLOCK_CO",
        "num_warps",
        "num_stages",
        "legacy_shape",
        "shape_detail_batch",
        "shape_detail_input_c",
        "shape_detail_input_h",
        "shape_detail_input_w",
        "shape_detail_out_c",
        "shape_detail_kernel_h",
        "shape_detail_kernel_w",
        "shape_detail_groups",
        "shape_detail_stride",
        "shape_detail_padding",
        "latency_base",
        "latency",
        "gbps_base",
        "gbps",
        "speedup",
        "accuracy",
        "tflops",
        "utilization",
        "latency_torch_compile",
        "speedup_vs_torch_compile",
        "latency_native_flaggems",
        "speedup_vs_native_flaggems",
        "speedup_vs_native_flaggems_trainset",
        "error_msg",
    ]
)

lines = json_data

for line in lines:
    if line.startswith("[INFO]"):
        json_str = line.split("[INFO] ")[1]
        data = json.loads(json_str)

        # autotune_configs
        autotune_configs = data["autotune_configs"]

        # result
        results = data["result"]

        rows = []
        for result in results:
            shape_detail = result["shape_detail"]
            row = {
                "op_name": data["op_name"],
                "dtype": data["dtype"],
                "mode": data["mode"],
                "level": data["level"],
                "BLOCK_NI_HO_WO": autotune_configs["BLOCK_NI_HO_WO"],
                "BLOCK_CI": autotune_configs["BLOCK_CI"],
                "BLOCK_CO": autotune_configs["BLOCK_CO"],
                "num_warps": autotune_configs["num_warps"],
                "num_stages": autotune_configs["num_stages"],
                "legacy_shape": result["legacy_shape"],
                "shape_detail_batch": shape_detail["input"][0],
                "shape_detail_input_c": shape_detail["input"][1],
                "shape_detail_input_h": shape_detail["input"][2],
                "shape_detail_input_w": shape_detail["input"][3],
                "shape_detail_out_c": shape_detail["weight"][0],
                "shape_detail_kernel_h": shape_detail["weight"][2],
                "shape_detail_kernel_w": shape_detail["weight"][3],
                "shape_detail_groups": shape_detail["groups"],
                "shape_detail_stride": shape_detail["stride"],
                "shape_detail_padding": shape_detail["padding"],
                "latency_base": result["latency_base"],
                "latency": result["latency"],
                "gbps_base": result["gbps_base"],
                "gbps": result["gbps"],
                "speedup": result["speedup"],
                "accuracy": result["accuracy"],
                "tflops": result["tflops"],
                "utilization": result["utilization"],
                "latency_torch_compile": result["latency_torch_compile"],
                "speedup_vs_torch_compile": result["speedup_vs_torch_compile"],
                "latency_native_flaggems": result["latency_native_flaggems"],
                "speedup_vs_native_flaggems": result["speedup_vs_native_flaggems"],
                "speedup_vs_native_flaggems_trainset": result[
                    "speedup_vs_native_flaggems_trainset"
                ],
                "error_msg": result["error_msg"],
            }

            # df = df.append(row, ignore_index=True)
            rows.append(row)

        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

df.to_csv(output_excel, index=False)

print("Written to " + output_excel)
