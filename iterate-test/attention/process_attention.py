import json

import pandas as pd

# ===================================
input_log = "results/attention-result.txt"
output_excel = (
    "results/attention-eval-fixedshape-allcover-4090-withvstrainsetspeedup.csv"
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
        "block_m",
        "block_n",
        "pre_load_v",
        "warps",
        "stages",
        "legacy_shape",
        "shape_detail_B",
        "shape_detail_H",
        "shape_detail_L",
        "shape_detail_D",
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
                "block_m": autotune_configs["block_m"],
                "block_n": autotune_configs["block_n"],
                "pre_load_v": autotune_configs["pre_load_v"],
                "warps": autotune_configs["warps"],
                "stages": autotune_configs["stages"],
                "legacy_shape": result["legacy_shape"],
                "shape_detail_B": shape_detail[0][0],
                "shape_detail_H": shape_detail[0][1],
                "shape_detail_L": shape_detail[0][2],
                "shape_detail_D": shape_detail[0][3],
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
