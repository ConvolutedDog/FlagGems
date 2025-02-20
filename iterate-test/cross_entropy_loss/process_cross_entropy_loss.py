import json

import pandas as pd

# ===================================
input_log = "results/cross_entropy_loss-result.txt"
output_excel = "results/cross_entropy_loss-result-512-512-8192-512-512-8192.csv"
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
        "BLOCK_C",
        "BLOCK_D",
        "num_warps",
        "legacy_shape",
        "shape_detail_N",
        "shape_detail_L",
        "form_detail_has_weight",
        "form_detail_reduction",
        "form_detail_label_smoothing",
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
                "BLOCK_C": autotune_configs["BLOCK_C"],
                "BLOCK_D": autotune_configs["BLOCK_D"],
                "num_warps": autotune_configs["num_warps"],
                "legacy_shape": result["legacy_shape"],
                "shape_detail_N": shape_detail[0][0][0],
                "shape_detail_L": shape_detail[0][0][1],
                "form_detail_has_weight": 1 if "weight" in shape_detail[1] else None,
                "form_detail_reduction": 0
                if shape_detail[1]["reduction"] == "none"
                else (
                    1
                    if shape_detail[1]["reduction"] == "mean"
                    else (2 if shape_detail[1]["reduction"] == "sum" else None)
                ),
                "form_detail_label_smoothing": shape_detail[1]["label_smoothing"],
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
