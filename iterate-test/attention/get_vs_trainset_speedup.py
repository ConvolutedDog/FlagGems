from pathlib import Path

import pandas as pd

# Excel 配置
excel_config = {
    "dtype_col": "dtype",
    "shape_cols": [
        "shape_detail_B",
        "shape_detail_H",
        "shape_detail_L",
        "shape_detail_D",
    ],
    "config_cols": [
        "block_m",
        "block_n",
        "pre_load_v",
        "warps",
        "stages",
    ],
    "latency_col": "latency",
    "bench_name": "AttentionBenchmark",
    "shape_desc": [],
}

# 文件路径
results_file = (
    "results/attention-eval-fixedshape-allcover-4090-withvstrainsetspeedup.csv"
)
train_set_dir = "train-set"

# 读取 results 文件
results_df = pd.read_csv(results_file)

# 初始化新列
results_df["speedup_vs_native_flaggems_trainset"] = None

# 遍历 results 文件的每一行
for index, row in results_df.iterrows():
    # 构建键：dtype + shape_cols
    key = tuple(
        [row[excel_config["dtype_col"]]]
        + [row[col] for col in excel_config["shape_cols"]]
    )

    # 初始化最小 latency
    min_latency = None

    # 遍历 train-set 目录下的所有 CSV 文件
    for train_file in Path(train_set_dir).glob("*.csv"):
        train_df = pd.read_csv(train_file)

        # 查找匹配的行
        match_rows = train_df[
            (train_df[excel_config["dtype_col"]] == key[0])
            & (train_df[excel_config["shape_cols"][0]] == key[1])
            & (train_df[excel_config["shape_cols"][1]] == key[2])
            & (train_df[excel_config["shape_cols"][2]] == key[3])
            & (train_df[excel_config["shape_cols"][3]] == key[4])
        ]

        # 如果找到匹配的行，更新最小 latency
        if not match_rows.empty:
            current_min_latency = match_rows[excel_config["latency_col"]].min()
            if min_latency is None or current_min_latency < min_latency:
                min_latency = current_min_latency

    # 如果找到最小 latency，填充到 results 文件的对应列
    if min_latency is not None:
        results_df.at[index, "speedup_vs_native_flaggems_trainset"] = (
            min_latency / row[excel_config["latency_col"]]
        )

# 保存更新后的 results 文件
results_df.to_csv(results_file, index=False)

print("Processing complete. Updated results saved to:", results_file)
