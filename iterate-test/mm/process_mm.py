
import pandas as pd
import json

# ===================================
input_log = "mm-512-8192-32x-Random-4090.txt"
output_excel = "mm-512-8192-32x-Random-4090.xlsx"
# ===================================

f = open(input_log, 'r')
json_data = f.readlines()
f.close()

df = pd.DataFrame(columns=[
    'op_name', 'dtype', 'mode', 'level', 
    'BLOCK_M', 'BLOCK_N', 'BLOCK_K', 'SPLIT_K', 'num_stages', 'num_warps', 'num_ctas'
    'legacy_shape', 'shape_detail_M', 'shape_detail_K', 'shape_detail_N', 
    'latency_base', 'latency', 'gbps_base', 'gbps', 
    'speedup', 'accuracy', 'tflops', 'utilization', 'error_msg',
])

lines = json_data

for line in lines:
    if line.startswith('[INFO]'):
        json_str = line.split('[INFO] ')[1]
        data = json.loads(json_str)
        
        # autotune_configs       
        autotune_configs = data['autotune_configs']

        # result
        results = data['result']
        
        rows = []
        for result in results:
            shape_detail = result['shape_detail']
            row = {
                'op_name': data['op_name'],
                'dtype': data['dtype'],
                'mode': data['mode'],
                'level': data['level'],
                'BLOCK_M': autotune_configs['BLOCK_M'],
                'BLOCK_N': autotune_configs['BLOCK_N'],
                'BLOCK_K': autotune_configs['BLOCK_K'],
                'SPLIT_K': autotune_configs['SPLIT_K'],
                'num_stages': autotune_configs['num_stages'],
                'num_warps': autotune_configs['num_warps'],
                'num_ctas': autotune_configs['num_ctas'],
                'legacy_shape': result['legacy_shape'],
                'shape_detail_M': shape_detail[0][0],
                'shape_detail_K': shape_detail[0][1],
                'shape_detail_N': shape_detail[1][1],
                'latency_base': result['latency_base'],
                'latency': result['latency'],
                'gbps_base': result['gbps_base'],
                'gbps': result['gbps'],
                'speedup': result['speedup'],
                'accuracy': result['accuracy'],
                'tflops': result['tflops'],
                'utilization': result['utilization'],
                'error_msg': result['error_msg']
            }
        
            # df = df.append(row, ignore_index=True)
            rows.append(row)

        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

df.to_excel(output_excel, index=False)

print("Written to " + output_excel)
