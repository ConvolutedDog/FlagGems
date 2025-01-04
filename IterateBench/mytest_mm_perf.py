import os
import torch
import shutil
import flag_gems

def remove_triton_cache():
    cache_dir = os.path.expanduser("~/.triton/cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Deleted Triton cache directory: {cache_dir}")
    else:
        print(f"Triton cache directory does not exist: {cache_dir}")

if __name__ == "__main__":
    
    remove_triton_cache()
    
    m = n = k = 2048
    
    inp1 = torch.randn([m, k], dtype=torch.float16, device=torch.device("cuda:0"))
    inp2 = torch.randn([k, n], dtype=torch.float16, device=torch.device("cuda:0"))
    
    with flag_gems.use_gems():
        torch.mm(inp1, inp2)
