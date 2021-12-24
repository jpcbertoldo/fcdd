"""
src: https://stackoverflow.com/a/48152675/9582881
"""

from torch import cuda

print(f"{(ngpus := cuda.device_count())=} {cuda.current_device()=}\n")

for gpu_idx in range(ngpus):
    print(f"{gpu_idx=} {cuda.device(gpu_idx)=} {cuda.get_device_name(gpu_idx)=}\n")

