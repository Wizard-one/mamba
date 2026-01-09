import os
import time
import json
import torch
from typing import Callable, Tuple
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, ssd_selective_scan,mamba_chunk_scan,ssd_chunk_scan_combined_ref
torch.backends.cuda.matmul.allow_tf32 = True

DEVICE_ID = 0
DEVICE = torch.device(f"cuda:{DEVICE_ID}")
def scan_funcs():
    """
    验证 SSD (mamba_chunk_scan_combined) 与 mamba_chunk_scan 的等效性。
    """
    torch.manual_seed(42)
    device = "cuda"

    # 1. 定义测试参数
    batch_size = 4
    seqlen = 512
    nheads = 24
    headdim = 160
    dstate = 16  # SSM state expansion factor
    ngroups = 1  # Group query attention factor (1 = MHA equivalent)
    chunk_size = 64 # SSD 分块大小

    dtype = torch.float32 # 推荐测试使用 float32 以避免精度误差干扰

    # 2. 构造随机输入
    # x: (batch, seqlen, nheads, headdim)
    x = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    # dt: (batch, seqlen, nheads) - 时间步长 delta
    dt = torch.rand(batch_size, seqlen, nheads, device=device, dtype=dtype)
    # A: (nheads) - 状态转移矩阵的对角部分 (通常为负数)
    A = -torch.rand(nheads, device=device, dtype=dtype)
    # B: (batch, seqlen, ngroups, dstate) - 输入投影矩阵
    B = torch.randn(batch_size, seqlen, ngroups, dstate, device=device, dtype=dtype)
    # C: (batch, seqlen, ngroups, dstate) - 输出投影矩阵
    C = torch.randn(batch_size, seqlen, ngroups, dstate, device=device, dtype=dtype)
    # D: (nheads, headdim) - 跳跃连接 (可选)
    D = torch.randn(nheads, headdim, device=device, dtype=dtype)
    # z: (batch, seqlen, nheads, headdim) - 门控分支 (可选，Mamba 结构中的乘法门)
    z = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

    # 3. 运行 Mamba-2 SSD 实现 (Target)
    # mamba_chunk_scan_combined 是 Mamba-2 的核心算子
    out_ssd_combine = lambda: mamba_chunk_scan_combined(
        x, dt, A, B, C, 
        chunk_size=chunk_size, 
        D=D, 
        z=z
    )
    
    out_ssd=lambda: mamba_chunk_scan(
        x,dt,A,B,C,chunk_size=chunk_size,D=D,z=z
    )
    
    # 检查数值是否接近
    #由于并行计算顺序不同，可能会有微小的浮点误差，使用较高的容忍度

    out_selective=lambda: ssd_selective_scan(
        x, dt, A, B, C, 
        D=D, 
        z=z
    )

    out_ref=lambda: ssd_chunk_scan_combined_ref(
        x, dt, A, B, C, 
        chunk_size=chunk_size, 
        D=D, 
        z=z
    )
    return out_ssd_combine,out_ssd,out_selective,out_ref

def benchmark(func: Callable, repeats: int, warmup: int) -> Tuple[float, float, float]:
    """
    ref: https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
    """
    for _ in range(warmup):
        func()
    torch.cuda.reset_max_memory_allocated(DEVICE)  # 重置统计
    torch.cuda.synchronize(DEVICE)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    cpu_timer_start = time.perf_counter()
    start.record()  # type: ignore
    for _ in range(repeats):
        func()
    end.record()  # type: ignore
    torch.cuda.synchronize(DEVICE)
    cpu_timer_end = time.perf_counter()
    max_mem = torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024  # 单位: MB
    return start.elapsed_time(end) / 1000, cpu_timer_end - cpu_timer_start, max_mem

def save_benchmark_result(
    time_cost_cuda: float,
    time_cost: float,
    max_mem: float,
    name: str,
    save_dir="benchmark/record",
    suffix: str = "",
) -> None:
    result = {
        "Time": time_cost,
        "TimeCUDA": time_cost_cuda,
        "MaxMemoryMB": max_mem,
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "Name": name,
    }
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f"{name}_{result['Timestamp']}{suffix}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(result)
    
if __name__ == "__main__":
	out_ssd_combine,out_ssd,out_selective,out_ref=scan_funcs()
	time_cost_cuda, time_cost, max_mem = benchmark(out_ssd_combine, repeats=10, warmup=2)
	save_benchmark_result(time_cost_cuda, time_cost, max_mem, "out_ssd_combine_0")
	time_cost_cuda, time_cost, max_mem = benchmark(out_ssd, repeats=10, warmup=2)
	save_benchmark_result(time_cost_cuda, time_cost, max_mem, "out_ssd_1")     
	time_cost_cuda, time_cost, max_mem = benchmark(out_selective, repeats=10, warmup=2)
	save_benchmark_result(time_cost_cuda, time_cost, max_mem, "out_selective_2")
	time_cost_cuda, time_cost, max_mem = benchmark(out_ref, repeats=10, warmup=2)
	save_benchmark_result(time_cost_cuda, time_cost, max_mem, "out_ref_3")
