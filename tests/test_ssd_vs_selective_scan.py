import torch
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, ssd_selective_scan,mamba_chunk_scan,ssd_chunk_scan_combined_ref
torch.backends.cuda.matmul.allow_tf32 = True  # The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
torch.backends.cudnn.allow_tf32 = True  # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
def test_mamba_chunk_scan_combined_vs_mamba_chunk_scan():
    """
    验证 SSD (mamba_chunk_scan_combined) 与 mamba_chunk_scan 的等效性。
    """
    torch.manual_seed(42)
    device = "cuda"

    # 1. 定义测试参数
    batch_size = 1
    seqlen = 128
    nheads = 4
    headdim = 64
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
    out_ssd = mamba_chunk_scan_combined(
        x, dt, A, B, C, 
        chunk_size=chunk_size, 
        D=D, 
        z=z
    )
    
    out_ref=mamba_chunk_scan(
        x,dt,A,B,C,chunk_size=chunk_size,D=D,z=z
    )
    # 5. 验证结果
    # 检查输出形状是否一致
    assert out_ssd.shape == out_ref.shape, f"Shape mismatch: {out_ssd.shape} vs {out_ref.shape}"
    
    # 检查数值是否接近
    #由于并行计算顺序不同，可能会有微小的浮点误差，使用较高的容忍度
    torch.testing.assert_close(out_ssd, out_ref, rtol=1e-3, atol=1e-3)

    out_selective=ssd_selective_scan(
        x, dt, A, B, C, 
        D=D, 
        z=z
    )
    torch.testing.assert_close(out_ssd, out_selective,msg="SSD 与 Selective Scan 会不匹配")# NOTE: 此处两个代码实现相同功能,但数值会有差异

    out_ssd_ref=ssd_chunk_scan_combined_ref(
        x, dt, A, B, C, 
        chunk_size=chunk_size, 
        D=D, 
        z=z
    )
    torch.testing.assert_close(out_ssd, out_ssd_ref, rtol=1e-3, atol=1e-3,msg="Difference between out_ssd and out_ssd_ref") # NOTE: 此处两个代码实现相同功能,但数值会有差异


    

if __name__ == "__main__":
    test_mamba_chunk_scan_combined_vs_mamba_chunk_scan()