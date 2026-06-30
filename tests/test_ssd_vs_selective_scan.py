import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, ssd_selective_scan,mamba_chunk_scan,ssd_chunk_scan_combined_ref
from einops import repeat,rearrange
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


def test_selective_scan_group_vs_ssd_selective_scan():
    d_model = 80
    d_state = 16
    DEVICE = torch.device("cuda:0")
    batch,L=2,100
    Group=8

    u = torch.randn(batch,Group, d_model, L, device=DEVICE)
    z = u.clone()
    delta = torch.randn(batch,Group, d_model, L, device=DEVICE)
    A = repeat(
        torch.arange(1, d_state + 1, dtype=torch.float32, device=DEVICE),
        "n -> d n",
        d=d_model,
    ).contiguous()
    A=repeat(A,"d n -> g d n",g=Group).contiguous()
    A = torch.log(A)  # Keep A_log in fp32
    A = -torch.exp(A.float())
    B = torch.randn(batch,Group, d_state, L, device=DEVICE)
    C = torch.randn(batch,Group, d_state, L, device=DEVICE)
    D = torch.randn(Group,d_model, device=DEVICE)
    delta_bias = torch.randn(Group,d_model, device=DEVICE)    
    u_ssd=rearrange(u,"b g d l -> b l g d")
    dt_ssd=rearrange(delta,"b g d l -> b l g d")
    B_ssd=rearrange(B,"b g n l -> b l g n")
    C_ssd=rearrange(C,"b g n l -> b l g n")
    z_ssd=rearrange(z,"b g d l -> b l g d")
    A_ssd=A.view(Group*d_model,d_state)
    y=ssd_selective_scan(u_ssd,dt_ssd,A_ssd,B_ssd,C_ssd,D.float(),z=z_ssd,dt_bias=delta_bias.float(),dt_softplus=True)
    mask = torch.isnan(y)
    num_nans = int(mask.sum().item())
    print(f"Number of NaNs in output: {num_nans}")

    u_group=u.view(batch,Group*d_model,L)
    z_group=z.view(batch,Group*d_model,L)
    dt_group=delta.view(batch,Group*d_model,L)
    A_group=A.view(Group*d_model,d_state)
    D_group=D.view(Group*d_model)
    delta_bias_group=delta_bias.view(Group*d_model)
    out_ref = selective_scan_fn(
        u_group,
        dt_group,
        A_group,
        B,
        C,
        D_group.float(),
        z=z_group,
        delta_bias=delta_bias_group.float(),
        delta_softplus=True,        
    )
    out_ref = out_ref.view(batch,Group,d_model,L).permute(0, 3, 1, 2)  # (batch, L, Group, d_model)
    torch.testing.assert_close(y, out_ref)


if __name__ == "__main__":
    test_selective_scan_group_vs_ssd_selective_scan()