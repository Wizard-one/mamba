import math
import torch
from einops import rearrange, repeat
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn,selective_scan_ref

def my_selective_scan_ref_group(u, delta, A, B, C, D, z, delta_bias):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    delta = delta + delta_bias[..., None].float()
    delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    B = B.float()
    C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
    deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y + u * rearrange(D, "d -> d 1")
    out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out

def my_selective_scan_ref(u, delta, A, B, C, D, z, delta_bias, delta_softplus=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    delta = delta + delta_bias[..., None].float()
    delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    B = B.float()
    C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y + u * rearrange(D, "d -> d 1")
    out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out
     

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

ys=[]
for g in range(Group):
	y_g = selective_scan_fn(
		u[:, g].contiguous(),
		delta[:, g].contiguous(),
		A[g],
		B[:, g].contiguous(),
		C[:, g].contiguous(),
		D[g].float(),
		z=z[:, g].contiguous(),
		delta_bias=delta_bias[g].float(), # Bias 加在这里
		delta_softplus=True,
	)
	ys.append(y_g)

y = torch.stack(ys, dim=1)

mask = torch.isnan(y)
num_nans = int(mask.sum().item())
print(f"Number of NaNs in output: {num_nans}")

u_group=u.view(batch,Group*d_model,L)
z_group=z.view(batch,Group*d_model,L)
delta_group=delta.view(batch,Group*d_model,L)
A_group=A.view(Group*d_model,d_state)
D_group=D.view(Group*d_model)
delta_bias_group=delta_bias.view(Group*d_model)
out_ref = selective_scan_fn(
    u_group,
    delta_group,
    A_group,
    B,
    C,
    D_group.float(),
    z=z_group,
    delta_bias=delta_bias_group.float(),
    delta_softplus=True,
)
out_ref = out_ref.view(batch,Group,d_model,L)
torch.testing.assert_close(y, out_ref)