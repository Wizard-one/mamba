import math
import torch
from einops import rearrange, repeat
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn,selective_scan_ref


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