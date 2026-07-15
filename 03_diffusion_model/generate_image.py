"""
保存した学習済みモデルを用いて画像を生成するコード
"""

import torch
import matplotlib.pyplot as plt
import os

# ======== モデル定義（学習時と同じ） ========

T = 300
device = "cuda" if torch.cuda.is_available() else "cpu"

beta = torch.linspace(1e-4, 0.02, T, device=device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

class SinusoidalTimeEmbedding(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=t.device, dtype=t.dtype)
        exponent = -torch.log(torch.tensor(10000.0, device=t.device)) * exponent / max(half_dim - 1, 1)
        emb = t[:, None] * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb

class ResBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        num_groups = 8 if out_ch >= 8 else 1
        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(num_groups, out_ch)
        self.conv2 = torch.nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(num_groups, out_ch)
        self.time_mlp = torch.nn.Linear(time_emb_dim, out_ch)
        self.shortcut = torch.nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else torch.nn.Identity()

    def forward(self, x, t_emb):
        h = torch.relu(self.gn1(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = torch.relu(self.gn2(self.conv2(h)))
        return h + self.shortcut(x)

class FinalBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = torch.nn.Linear(time_emb_dim, out_ch)

    def forward(self, x, t_emb):
        return self.conv(x) + self.time_mlp(t_emb)[:, :, None, None]

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        time_dim = 256
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(time_dim, time_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(time_dim, time_dim),
        )
        self.down1 = ResBlock(1, 64, time_dim)
        self.down2 = ResBlock(64, 128, time_dim)
        self.pool = torch.nn.MaxPool2d(2)
        self.mid = ResBlock(128, 128, time_dim)
        self.up2 = ResBlock(128 + 128, 64, time_dim)
        self.up1 = FinalBlock(64 + 64, 1, time_dim)

    def forward(self, x, t):
        t = t.view(-1).float() / (T - 1)
        t_emb = self.time_mlp(self.time_emb(t))
        h1 = self.down1(x, t_emb)
        h2 = self.down2(self.pool(h1), t_emb)
        mid = self.mid(self.pool(h2), t_emb)
        u2 = torch.nn.functional.interpolate(mid, scale_factor=2)
        u2 = self.up2(torch.cat([u2, h2], dim=1), t_emb)
        u1 = torch.nn.functional.interpolate(u2, scale_factor=2)
        return self.up1(torch.cat([u1, h1], dim=1), t_emb)

# ======== モデル読み込み ========

model = UNet().to(device)
model.load_state_dict(torch.load("./data/diffusion_unet.pth", map_location=device))
model.eval()

# ======== 画像生成関数 ========

@torch.no_grad()
def sample(model):
    x = torch.randn(1, 1, 28, 28, device=device)
    for t in reversed(range(T)):
        t_tensor = torch.tensor([t], device=device).float()
        eps = model(x, t_tensor[:, None])
        a = alpha[t]
        ab = alpha_bar[t]
        b = beta[t]
        ab_prev = alpha_bar[t - 1] if t > 0 else torch.tensor(1.0, device=device)
        mean = (1 / torch.sqrt(a)) * (x - (1 - a) / torch.sqrt(1 - ab) * eps)
        if t > 0:
            posterior_var = b * (1 - ab_prev) / (1 - ab)
            x = mean + torch.sqrt(posterior_var) * torch.randn_like(x)
        else:
            x = mean
    return x

# ======== 複数枚生成 ========

os.makedirs("./data/generated_images", exist_ok=True)

for i in range(10):
    img = sample(model).cpu().squeeze().numpy()
    plt.imsave(f"./data/generated_images/sample_{i}.png", img, cmap="gray")
    print(f"Saved ./data/generated_images/sample_{i}.png")
