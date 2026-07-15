"""
UNetを用いた拡散モデルの学習とサンプリングの例
最終的に学習済みモデルを保存し、学習損失の推移をプロット
"""

import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # [-1,1] に正規化
])

dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


T = 300  # diffusion steps

beta = torch.linspace(1e-4, 0.02, T, device=device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        if half_dim == 0:
            return t[:, None]

        exponent = torch.arange(half_dim, device=t.device, dtype=t.dtype)
        exponent = -torch.log(torch.tensor(10000.0, device=t.device, dtype=t.dtype)) * exponent / max(half_dim - 1, 1)
        emb = t[:, None] * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)

        return emb


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.ReLU(),
        )
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

    def forward(self, x, t_emb):
        h = self.conv(x)
        t = self.time_mlp(t_emb)[:, :, None, None]
        return h + t

    
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()

        num_groups = 8 if out_ch >= 8 else 1

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, out_ch)

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = torch.relu(self.gn1(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = torch.relu(self.gn2(self.conv2(h)))
        return h + self.shortcut(x)


class FinalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

    def forward(self, x, t_emb):
        t = self.time_mlp(t_emb)[:, :, None, None]
        return self.conv(x) + t


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        time_dim = 256

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Down: 28 -> 14 -> 7
        self.down1 = ResBlock(1, 64, time_dim)
        self.down2 = ResBlock(64, 128, time_dim)
        self.pool = nn.MaxPool2d(2)

        # Middle: 7x7
        self.mid = ResBlock(128, 128, time_dim)

        # Up: 7 -> 14 -> 28
        self.up2 = ResBlock(128 + 128, 64, time_dim)
        self.up1 = FinalBlock(64 + 64, 1, time_dim)

    def forward(self, x, t):
        t = t.view(-1).float() / (T - 1)
        t_emb = self.time_mlp(self.time_emb(t))

        # Down
        h1 = self.down1(x, t_emb)          # 28x28
        h2 = self.down2(self.pool(h1), t_emb)  # 14x14

        mid = self.mid(self.pool(h2), t_emb)   # 7x7

        # Up
        u2 = nn.functional.interpolate(mid, scale_factor=2)  # 14x14
        u2 = self.up2(torch.cat([u2, h2], dim=1), t_emb)

        u1 = nn.functional.interpolate(u2, scale_factor=2)   # 28x28
        out = self.up1(torch.cat([u1, h1], dim=1), t_emb)

        return out

model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
num_epochs = int(os.getenv("DIFFUSION_EPOCHS", "50"))


def sample_xt(x0, t):
    ab = alpha_bar[t]
    eps = torch.randn_like(x0)
    xt = torch.sqrt(ab)[:, None, None, None] * x0 + \
         torch.sqrt(1 - ab)[:, None, None, None] * eps
    return xt, eps


loss_history = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for x, _ in dataloader:
        x = x.to(device)
        print(f"x_size: {x.size()}")  # デバッグ用にバッチサイズを表示

        t = torch.randint(0, T, (x.size(0),), device=device).float()

        xt, eps = sample_xt(x, t.long())

        pred = model(xt, t[:, None])

        loss = nn.functional.mse_loss(pred, eps)
        print(f"pred_size: {pred.size()}, eps_size: {eps.size()}, loss: {loss.item()}")  # デバッグ用にサイズと損失を表示

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1
        exit()

    avg_loss = epoch_loss / max(num_batches, 1)
    loss_history.append(avg_loss)
    print(f"epoch {epoch} loss {avg_loss:.4f}")


@torch.no_grad()
def sample(model):
    model.eval()
    x = torch.randn(1, 1, 28, 28, device=device)

    for t in reversed(range(T)):
        t_tensor = torch.tensor([t], device=device).float()
        eps = model(x, t_tensor[:, None])

        a = alpha[t]
        ab = alpha_bar[t]
        b = beta[t]
        ab_prev = alpha_bar[t - 1] if t > 0 else torch.tensor(1.0, device=device)

        # DDPM の逆拡散の平均
        mean = (1 / torch.sqrt(a)) * (x - (1 - a) / torch.sqrt(1 - ab) * eps)

        if t > 0:
            posterior_var = b * (1 - ab_prev) / (1 - ab)
            x = mean + torch.sqrt(posterior_var) * torch.randn_like(x)
        else:
            x = mean

    return x


img = sample(model).cpu().squeeze().numpy()
plt.imshow(img, cmap="gray")
plt.show()

plt.figure()
plt.plot(loss_history)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Training loss")
plt.grid(True)
plt.show()

# モデル保存
save_path = "./data/diffusion_unet.pth"
torch.save(model.state_dict(), save_path)
print(f"Saved model to {save_path}")
