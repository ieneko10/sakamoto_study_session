"""
拡散過程のシミュレーションを行うコード
画像が徐々にノイズで拡散していく様子を確認できる
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# 保存先フォルダ
os.makedirs("./data/noisy_images", exist_ok=True)
os.makedirs("./data/noise_eps", exist_ok=True)   # ← ノイズ保存用フォルダ追加

# 画像読み込み
img = Image.open("./data/sample.png").convert("RGB")

# [-1,1] に正規化
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

x0 = transform(img)  # shape: [3,256,256]
x0 = x0.unsqueeze(0)  # [1,3,256,256]

# ノイズステップ数
steps = 10
alphas = torch.linspace(1.0, 0.0, steps)  # α_t を段階的に減らす

for i, a in enumerate(alphas):
    # DDPM の forward process と同じ式
    noise = torch.randn_like(x0)  # ← これが ε_t
    xt = torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise

    # xt を画像として保存するために [0,1] に戻す
    xt_img = xt.squeeze(0)
    xt_img = xt_img * 0.5 + 0.5  # [-1,1] → [0,1]
    xt_img = xt_img.clamp(0, 1)

    transforms.ToPILImage()(xt_img).save(f"./data/noisy_images/noise_step_{i}.png")
    print(f"Saved ./data/noisy_images/noise_step_{i}.png")

    # ε_t（ノイズ）も保存する
    eps_img = noise.squeeze(0)

    # ノイズは [-∞,∞] なので 0〜1 に正規化して保存
    eps_img = (eps_img - eps_img.min()) / (eps_img.max() - eps_img.min())

    transforms.ToPILImage()(eps_img).save(f"./data/noise_eps/eps_step_{i}.png")
    print(f"Saved ./data/                       noise_eps/eps_step_{i}.png")
