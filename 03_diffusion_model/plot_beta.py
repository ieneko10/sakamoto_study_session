"""
拡散過程におけるβスケジュールの例を示すコード
"""

import torch
import matplotlib.pyplot as plt

T = 1000  # diffusion steps

# Linear schedule
beta_linear = torch.linspace(1e-4, 0.02, T)

# Quadratic schedule
beta_quad = (torch.linspace(0, 1, T) ** 2) * (0.02 - 1e-4) + 1e-4

# Cosine schedule (Nichol & Dhariwal 2021)
t = torch.linspace(0, T, T)
s = 0.008
alpha_bar_cos = torch.cos(((t/T + s) / (1 + s)) * torch.pi / 2) ** 2
beta_cos = 1 - (alpha_bar_cos[1:] / alpha_bar_cos[:-1])
beta_cos = torch.cat([beta_cos[0:1], beta_cos])  # pad first value

# Plot side-by-side
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(beta_linear)
axes[0].set_title("Linear β")
axes[0].set_xlabel("t")
axes[0].set_ylabel("β_t")
axes[0].grid(True)

axes[1].plot(beta_quad)
axes[1].set_title("Quadratic β")
axes[1].set_xlabel("t")
axes[1].grid(True)

axes[2].plot(beta_cos)
axes[2].set_title("Cosine β")
axes[2].set_xlabel("t")
axes[2].grid(True)

plt.tight_layout()
plt.savefig("beta_schedules_side_by_side.png")
plt.show()
