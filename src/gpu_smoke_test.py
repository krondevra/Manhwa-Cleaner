"""Quick GPU (ROCm) smoke test for this machine's 890M iGPU.

Run with the required workaround env var:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 <venv>/bin/python src/gpu_smoke_test.py

IMPORTANT: includes a BatchNorm2d layer on purpose. On gfx1151, MIOpen's JIT
kernel compiler fails on BatchNorm specifically (miopenStatusUnknownError /
"cannot compile inline asm") while a bare Conv2d passes -- so a conv-only
smoke test gives false confidence (documented hardware lesson, see
.tmp/notes/gpu_training_operational_gotchas.md and next_session_handoff.md).
"""
import time
import torch
import torch.nn as nn

device = torch.device("cuda")
print(f"device: {torch.cuda.get_device_name(0)}")

# BatchNorm included deliberately -- the known MIOpen failure mode on this GPU.
model = nn.Sequential(
    nn.Conv2d(7, 24, 3, padding=1),
    nn.BatchNorm2d(24),
    nn.ReLU(),
    nn.Conv2d(24, 24, 3, padding=1),
).to(device)
x = torch.randn(2, 7, 512, 512, device=device, requires_grad=True)

# warmup (includes one-time MIOpen JIT kernel compile)
t0 = time.time()
for _ in range(2):
    y = model(x)
    y.sum().backward()
torch.cuda.synchronize()
print(f"warmup (incl. MIOpen JIT): {time.time()-t0:.1f}s")

n = 10
t0 = time.time()
for _ in range(n):
    y = model(x)
    loss = y.sum()
    loss.backward()
torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"{n} conv+batchnorm forward+backward iters: {elapsed:.3f}s total, {elapsed/n*1000:.1f}ms/iter")
print("OK: GPU path works including BatchNorm")
