import sys
import os
import torch

# 1) Ensure Python can import fairseq_signals
sys.path.append(os.path.join(os.getcwd(), "fairseq-signals"))

from fairseq_signals.models import build_model_from_checkpoint

# 2) Point to your finetuned checkpoint under ckpts/
ckpt_path = os.path.join("ckpts", "mimic_iv_ecg_finetuned.pt")

# 3) Build (load) the model from that checkpoint
model = build_model_from_checkpoint(ckpt_path)
model.eval()

# 4) Create a fake ECG tensor: batch=1, 12 leads, 5000 samples
x = torch.randn(1, 12, 5000)

# 5) Run inference – note we pass via keyword `source`
with torch.no_grad():
    out = model(source=x)

# 6) Extract the logits (they’re under the "out" key)
logits = out["out"]

print("✅ Inference successful! Logits shape:", logits.shape)
