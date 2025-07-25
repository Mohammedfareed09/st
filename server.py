from flask import Flask, request, jsonify
import torch
import numpy as np
import json
import sys, os

# Add fairseq-signals path
sys.path.append(os.path.join(os.getcwd(), "fairseq-signals"))
from fairseq_signals.models import build_model_from_checkpoint

app = Flask(__name__)

# Load model once
ckpt_path = os.path.join("ckpts", "mimic_iv_ecg_finetuned.pt")
model = build_model_from_checkpoint(ckpt_path)
model.eval()

label_names = [
    "Myocardial Infarction", "Atrial Fibrillation", "Bundle Branch Block",
    "Class_3", "Class_4", "Class_5", "Class_6", "Class_7", "Class_8",
    "Class_9", "Class_10", "Class_11", "Class_12", "Class_13", "Class_14",
    "Class_15", "Class_16"
]

def interpret(p):
    if p < 0.05: return "impossible"
    elif p < 0.2: return "good"
    elif p < 0.5: return "okay"
    elif p < 0.8: return "monitor"
    else: return "dangerous"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('ecg', [])
    if not data or len(data) < 100:
        return jsonify({'error': 'Invalid ECG data'}), 400

    signal = np.array(data, dtype=np.float32)
    x = torch.from_numpy(signal).unsqueeze(0).unsqueeze(0)  # (1,1,L)
    x = x.repeat(1, 12, 1)  # (1,12,L)

    with torch.no_grad():
        out = model(source=x)
        probs = torch.sigmoid(out["out"])[0].tolist()

    results = []
    for name, p in zip(label_names, probs):
        results.append({
            "class": name,
            "probability": round(p, 3),
            "interpretation": interpret(p),
        })

    return jsonify({"predictions": results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
