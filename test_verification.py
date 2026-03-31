import torch
import numpy as np
import sys
sys.path.insert(0, 'C:\\Users\\Admin\\Desktop\\verity')
from model.model import VerityNet

checkpoint = torch.load('model/verity_model.pt', map_location='cpu')
model = VerityNet(input_dim=checkpoint['input_dim'], hidden_dims=checkpoint['hidden_dims'])
model.load_state_dict(checkpoint['model_state'])
model.eval()

mean = np.load('model/mean.npy')
std = np.load('model/std.npy')

def verify(signals, label):
    x = (np.array(signals, dtype=np.float32) - mean) / (std + 1e-8)
    t = torch.FloatTensor(x).unsqueeze(0)
    with torch.no_grad():
        prob = model(t).item()
    verdict = 'HUMAN VERIFIED' if prob >= 0.5 else 'BOT DETECTED'
    print(f'\n{label}')
    print(f'Human score: {prob:.2%}')
    print(f'Verdict:     {verdict}')

# Real human signals
verify([187, 272, 21, 0.82, 312, 0.91, 4.2, 0.78], 'Test 1 — Real human typing')

# Bot signals
verify([5, 1, 0, 0.02, 300, 0.1, 0.01, 0.01], 'Test 2 — AI bot submission')

# Edge case — fast but some edits
verify([25, 8, 3, 0.12, 280, 0.25, 0.08, 0.03], 'Test 3 — Suspicious bot')

# Slow careful human
verify([220, 480, 35, 0.91, 450, 0.95, 5.1, 0.85], 'Test 4 — Careful human writer')

# Very fast paste — bot
verify([3, 0.5, 0, 0.01, 500, 0.05, 0.005, 0.001], 'Test 5 — Pure paste bot')
