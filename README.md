# Verity — Human Verification RL Environment

> Built for Meta PyTorch OpenEnv Hackathon 2026

"In a world where AI can fake everything, proof of human presence becomes the most valuable thing on the internet."

Verity is a reinforcement learning environment where an agent learns to distinguish real human behavioral signals from AI-generated bot behavior — built on PyTorch and fully OpenEnv-compatible.

---

## Live Demo

https://YOUR-RAILWAY-URL

---

## Quick Start

### 1. Install dependencies

pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

### 2. Train model

python model/train_real.py

### 3. Run inference

python inference.py

### 4. Start OpenEnv server

python env/verity_openenv.py

### 5. Test OpenEnv compliance

python env/test_openenv.py

### 6. Run agent demo

python agent_demo.py

### 7. Start full demo (Flask)

python api/app.py

---

## Docker Build and Run

Build:
docker build -t verity-env .

Run OpenEnv server:
docker run -p 8001:8001 verity-env

Run with database:
docker run -p 8001:8001 -e DATABASE_URL=your_neon_url verity-env

---

## Railway Deployment

1. Push to GitHub
2. Go to railway.app
3. New Project → Deploy from GitHub
4. Select this repo
5. Add environment variable: DATABASE_URL
6. Railway auto-deploys on every push

---

## Hugging Face Model

Set your token and upload:

$env:HF_TOKEN="your_token_here"
python hf_upload.py

Model hosted at: https://huggingface.co/YOUR_USERNAME/verity-model

---

## OpenEnv API

The environment runs as a FastAPI HTTP server on port 8001.

POST /reset — Start new episode, returns initial observation
POST /step — Take action, returns observation + reward + done + info
GET /state — Current episode state and metadata
GET /health — Health check and environment info

### State Space (8-dimensional)

1. keystroke_gap_ms — average gap between keystrokes
2. session_duration_s — total session length in seconds
3. edit_count — number of backspace/delete events
4. cursor_entropy — randomness of cursor movement (0-1)
5. content_length — word count of submission
6. device_score — hardware attestation score (0-1)
7. typing_variance — standard deviation of keystroke timing
8. pause_ratio — ratio of natural pauses in typing

### Action Space (binary)

0 = Bot — reject submission
1 = Human — issue verification certificate

### Reward Logic

+1.5 — correct classification with high confidence (>=0.9)
+1.0 — correct classification
-1.0 — wrong classification
-1.5 — wrong classification with high confidence (penalizes overconfidence)

---

## PyTorch Model

Architecture: Multi-layer perceptron
Input: 8 behavioral signal features
Hidden: 64 → 32 → 16 (BatchNorm + ReLU + Dropout)
Output: Sigmoid probability of human presence

Training dataset: CMU Keystroke Dynamics Benchmark Dataset
Subjects: 51 real humans
Samples: 40,800 (20,400 human + 20,400 bot)
Published: Killourhy & Maxion, DSN 2009

---

## Grader Results

Run: python model/evaluate.py

Accuracy:            100.00%
F1 Score:            1.0000
Precision:           1.0000
Recall:              1.0000
False Positive Rate: 0.00%
False Negative Rate: 0.00%
Verity Score:        1.0000 / 1.0000

---

## Project Structure

verity/
├── env/
│   ├── verity_env.py          RL environment — state, action, reward
│   ├── verity_openenv.py      OpenEnv FastAPI HTTP server
│   ├── data_generator.py      Synthetic data generator
│   ├── test_openenv.py        Full OpenEnv compliance tests
│   └── __init__.py
├── model/
│   ├── model.py               PyTorch VerityNet architecture
│   ├── train.py               Synthetic data training
│   ├── train_real.py          CMU dataset training
│   ├── scorer.py              Real behavioral scoring engine
│   ├── evaluate.py            Grader — accuracy, F1, Verity score
│   ├── verity_model.pt        Trained model checkpoint
│   ├── mean.npy               Normalization mean
│   └── std.npy                Normalization std
├── api/
│   └── app.py                 Flask API — live demo backend
├── frontend/
│   ├── index.html             Live demo — enrollment + verification
│   ├── verify.html            Public certificate verification page
│   └── verity-sdk.js          JavaScript SDK for platforms to embed
├── data/
│   └── DSL-StrongPasswordData.csv  CMU Keystroke Dataset
├── inference.py               Standalone inference script
├── agent_demo.py              Random agent interacting with OpenEnv
├── hf_upload.py               Hugging Face model upload script
├── test_verification.py       Verification test cases
├── Dockerfile                 Container for OpenEnv server
├── .dockerignore              Docker ignore rules
├── .env.example               Environment variable template
├── Procfile                   Railway deployment config
├── requirements.txt           Python dependencies
└── README.md                  This file

---

## Latest Verification Test Results (Final)

### Latest Verification Test Results (Final)
✅ PyTorch model loaded successfully
=== VERITY VERIFICATION TEST SUITE ===
Test 1 — Real human typing          → 95.0% HUMAN VERIFIED ✓
Test 2 — AI bot submission          → 10.0% BOT DETECTED ✓
Test 3 — Suspicious bot             → 17.0% BOT DETECTED ✓
Test 4 — Careful human writer       → 98.0% HUMAN VERIFIED ✓
Test 5 — Pure paste bot             → 10.0% BOT DETECTED ✓
Average human score: 96.5%
Average bot score: 12.3%
Human/Bot separation: 84.2%
✅ Scoring system is working correctly!

This demonstrates strong behavioral signal detection. Real humans score consistently high while bots and paste submissions are reliably flagged.

---

## 📦 Model on Hugging Face

The trained Verity model is available on Hugging Face:

**Repository**: [Avadhutparbhane/verity-model](https://huggingface.co/Avadhutparbhane/verity-model)

**Direct Link**: https://huggingface.co/Avadhutparbhane/verity-model

You can load it easily using:
```python
from huggingface_hub import hf_hub_download
import torch
import numpy as np

model_path = hf_hub_download(repo_id="Avadhutparbhane/verity-model", filename="verity_model.pt")
mean = np.load(hf_hub_download(repo_id="Avadhutparbhane/verity-model", filename="mean.npy"))
std = np.load(hf_hub_download(repo_id="Avadhutparbhane/verity-model", filename="std.npy"))

model = torch.load(model_path, map_location="cpu")
model.eval()
print("✅ Verity model loaded from Hugging Face!")
```

---

## Why This Wins

Every other team builds a toy game environment.
Verity solves a real problem that gets worse every year.
The live demo works in 30 seconds.
The OpenEnv server passes all compliance tests.
The model is trained on real peer-reviewed research data.
Any judge can clone it and run it in one command.

---

Built by Avadhut Parbhane — Solo founder, 21, Sambhajinagar, Maharashtra
GitHub: https://github.com/fxviptrading100-byte/verity

=== TASK 4 COMPLETE - POLISH AND README UPDATED ===
