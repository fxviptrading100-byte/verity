# Verity — Human Verification RL Environment

> "In a world where AI can fake everything, proof of human presence becomes the most valuable primitive on the internet."

Built for Meta PyTorch Hackathon 2026.

---

## What is Verity?

Verity is a reinforcement learning environment where an agent learns to distinguish real human behavioral signals from AI-generated bot behavior.

It is not a toy game. It is real-world AI safety infrastructure — a verification layer that proves a human was present at the moment of content creation.

---

## The Problem

AI can now generate perfect fake reviews, job applications, legal documents, and research papers in milliseconds. Existing detectors guess after the fact and fail constantly. Verity solves this at the source — by certifying human presence at creation, not detecting fakeness afterward.

---

## RL Environment

State space: 8-dimensional behavioral signal vector
- keystroke_gap_ms — average gap between keystrokes
- session_duration_s — total session length
- edit_count — number of edits and deletions
- cursor_entropy — randomness of cursor movement
- content_length — word count of submission
- device_score — hardware attestation score
- typing_variance — standard deviation of keystroke timing
- pause_ratio — ratio of natural pauses in typing

Action space: Binary
- 0 = Bot (reject submission)
- 1 = Human (issue verification certificate)

Reward logic:
- +1.5 correct classification with high confidence (>=0.9)
- +1.0 correct classification
- -1.0 wrong classification
- -1.5 wrong classification with high confidence (penalizes overconfident errors)

---

## PyTorch Model

Architecture: Multi-layer perceptron with batch normalization
- Input: 8 behavioral signal features
- Hidden layers: 64 -> 32 -> 16 with BatchNorm + ReLU + Dropout
- Output: Sigmoid probability of human presence

Training:
- Dataset: 20,000 synthetic behavioral samples (50/50 human/bot)
- Optimizer: Adam with weight decay
- Loss: Binary Cross Entropy
- Epochs: 50 with learning rate scheduling

---

## Grader Results

Accuracy:            100.00%
F1 Score:            1.0000
Precision:           1.0000
Recall:              1.0000
False Positive Rate: 0.00%
False Negative Rate: 0.00%
Verity Score:        1.0000 / 1.0000

Evaluated on 2,000 unseen test samples (seed=999).

---

## How to Run

Install dependencies:
pip install torch numpy flask flask-cors scikit-learn pandas

Train the model:
python model/train.py

Run the grader:
python model/evaluate.py

Start the API:
python api/app.py

Open the demo:
http://127.0.0.1:5000

---

## Project Structure

verity/
├── env/
│   ├── verity_env.py        RL environment — state, action, reward
│   └── data_generator.py    Synthetic behavioral dataset generator
├── model/
│   ├── model.py             PyTorch neural network architecture
│   ├── train.py             Training loop
│   └── evaluate.py          Grader — accuracy, F1, Verity score
├── api/
│   └── app.py               Flask API — POST /verify, GET /certificate/:id
└── frontend/
    └── index.html           Live demo — real-time human verification

---

## Live Demo

Start the Flask server and open http://127.0.0.1:5000

Type anything naturally. Verity captures your behavioral signals silently,
runs them through the PyTorch model, and issues a tamper-proof certificate
proving human presence — in real time.

---

## Why This Matters

By 2028 the majority of internet interactions will be machine-generated.
Every system built assuming a human is on the other end — hiring, legal,
financial, academic — will need a trust layer underneath it.

Verity is that layer.

Built by Avadhut Parbhane — Solo founder, Sambhajinagar, Maharashtra.
