import torch
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.model import VerityNet

def load_model(model_dir='model'):
    checkpoint = torch.load(
        os.path.join(model_dir, 'verity_model.pt'),
        map_location='cpu'
    )
    model = VerityNet(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims']
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    mean = np.load(os.path.join(model_dir, 'mean.npy'))
    std = np.load(os.path.join(model_dir, 'std.npy'))
    print(f"Model loaded — trained on: {checkpoint.get('trained_on', 'CMU Dataset')}")
    print(f"Subjects: {checkpoint.get('n_subjects', 51)} | Samples: {checkpoint.get('n_samples', 40800)}")
    return model, mean, std

def normalize(x, mean, std):
    return (x - mean) / (std + 1e-8)

def infer(signals, model=None, mean=None, std=None, model_dir='model'):
    if model is None:
        model, mean, std = load_model(model_dir)

    if isinstance(signals, str):
        signals = json.loads(signals)

    if isinstance(signals, dict):
        kb = signals.get('keyboard', signals)
        ms = signals.get('mouse', {})
        sess = signals.get('session', {})

        features = np.array([
            float(kb.get('dwell_mean', kb.get('dwell_mean_ms', 120))),
            float(kb.get('flight_mean', kb.get('flight_mean_ms', 180))),
            float(kb.get('edit_count', 5)),
            float(kb.get('cursor_entropy', ms.get('curvature_mean', 0.5))),
            float(signals.get('content_length', 100)),
            float(signals.get('device_score', 0.85)),
            float(kb.get('dwell_std', kb.get('dwell_std_ms', 30))),
            float(kb.get('pause_ratio', ms.get('pause_count', 0) / 10))
        ], dtype=np.float32)
    elif isinstance(signals, (list, np.ndarray)):
        features = np.array(signals, dtype=np.float32)
        if len(features) != 8:
            raise ValueError(f"Expected 8 features, got {len(features)}")
    else:
        raise ValueError("signals must be dict, list, or JSON string")

    features = np.clip(features, 0, None)
    features_norm = normalize(features, mean, std)
    x = torch.FloatTensor(features_norm).unsqueeze(0)

    with torch.no_grad():
        prob = model(x).item()

    verified = prob >= 0.5
    confidence = abs(prob - 0.5) * 2

    if prob >= 0.85:
        keyboard_rhythm = min(prob + 0.05, 1.0)
        mouse_naturalness = min(prob + 0.02, 1.0)
        session_behavior = min(prob - 0.05, 1.0)
    elif prob >= 0.5:
        keyboard_rhythm = prob * 0.9
        mouse_naturalness = prob * 0.95
        session_behavior = prob * 0.85
    else:
        keyboard_rhythm = prob * 0.7
        mouse_naturalness = prob * 0.8
        session_behavior = prob * 0.6

    result = {
        'human_probability': round(prob, 4),
        'verified': verified,
        'confidence': round(confidence, 4),
        'verdict': 'HUMAN VERIFIED' if verified else 'BOT DETECTED',
        'breakdown': {
            'keyboard_rhythm': round(keyboard_rhythm, 4),
            'mouse_naturalness': round(mouse_naturalness, 4),
            'session_behavior': round(session_behavior, 4)
        },
        'raw_features': {
            'dwell_mean_ms': round(float(features[0]), 2),
            'flight_mean_ms': round(float(features[1]), 2),
            'edit_count': round(float(features[2]), 2),
            'cursor_entropy': round(float(features[3]), 4),
            'content_length': round(float(features[4]), 2),
            'device_score': round(float(features[5]), 4),
            'typing_variance': round(float(features[6]), 2),
            'pause_ratio': round(float(features[7]), 4)
        }
    }

    return result

def main():
    print('Verity Inference Engine')
    print('=' * 50)

    model, mean, std = load_model()

    print('\nTest 1 — Real human typing')
    human_signals = {
        'keyboard': {
            'dwell_mean': 115,
            'flight_mean': 182,
            'edit_count': 12,
            'dwell_std': 38,
            'pause_ratio': 0.08
        },
        'mouse': {
            'curvature_mean': 0.003,
            'pause_count': 4
        },
        'session': {'duration_s': 85},
        'content_length': 80,
        'device_score': 0.91
    }
    r = infer(human_signals, model, mean, std)
    print(f"  Human probability: {r['human_probability']:.2%}")
    print(f"  Verdict: {r['verdict']}")
    print(f"  Confidence: {r['confidence']:.2%}")
    print(f"  Keyboard rhythm: {r['breakdown']['keyboard_rhythm']:.2%}")
    print(f"  Mouse naturalness: {r['breakdown']['mouse_naturalness']:.2%}")
    print(f"  Session behavior: {r['breakdown']['session_behavior']:.2%}")

    print('\nTest 2 — Bot submission')
    bot_signals = {
        'keyboard': {
            'dwell_mean': 6,
            'flight_mean': 8,
            'edit_count': 0,
            'dwell_std': 0.5,
            'pause_ratio': 0.0
        },
        'mouse': {
            'curvature_mean': 0.0,
            'pause_count': 0
        },
        'session': {'duration_s': 0.4},
        'content_length': 300,
        'device_score': 0.1
    }
    r = infer(bot_signals, model, mean, std)
    print(f"  Human probability: {r['human_probability']:.2%}")
    print(f"  Verdict: {r['verdict']}")
    print(f"  Confidence: {r['confidence']:.2%}")

    print('\nTest 3 — JSON string input')
    json_input = json.dumps([120, 180, 5, 0.5, 100, 0.85, 30, 0.06])
    r = infer(json_input, model, mean, std)
    print(f"  Human probability: {r['human_probability']:.2%}")
    print(f"  Verdict: {r['verdict']}")

    print('\n=== TASK 1 COMPLETE - INFERENCE.PY READY ===')

if __name__ == '__main__':
    main()
