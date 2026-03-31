from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import numpy as np
import hashlib
import datetime
import uuid
import os
import sys

sys.path.insert(0, 'C:\\Users\\Admin\\Desktop\\verity')

from model.model import VerityNet

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

checkpoint = torch.load(
    'C:\\Users\\Admin\\Desktop\\verity\\model\\verity_model.pt',
    map_location='cpu'
)
model = VerityNet(
    input_dim=checkpoint['input_dim'],
    hidden_dims=checkpoint['hidden_dims']
)
model.load_state_dict(checkpoint['model_state'])
model.eval()

mean = np.load('C:\\Users\\Admin\\Desktop\\verity\\model\\mean.npy')
std = np.load('C:\\Users\\Admin\\Desktop\\verity\\model\\std.npy')

certificates = {}

def normalize(x, mean, std):
    return (x - mean) / (std + 1e-8)

def generate_cert_id():
    return 'verity_' + uuid.uuid4().hex[:10]

def hash_content(content):
    return hashlib.sha256(content.encode()).hexdigest()[:16]

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'product': 'Verity'})

@app.route('/verify', methods=['POST'])
def verify():
    data = request.json

    signals = np.array([
        float(data.get('keystroke_gap_ms', 150)),
        float(data.get('session_duration_s', 60)),
        float(data.get('edit_count', 5)),
        float(data.get('cursor_entropy', 0.5)),
        float(data.get('content_length', 100)),
        float(data.get('device_score', 0.8)),
        float(data.get('typing_variance', 2.0)),
        float(data.get('pause_ratio', 0.5)),
    ], dtype=np.float32)

    signals_norm = normalize(signals, mean, std)
    x = torch.FloatTensor(signals_norm).unsqueeze(0)

    with torch.no_grad():
        prob = model(x).item()

    verified = prob >= 0.5
    content = data.get('content', '')
    content_hash = hash_content(content) if content else 'no_content'
    cert_id = generate_cert_id()
    timestamp = datetime.datetime.utcnow().isoformat() + 'Z'

    certificate = {
        'certificate_id': cert_id,
        'content_hash': content_hash,
        'human_score': round(prob, 4),
        'verified': verified,
        'timestamp': timestamp,
        'signals': {
            'keystroke_gap_ms': round(float(signals[0]), 2),
            'session_duration_s': round(float(signals[1]), 2),
            'edit_count': int(signals[2]),
            'cursor_entropy': round(float(signals[3]), 4),
            'device_score': round(float(signals[5]), 4),
        }
    }

    certificates[cert_id] = certificate
    return jsonify(certificate)

@app.route('/certificate/<cert_id>', methods=['GET'])
def get_certificate(cert_id):
    cert = certificates.get(cert_id)
    if not cert:
        return jsonify({'error': 'Certificate not found'}), 404
    return jsonify(cert)

if __name__ == '__main__':
    app.run(debug=True, port=5000)