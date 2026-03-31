from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import torch
import numpy as np
import hashlib
import datetime
import uuid
import os
import sys

load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from model.model import VerityNet

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

DATABASE_URL = os.getenv('DATABASE_URL')

# Fallback to in-memory storage for local development
if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
    use_db = True
    print('Using PostgreSQL database')
else:
    use_db = False
    print('DATABASE_URL not found, using in-memory storage')

def init_db():
    if use_db:
        with engine.connect() as conn:
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS certificates (
                    id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    human_score FLOAT NOT NULL,
                    verified BOOLEAN NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    keystroke_gap_ms FLOAT,
                    session_duration_s FLOAT,
                    edit_count INTEGER,
                    cursor_entropy FLOAT,
                    device_score FLOAT,
                    content_length INTEGER,
                    typing_variance FLOAT,
                    pause_ratio FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''))
            conn.commit()
        print('Database initialized')
    else:
        print('Using in-memory storage (no database)')

# In-memory certificates for local development
certificates = {}

checkpoint = torch.load(
    os.path.join(BASE_DIR, 'model', 'verity_model.pt'),
    map_location='cpu'
)
model = VerityNet(
    input_dim=checkpoint['input_dim'],
    hidden_dims=checkpoint['hidden_dims']
)
model.load_state_dict(checkpoint['model_state'])
model.eval()

mean = np.load(os.path.join(BASE_DIR, 'model', 'mean.npy'))
std = np.load(os.path.join(BASE_DIR, 'model', 'std.npy'))

def normalize(x, mean, std):
    return (x - mean) / (std + 1e-8)

def generate_cert_id():
    return 'verity_' + uuid.uuid4().hex[:10]

def hash_content(content):
    return hashlib.sha256(content.encode()).hexdigest()[:16]

def save_certificate(cert, signals):
    if use_db:
        with engine.connect() as conn:
            conn.execute(text('''
                INSERT INTO certificates (
                    id, content_hash, human_score, verified, timestamp,
                    keystroke_gap_ms, session_duration_s, edit_count,
                    cursor_entropy, device_score, content_length,
                    typing_variance, pause_ratio
                ) VALUES (
                    :id, :content_hash, :human_score, :verified, :timestamp,
                    :keystroke_gap_ms, :session_duration_s, :edit_count,
                    :cursor_entropy, :device_score, :content_length,
                    :typing_variance, :pause_ratio
                )
            '''), {
                'id': cert['certificate_id'],
                'content_hash': cert['content_hash'],
                'human_score': cert['human_score'],
                'verified': cert['verified'],
                'timestamp': datetime.datetime.utcnow(),
                'keystroke_gap_ms': signals[0],
                'session_duration_s': signals[1],
                'edit_count': int(signals[2]),
                'cursor_entropy': signals[3],
                'device_score': signals[5],
                'content_length': int(signals[4]),
                'typing_variance': signals[6],
                'pause_ratio': signals[7]
            })
            conn.commit()
    else:
        # In-memory storage
        certificates[cert['certificate_id']] = {
            **cert,
            'signals_raw': {
                'keystroke_gap_ms': float(signals[0]),
                'session_duration_s': float(signals[1]),
                'edit_count': int(signals[2]),
                'cursor_entropy': float(signals[3]),
                'device_score': float(signals[5]),
                'content_length': int(signals[4]),
                'typing_variance': float(signals[6]),
                'pause_ratio': float(signals[7])
            }
        }

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/verify/<path:cert_id>')
def verify_page(cert_id):
    return send_from_directory('../frontend', 'verify.html')

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

    save_certificate(certificate, signals)
    return jsonify(certificate)

@app.route('/certificate/<cert_id>', methods=['GET'])
def get_certificate(cert_id):
    if use_db:
        with engine.connect() as conn:
            result = conn.execute(
                text('SELECT * FROM certificates WHERE id = :id'),
                {'id': cert_id}
            ).fetchone()

        if not result:
            return jsonify({'error': 'Certificate not found'}), 404

        return jsonify({
            'certificate_id': result.id,
            'content_hash': result.content_hash,
            'human_score': result.human_score,
            'verified': result.verified,
            'timestamp': result.timestamp.isoformat() + 'Z',
            'signals': {
                'keystroke_gap_ms': result.keystroke_gap_ms,
                'session_duration_s': result.session_duration_s,
                'edit_count': result.edit_count,
                'cursor_entropy': result.cursor_entropy,
                'device_score': result.device_score,
            }
        })
    else:
        # In-memory storage
        cert = certificates.get(cert_id)
        if not cert:
            return jsonify({'error': 'Certificate not found'}), 404
        
        return jsonify({
            'certificate_id': cert['certificate_id'],
            'content_hash': cert['content_hash'],
            'human_score': cert['human_score'],
            'verified': cert['verified'],
            'timestamp': cert['timestamp'],
            'signals': {
                'keystroke_gap_ms': cert['signals_raw']['keystroke_gap_ms'],
                'session_duration_s': cert['signals_raw']['session_duration_s'],
                'edit_count': cert['signals_raw']['edit_count'],
                'cursor_entropy': cert['signals_raw']['cursor_entropy'],
                'device_score': cert['signals_raw']['device_score'],
            }
        })

@app.route('/certificates', methods=['GET'])
def list_certificates():
    if use_db:
        with engine.connect() as conn:
            results = conn.execute(
                text('SELECT * FROM certificates ORDER BY created_at DESC LIMIT 20')
            ).fetchall()

        certs = [{
            'certificate_id': r.id,
            'content_hash': r.content_hash,
            'human_score': r.human_score,
            'verified': r.verified,
            'timestamp': r.timestamp.isoformat() + 'Z'
        } for r in results]

        return jsonify({'total': len(certs), 'certificates': certs})
    else:
        # In-memory storage
        cert_list = list(certificates.values())[-20:]
        certs = [{
            'certificate_id': cert['certificate_id'],
            'content_hash': cert['content_hash'],
            'human_score': cert['human_score'],
            'verified': cert['verified'],
            'timestamp': cert['timestamp']
        } for cert in cert_list]

        return jsonify({'total': len(certs), 'certificates': certs})

if __name__ == '__main__':
    init_db()
    port = int(os.getenv('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)