from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import numpy as np
import hashlib
import datetime
import uuid
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Load DATABASE_URL — try multiple methods
DATABASE_URL = None

# Method 1: already in environment
DATABASE_URL = os.environ.get('DATABASE_URL')

# Method 2: read .env file manually stripping BOM
if not DATABASE_URL:
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        '.env'
    )
    if os.path.exists(env_path):
        with open(env_path, 'rb') as f:
            content = f.read()
        # Strip BOM if present
        if content.startswith(b'\xef\xbb\xbf'):
            content = content[3:]
        lines = content.decode('utf-8').strip().split('\n')
        for line in lines:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip()
                if key == 'DATABASE_URL':
                    DATABASE_URL = val
                    os.environ['DATABASE_URL'] = val
                    print(f'DATABASE_URL loaded from .env file manually')
                    break

# Method 3: hardcoded fallback for local dev
if not DATABASE_URL:
    DATABASE_URL = "postgresql://neondb_owner:npg_ifVDayco0N5n@ep-winter-lab-a146tpoh-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    print('Using hardcoded DATABASE_URL for local development')

print(f'DATABASE_URL active: {bool(DATABASE_URL)}')

from model.scorer import score_signals

app = Flask(__name__, 
    static_folder=os.path.join(BASE_DIR, 'frontend'),
    static_url_path='')
CORS(app)

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
                    keyboard_rhythm FLOAT,
                    mouse_naturalness FLOAT,
                    session_behavior FLOAT,
                    baseline_match FLOAT,
                    dwell_mean FLOAT,
                    flight_mean FLOAT,
                    backspace_ratio FLOAT,
                    mouse_speed_mean FLOAT,
                    mouse_jerk_mean FLOAT,
                    session_duration FLOAT,
                    total_keys INTEGER,
                    raw_signals JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''))
            conn.commit()
        print('Database initialized')
    else:
        print('Using in-memory storage (no database)')

# In-memory certificates for local development
certificates = {}

def generate_cert_id():
    return 'verity_' + uuid.uuid4().hex[:10]

def hash_content(content):
    return hashlib.sha256(content.encode()).hexdigest()[:16]

def save_certificate(cert):
    if use_db:
        with engine.connect() as conn:
            conn.execute(text('''
                INSERT INTO certificates (
                    id, content_hash, human_score, verified, timestamp,
                    keyboard_rhythm, mouse_naturalness, session_behavior,
                    baseline_match, dwell_mean, flight_mean, backspace_ratio,
                    mouse_speed_mean, mouse_jerk_mean, session_duration,
                    total_keys, raw_signals
                ) VALUES (
                    :id, :content_hash, :human_score, :verified, :timestamp,
                    :keyboard_rhythm, :mouse_naturalness, :session_behavior,
                    :baseline_match, :dwell_mean, :flight_mean, :backspace_ratio,
                    :mouse_speed_mean, :mouse_jerk_mean, :session_duration,
                    :total_keys, :raw_signals
                )
            '''), cert)
            conn.commit()
    else:
        # In-memory storage
        certificates[cert['id']] = cert

@app.route('/')
def index():
    return send_from_directory(
        os.path.join(BASE_DIR, 'frontend'), 'index.html')

@app.route('/verify/<path:cert_id>')
def verify_page(cert_id):
    return send_from_directory(
        os.path.join(BASE_DIR, 'frontend'), 'verify.html')

@app.route('/verity-sdk.js')
def serve_sdk():
    return send_from_directory(
        os.path.join(BASE_DIR, 'frontend'),
        'verity-sdk.js',
        mimetype='application/javascript'
    )

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'product': 'Verity'})

@app.route('/verify', methods=['POST'])
def verify():
    import json
    data = request.json
    scores = score_signals(data)

    content = data.get('content', '')
    content_hash = hash_content(content) if content else 'no_content'
    cert_id = generate_cert_id()
    timestamp = datetime.datetime.utcnow()
    raw = scores.get('raw', {})

    cert_data = {
        'id': cert_id,
        'content_hash': content_hash,
        'human_score': float(scores['human_probability']),
        'verified': bool(scores['verified']),
        'timestamp': timestamp,
        'keyboard_rhythm': float(scores.get('keyboard_rhythm', 0)),
        'mouse_naturalness': float(scores.get('mouse_naturalness', 0)),
        'session_behavior': float(scores.get('session_behavior', 0)),
        'baseline_match': float(scores['baseline_match']) if scores.get('baseline_match') is not None else None,
        'dwell_mean': float(raw.get('dwell_mean_ms', 0)),
        'flight_mean': float(raw.get('flight_mean_ms', 0)),
        'backspace_ratio': float(raw.get('backspace_ratio', 0)),
        'mouse_speed_mean': float(raw.get('mouse_speed_mean', 0)),
        'mouse_jerk_mean': float(raw.get('mouse_jerk_mean', 0)),
        'session_duration': float(raw.get('session_duration_s', 0)),
        'total_keys': int(raw.get('total_keys', 0)),
        'raw_signals': json.dumps(raw)
    }

    save_certificate(cert_data)

    return jsonify({
        'certificate_id': cert_id,
        'content_hash': content_hash,
        'human_score': float(scores['human_probability']),
        'verified': bool(scores['verified']),
        'timestamp': timestamp.isoformat() + 'Z',
        'breakdown': {
            'keyboard_rhythm': float(scores.get('keyboard_rhythm', 0)),
            'mouse_naturalness': float(scores.get('mouse_naturalness', 0)),
            'session_behavior': float(scores.get('session_behavior', 0)),
            'baseline_match': float(scores['baseline_match']) if scores.get('baseline_match') is not None else None
        },
        'raw': raw
    })

@app.route('/certificate/<cert_id>', methods=['GET'])
@app.route('/api/certificate/<cert_id>', methods=['GET'])
def get_certificate(cert_id):
    import json
    if use_db:
        with engine.connect() as conn:
            result = conn.execute(
                text('SELECT * FROM certificates WHERE id = :id'),
                {'id': cert_id}
            ).fetchone()

        if not result:
            return jsonify({'error': 'Certificate not found'}), 404

        raw = {}
        try:
            raw = json.loads(result.raw_signals) if result.raw_signals else {}
        except:
            pass

        return jsonify({
            'certificate_id': result.id,
            'content_hash': result.content_hash,
            'human_score': result.human_score,
            'verified': result.verified,
            'timestamp': result.timestamp.isoformat() + 'Z',
            'breakdown': {
                'keyboard_rhythm': result.keyboard_rhythm,
                'mouse_naturalness': result.mouse_naturalness,
                'session_behavior': result.session_behavior,
                'baseline_match': result.baseline_match
            },
            'raw': raw
        })
    else:
        # In-memory storage
        cert = certificates.get(cert_id)
        if not cert:
            return jsonify({'error': 'Certificate not found'}), 404
        
        raw = {}
        try:
            raw = json.loads(cert.get('raw_signals', '{}')) if cert.get('raw_signals') else {}
        except:
            pass

        return jsonify({
            'certificate_id': cert['id'],
            'content_hash': cert['content_hash'],
            'human_score': cert['human_score'],
            'verified': cert['verified'],
            'timestamp': cert['timestamp'].isoformat() + 'Z',
            'breakdown': {
                'keyboard_rhythm': cert.get('keyboard_rhythm', 0),
                'mouse_naturalness': cert.get('mouse_naturalness', 0),
                'session_behavior': cert.get('session_behavior', 0),
                'baseline_match': cert.get('baseline_match', None)
            },
            'raw': raw
        })

@app.route('/certificates', methods=['GET'])
def list_certificates():
    if use_db:
        with engine.connect() as conn:
            results = conn.execute(
                text('''SELECT id, content_hash, human_score, verified, 
                    timestamp FROM certificates 
                    ORDER BY created_at DESC LIMIT 20''')
            ).fetchall()

        return jsonify({
            'total': len(results),
            'certificates': [{
                'certificate_id': r.id,
                'content_hash': r.content_hash,
                'human_score': r.human_score,
                'verified': r.verified,
                'timestamp': r.timestamp.isoformat() + 'Z'
            } for r in results]
        })
    else:
        # In-memory storage
        cert_list = list(certificates.values())[-20:]
        return jsonify({
            'total': len(cert_list),
            'certificates': [{
                'certificate_id': cert['id'],
                'content_hash': cert['content_hash'],
                'human_score': cert['human_score'],
                'verified': cert['verified'],
                'timestamp': cert['timestamp'].isoformat() + 'Z'
            } for cert in cert_list]
        })

if __name__ == '__main__':
    init_db()
    port = int(os.getenv('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)