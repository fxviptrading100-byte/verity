from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import uuid
import datetime
import hashlib
import os

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Simple, reliable scorer
def score_signals(data):
    try:
        kb = data.get('keyboard', {})
        ms = data.get('mouse', {})
        
        # Use whatever variance fields are available
        dwell_var = float(kb.get('dwell_std', kb.get('dwell_std_ms', 30)))
        flight_var = float(kb.get('flight_std', kb.get('flight_std_ms', 50)))
        mouse_var = float(ms.get('speed_std', ms.get('mouse_speed_std', 150)))
        
        variance = (dwell_var + flight_var + mouse_var) / 3
        human_score = max(15, min(95, 45 + variance * 1.1))
        
        return {
            'human_score': human_score,
            'verified': human_score >= 60,
            'breakdown': {
                'keyboard_rhythm': round(dwell_var * 1.2, 1),
                'mouse_naturalness': round(mouse_var * 0.7, 1),
                'session_behavior': 80
            }
        }
    except:
        return {'human_score': 50, 'verified': False, 'breakdown': {}}

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/verify', methods=['POST'])
def verify():
    try:
        data = request.get_json(silent=True) or {}
        
        scores = score_signals(data)
        
        human_score = float(scores['human_score'])
        verdict = "HUMAN VERIFIED" if human_score >= 60 else "BOT DETECTED"
        
        cert_id = f"verity_{uuid.uuid4().hex[:12]}"
        content_hash = hashlib.sha256(str(data.get('content', '')).encode()).hexdigest()
        timestamp = datetime.datetime.utcnow().isoformat() + 'Z'
        
        return jsonify({
            "certificate_id": cert_id,
            "content_hash": content_hash,
            "human_score": round(human_score, 1),
            "verified": scores['verified'],
            "verdict": verdict,
            "timestamp": timestamp,
            "breakdown": scores['breakdown'],
            "raw": data
        })
    except Exception as e:
        print("Verify error:", str(e))
        return jsonify({
            "error": str(e),
            "human_score": 30.0,
            "verified": False,
            "verdict": "BOT DETECTED"
        }), 500

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8001))
    app.run(host='0.0.0.0', port=port)