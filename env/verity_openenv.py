from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import os
import sys
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.verity_env import VerityEnv

app = FastAPI(
    title="Verity OpenEnv",
    description="Human verification RL environment — OpenEnv compatible",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend", html=True), name="static")

# Serve frontend at root
@app.get("/")
async def read_root():
    return FileResponse("frontend/index.html")

# Serve verity-sdk.js
@app.get("/verity-sdk.js")
async def serve_sdk():
    return FileResponse("frontend/verity-sdk.js")

# Add /verify endpoint for frontend
@app.post("/verify")
async def verify_endpoint(request):
    """Handle verification requests from frontend SDK"""
    try:
        # Import required modules
        import json
        import sys
        import os
        from datetime import datetime
        import uuid
        import hashlib
        
        # Add model path
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from model.scorer import score_signals
        
        # Parse request body
        data = await request.json()
        
        # Ensure required fields exist with defaults
        if 'keyboard' not in data:
            data['keyboard'] = {}
        if 'mouse' not in data:
            data['mouse'] = {}
        if 'session' not in data:
            data['session'] = {}
        if 'device_score' not in data:
            data['device_score'] = 0.5
        if 'content_length' not in data:
            data['content_length'] = len(data.get('content', ''))
        
        # Call scorer with error handling
        try:
            scores = score_signals(data)
        except Exception as e:
            print(f"Scorer error: {e}")
            scores = {
                'human_score': 0.0,
                'verified': False,
                'keyboard_rhythm': 0.0,
                'mouse_naturalness': 0.0,
                'session_behavior': 0.0,
                'raw': {}
            }
        
        # Validate and ensure human_score is never NaN
        human_score = scores.get('human_score', 0.0)
        if human_score is None or (isinstance(human_score, float) and (human_score != human_score)):  # NaN check
            human_score = 0.0
        
        # Ensure human_score is within bounds
        human_score = max(0.0, min(100.0, float(human_score)))
        
        # Generate certificate and metadata
        content = data.get('content', '')
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16] if content else 'no_content'
        cert_id = 'verity_' + uuid.uuid4().hex[:10]
        timestamp = datetime.datetime.utcnow()
        raw = scores.get('raw', {})
        
        # Return clean response
        response = {
            'certificate_id': cert_id,
            'content_hash': content_hash,
            'human_score': float(human_score),
            'verified': bool(scores.get('verified', False)),
            'verdict': 'HUMAN VERIFIED' if scores.get('verified', False) else 'BOT DETECTED',
            'timestamp': timestamp.isoformat() + 'Z',
            'breakdown': {
                'keyboard_rhythm': float(scores.get('keyboard_rhythm', 0)),
                'mouse_naturalness': float(scores.get('mouse_naturalness', 0)),
                'session_behavior': float(scores.get('session_behavior', 0)),
                'baseline_match': float(scores['baseline_match']) if scores.get('baseline_match') is not None else None
            },
            'raw': raw
        }
        
        return response
        
    except Exception as e:
        print(f"Critical error in /verify: {e}")
        return {
            'error': f'Verification failed: {str(e)}',
            'human_score': 0.0,
            'verdict': 'BOT DETECTED',
            'verified': False,
            'certificate_id': None,
            'timestamp': None,
            'breakdown': {
                'keyboard_rhythm': 0.0,
                'mouse_naturalness': 0.0,
                'session_behavior': 0.0,
                'baseline_match': None
            },
            'raw': {}
        }

env = VerityEnv()

class Action(BaseModel):
    action: int
    confidence: Optional[float] = 1.0

class StepResponse(BaseModel):
    observation: List[float]
    reward: float
    done: bool
    info: dict

class ResetResponse(BaseModel):
    observation: List[float]

class StateResponse(BaseModel):
    steps: int
    total: int
    correct: int
    accuracy: float
    max_steps: int
    state_labels: List[str]
    current_state: List[float]

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "environment": "VerityEnv",
        "description": "Human verification RL environment",
        "state_dim": env.state_dim,
        "action_dim": env.action_dim,
        "openenv_compatible": True,
        "version": "1.0.0",
        "actions": {
            "0": "Bot — reject submission",
            "1": "Human — verify and issue certificate"
        },
        "reward_logic": {
            "+1.5": "correct + high confidence",
            "+1.0": "correct",
            "-1.0": "wrong",
            "-1.5": "wrong + high confidence"
        }
    }

@app.post("/reset", response_model=ResetResponse)
async def reset():
    obs = env.reset()
    return ResetResponse(observation=obs.tolist())

@app.post("/step", response_model=StepResponse)
async def step(action: Action):
    if action.action not in [0, 1]:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="action must be 0 (bot) or 1 (human)"
        )
    obs, reward, done, info = env.step(
        action.action,
        action.confidence
    )
    return StepResponse(
        observation=obs.tolist(),
        reward=float(reward),
        done=done,
        info={
            "correct": bool(info["correct"]),
            "label": int(info["label"]),
            "action": int(info["action"]),
            "accuracy": float(info["accuracy"]),
            "confidence": float(info["confidence"]),
            "label_name": "human" if info["label"] == 1 else "bot",
            "action_name": "human" if info["action"] == 1 else "bot"
        }
    )

@app.get("/state", response_model=StateResponse)
async def state():
    return StateResponse(
        steps=env.steps,
        total=env.total,
        correct=env.correct,
        accuracy=float(env.correct / env.total) if env.total > 0 else 0.0,
        max_steps=env.max_steps,
        state_labels=env.get_state_labels(),
        current_state=env.current_state.tolist()
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    print(f"Starting Verity OpenEnv server on port {port}")
    print(f"Health check: http://localhost:{port}/health")
    print(f"API docs: http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)
