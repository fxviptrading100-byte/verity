import numpy as np
import torch
import os

# Load PyTorch model once at module level
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'verity_model.pt')
MEAN_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'mean.npy')
STD_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'std.npy')

# Global model instance
_model = None
_mean = None
_std = None
MODEL_LOADED = False

def _load_model():
    """Load PyTorch model and normalization parameters"""
    global _model, _mean, _std, MODEL_LOADED
    
    if MODEL_LOADED:
        return
    
    try:
        # Load model checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        
        # Import and create model
        from model.model import VerityNet
        _model = VerityNet(
            input_dim=checkpoint['input_dim'], 
            hidden_dims=checkpoint['hidden_dims']
        )
        _model.load_state_dict(checkpoint['model_state'])
        _model.eval()
        
        # Load normalization
        _mean = np.load(MEAN_PATH)
        _std = np.load(STD_PATH)
        
        MODEL_LOADED = True
        print("✅ PyTorch model loaded successfully")
        
    except Exception as e:
        print(f"⚠️  Could not load PyTorch model: {e}")
        MODEL_LOADED = False

def score_signals(signals, baseline=None):
    """
    Real behavioral scoring engine using PyTorch model.
    Returns human probability (0-1) and verdict.
    """
    
    # Ensure model is loaded
    if _model is None:
        _load_model()
    
    # Extract 8-dimensional features directly from signals
    features = _extract_8d_features(signals)
    
    # Use PyTorch model if available
    if MODEL_LOADED and _model is not None:
        human_prob = _model_inference(features)
    else:
        # Fallback to statistical scoring
        human_prob = _statistical_score(features)
    
    # Calculate component scores for breakdown
    scores = _calculate_component_scores(signals, features)
    scores['human_score'] = round(human_prob * 100, 2)  # Convert to percentage
    scores['human_probability'] = round(human_prob, 4)
    scores['verified'] = human_prob >= 0.5
    scores['verdict'] = 'HUMAN VERIFIED' if scores['verified'] else 'BOT DETECTED'
    
    return scores

def _extract_8d_features(signals):
    """Extract 8-dimensional feature vector matching OpenEnv state space"""
    kb = signals.get('keyboard', {})
    ms = signals.get('mouse', {})
    sess = signals.get('session', {})
    content_length = signals.get('content_length', 0)

    # Build 8D feature vector exactly matching OpenEnv state space
    features = np.array([
        kb.get('dwell_mean', 0),           # keystroke_gap_ms
        sess.get('duration_s', 0),         # session_duration_s  
        kb.get('edit_count', 0),           # edit_count
        ms.get('curvature_mean', 0) * 10,  # cursor_entropy (scaled)
        content_length,                    # content_length
        signals.get('device_score', 0.5),  # device_score (default neutral)
        kb.get('dwell_std', 0),            # typing_variance
        _estimate_pause_ratio(kb, sess)    # pause_ratio (estimated)
    ], dtype=np.float32)
    
    return features

def _estimate_pause_ratio(kb, sess):
    """Estimate pause ratio from available signals"""
    duration = sess.get('duration_s', 1)
    total_keys = kb.get('total_keys', 1)
    
    if duration <= 0 or total_keys <= 0:
        return 0.1
    
    # Estimate pauses based on typing speed
    avg_time_per_key = duration * 1000 / total_keys  # ms per key
    expected_time_per_key = 200  # human average
    
    if avg_time_per_key > expected_time_per_key * 2:
        return 0.8  # lots of pauses
    elif avg_time_per_key < expected_time_per_key * 0.5:
        return 0.05  # very fast, bot-like
    else:
        return 0.3  # moderate pauses

def _model_inference(features):
    """Run PyTorch model inference with strong variance penalties"""
    try:
        # Normalize features using loaded mean/std
        x_norm = (features - _mean) / (_std + 1e-8)
        x_tensor = torch.FloatTensor(x_norm).unsqueeze(0)
        
        # Get model prediction
        with torch.no_grad():
            base_score = _model(x_tensor).item()
        
        # Convert to percentage
        human_score = base_score * 100
        
        # Apply strong variance penalties for bot detection
        penalty = _calculate_variance_penalty(features)
        
        # Final score with penalties
        final_score = human_score - penalty
        
        # Clamp to reasonable range
        final_score = np.clip(final_score, 10, 98)
        
        return final_score / 100  # Convert back to 0-1 range
        
    except Exception as e:
        print(f"Model inference failed: {e}")
        return 0.5  # fallback

def _calculate_variance_penalty(features):
    """Calculate strong penalties for bot-like low variance"""
    keystroke_gap, session_dur, edit_count, cursor_entropy, content_len, device_score, typing_var, pause_ratio = features
    
    penalty = 0
    
    # Strong penalty for very low typing variance (clear bot indicator)
    if typing_var < 0.5:
        penalty += 50  # Major penalty for robotic consistency
    elif typing_var < 1.0:
        penalty += 35  # Significant penalty for low variance
    elif typing_var < 2.0:
        penalty += 20  # Moderate penalty for somewhat consistent
    
    # Penalty for very fast keystroke gaps (pasting)
    if keystroke_gap < 10:
        penalty += 40  # Major penalty for instant typing
    elif keystroke_gap < 25:
        penalty += 25  # Significant penalty for very fast
    
    # Penalty for zero/near-zero edits (no backspacing)
    if edit_count < 1:
        penalty += 15  # Penalty for perfect typing
    elif edit_count < 3:
        penalty += 8   # Minor penalty for few edits
    
    # Penalty for very low cursor entropy (no mouse movement)
    if cursor_entropy < 0.1:
        penalty += 20  # Major penalty for no mouse movement
    elif cursor_entropy < 0.5:
        penalty += 10  # Moderate penalty for low mouse entropy
    
    # Penalty for very short sessions (instant submission)
    if session_dur < 2:
        penalty += 25  # Major penalty for instant
    elif session_dur < 10:
        penalty += 10  # Minor penalty for very short
    
    # Penalty for very low pause ratio (no thinking time)
    if pause_ratio < 0.05:
        penalty += 15  # Penalty for no pauses
    
    # Bonus for clear human indicators
    if typing_var > 5:
        penalty -= 10  # Bonus for natural variance
    if keystroke_gap > 80:
        penalty -= 5   # Bonus for natural typing speed
    if edit_count > 10:
        penalty -= 5   # Bonus for natural editing
    if cursor_entropy > 2:
        penalty -= 5   # Bonus for natural mouse movement
    
    return max(0, penalty)  # Penalty can't be negative

def _statistical_score(features):
    """Fallback statistical scoring when model unavailable"""
    keystroke_gap, session_dur, edit_count, cursor_entropy, content_len, device_score, typing_var, pause_ratio = features
    
    # Simple heuristic scoring
    score = 0.3  # baseline lower for bots
    
    # Keystroke timing (strong indicator)
    if 50 <= keystroke_gap <= 300:
        score += 0.3
    elif keystroke_gap < 20:
        score -= 0.2
    
    # Typing variance (very strong indicator)
    if typing_var > 5:
        score += 0.25
    elif typing_var < 1:
        score -= 0.25
    
    # Session duration
    if session_dur > 30:
        score += 0.1
    elif session_dur < 5:
        score -= 0.1
    
    # Natural editing
    if edit_count > 5:
        score += 0.1
    
    # Device trust
    if device_score > 0.7:
        score += 0.1
    elif device_score < 0.3:
        score -= 0.1
    
    # Cursor entropy
    if cursor_entropy > 1:
        score += 0.05
    elif cursor_entropy < 0.1:
        score -= 0.05
    
    return np.clip(score, 0, 1)

def _calculate_component_scores(signals, features):
    """Calculate component scores for UI breakdown"""
    kb = signals.get('keyboard', {})
    ms = signals.get('mouse', {})
    sess = signals.get('session', {})
    
    keystroke_gap, session_dur, edit_count, cursor_entropy, content_len, device_score, typing_var, pause_ratio = features
    
    # Keyboard rhythm score
    dwell_mean = kb.get('dwell_mean', 0)
    dwell_std = kb.get('dwell_std', 0)
    flight_mean = kb.get('flight_mean', 0)
    backspace_ratio = kb.get('backspace_ratio', 0)
    
    keyboard_score = 0.5
    if 50 <= dwell_mean <= 250 and dwell_std > 10:
        keyboard_score += 0.3
    if 80 <= flight_mean <= 400:
        keyboard_score += 0.2
    if 0.02 <= backspace_ratio <= 0.15:
        keyboard_score += 0.2
    if dwell_std > 15:
        keyboard_score += 0.3
    
    # Mouse naturalness score
    speed_mean = ms.get('speed_mean', 0)
    speed_std = ms.get('speed_std', 0)
    jerk_mean = ms.get('jerk_mean', 0)
    curvature_mean = ms.get('curvature_mean', 0)
    
    mouse_score = 0.5
    if speed_mean > 0:
        if 100 <= speed_mean <= 800:
            mouse_score += 0.3
        if speed_std > 50:
            mouse_score += 0.2
        if jerk_mean > 1000:
            mouse_score += 0.3
        if curvature_mean > 0.001:
            mouse_score += 0.2
    
    # Session behavior score
    words_per_second = content_len / session_dur if session_dur > 0 else 0
    session_score = 0.5
    if 0.5 <= words_per_second <= 3:
        session_score += 0.3
    if session_dur > 30:
        session_score += 0.2
    
    return {
        'keyboard_rhythm': round(np.clip(keyboard_score, 0, 1), 4),
        'mouse_naturalness': round(np.clip(mouse_score, 0, 1), 4),
        'session_behavior': round(np.clip(session_score, 0, 1), 4),
        'baseline_match': None,  # Not implemented in this version
        'raw': {
            'dwell_mean_ms': round(dwell_mean, 2),
            'dwell_std_ms': round(dwell_std, 2),
            'flight_mean_ms': round(flight_mean, 2),
            'backspace_ratio': round(backspace_ratio, 4),
            'total_keys': int(kb.get('total_keys', 0)),
            'mouse_speed_mean': round(speed_mean, 2),
            'mouse_speed_std': round(speed_std, 2),
            'mouse_jerk_mean': round(jerk_mean, 2),
            'mouse_curvature_mean': round(curvature_mean, 6),
            'session_duration_s': round(session_dur, 2),
            'words_per_second': round(words_per_second, 3)
        }
    }

# Initialize model on import
_load_model()
