import numpy as np

def score_signals(signals, baseline=None):
    """
    Real behavioral scoring engine.
    Uses weighted statistical distance from human baseline.
    Inspired by BioCatch and Plurilock methodologies.
    Returns scores per dimension and overall human probability.
    """
    
    kb = signals.get('keyboard', {})
    ms = signals.get('mouse', {})
    sess = signals.get('session', {})

    scores = {}

    # 1. KEYBOARD RHYTHM SCORE
    # Humans have natural variance — not too fast, not too robotic
    dwell_mean = kb.get('dwell_mean', 0)
    dwell_std = kb.get('dwell_std', 0)
    flight_mean = kb.get('flight_mean', 0)
    flight_std = kb.get('flight_std', 0)
    backspace_ratio = kb.get('backspace_ratio', 0)
    total_keys = kb.get('total_keys', 0)

    # Dwell time: humans average 80-180ms
    dwell_score = 0
    if dwell_mean > 0:
        if 50 <= dwell_mean <= 250:
            dwell_score = 1.0 - abs(dwell_mean - 120) / 250
        elif dwell_mean < 20:
            dwell_score = 0.05  # robotic
        else:
            dwell_score = 0.3

    # Flight time: humans average 100-300ms between keys
    flight_score = 0
    if flight_mean > 0:
        if 80 <= flight_mean <= 400:
            flight_score = 1.0 - abs(flight_mean - 180) / 400
        elif flight_mean < 15:
            flight_score = 0.05  # bot pasting
        else:
            flight_score = 0.3

    # Variance score: humans are inconsistent (high std = natural)
    # Bots have near-zero variance
    variance_score = 0
    if dwell_std > 0:
        # Good human std is 20-80ms
        if dwell_std >= 15:
            variance_score = min(dwell_std / 60, 1.0)
        else:
            variance_score = dwell_std / 15 * 0.3

    # Backspace ratio: humans make mistakes (0.03-0.15 is natural)
    backspace_score = 0
    if 0.02 <= backspace_ratio <= 0.20:
        backspace_score = 1.0
    elif backspace_ratio == 0 and total_keys > 20:
        backspace_score = 0.1  # suspiciously perfect
    elif backspace_ratio > 0.3:
        backspace_score = 0.5  # possibly human but erratic
    else:
        backspace_score = backspace_ratio / 0.02

    keyboard_score = (
        dwell_score * 0.30 +
        flight_score * 0.25 +
        variance_score * 0.30 +
        backspace_score * 0.15
    )
    scores['keyboard_rhythm'] = round(min(max(keyboard_score, 0), 1), 4)

    # 2. MOUSE NATURALNESS SCORE
    speed_mean = ms.get('speed_mean', 0)
    speed_std = ms.get('speed_std', 0)
    accel_mean = ms.get('accel_mean', 0)
    jerk_mean = ms.get('jerk_mean', 0)
    curvature_mean = ms.get('curvature_mean', 0)
    pause_count = ms.get('pause_count', 0)

    # Speed: humans move 100-800 px/s naturally
    speed_score = 0
    if speed_mean > 0:
        if 50 <= speed_mean <= 1000:
            speed_score = 1.0 - abs(speed_mean - 300) / 1000
            speed_score = max(speed_score, 0.3)
        elif speed_mean < 10:
            speed_score = 0.1
        else:
            speed_score = 0.4

    # Speed variance: humans are inconsistent
    speed_var_score = min(speed_std / 200, 1.0) if speed_std > 0 else 0.1

    # Jerk: humans have natural jerk (sudden direction changes)
    jerk_score = 0
    if jerk_mean > 0:
        if jerk_mean > 1000:
            jerk_score = 1.0
        elif jerk_mean > 100:
            jerk_score = 0.7
        else:
            jerk_score = 0.2  # too smooth = bot

    # Curvature: humans curve their paths
    curve_score = min(curvature_mean * 100, 1.0) if curvature_mean > 0 else 0.2

    # Natural pauses: humans pause while thinking
    pause_score = min(pause_count / 3, 1.0)

    if speed_mean == 0 and accel_mean == 0:
        # No mouse data at all
        mouse_score = 0.4  # neutral — can't tell
    else:
        mouse_score = (
            speed_score * 0.25 +
            speed_var_score * 0.20 +
            jerk_score * 0.25 +
            curve_score * 0.15 +
            pause_score * 0.15
        )

    scores['mouse_naturalness'] = round(min(max(mouse_score, 0), 1), 4)

    # 3. SESSION BEHAVIOR SCORE
    duration = sess.get('duration_s', 0)
    content_length = signals.get('content_length', 0)

    # Time vs content ratio: humans take time to write
    session_score = 0
    if duration > 0 and content_length > 0:
        words_per_second = content_length / duration
        # Natural: 0.5-3 words per second
        if 0.3 <= words_per_second <= 4:
            session_score = 1.0 - abs(words_per_second - 1.5) / 4
            session_score = max(session_score, 0.4)
        elif words_per_second > 10:
            session_score = 0.05  # impossible — bot pasting
        else:
            session_score = 0.5
    elif duration > 5:
        session_score = 0.6
    else:
        session_score = 0.2

    scores['session_behavior'] = round(min(max(session_score, 0), 1), 4)

    # 4. BASELINE COMPARISON SCORE (only if enrolled)
    baseline = signals.get('baseline')
    baseline_score = None

    if baseline and isinstance(baseline, dict):
        def zscore_dist(val, mean, std):
            if std <= 0:
                return 0
            return abs(val - mean) / std

        distances = []

        if baseline.get('dwell') and dwell_mean > 0:
            d = zscore_dist(
                dwell_mean,
                baseline['dwell']['mean'],
                baseline['dwell']['std']
            )
            distances.append(min(d, 5))

        if baseline.get('flight') and flight_mean > 0:
            d = zscore_dist(
                flight_mean,
                baseline['flight']['mean'],
                baseline['flight']['std']
            )
            distances.append(min(d, 5))

        if baseline.get('mouseSpeed') and speed_mean > 0:
            d = zscore_dist(
                speed_mean,
                baseline['mouseSpeed']['mean'],
                baseline['mouseSpeed']['std']
            )
            distances.append(min(d, 5))

        if distances:
            avg_distance = np.mean(distances)
            # Distance 0 = perfect match, 3+ = different person/bot
            baseline_score = max(0, 1 - (avg_distance / 3))
            scores['baseline_match'] = round(baseline_score, 4)

    # FINAL WEIGHTED SCORE
    if baseline_score is not None:
        final = (
            scores['keyboard_rhythm'] * 0.30 +
            scores['mouse_naturalness'] * 0.25 +
            scores['session_behavior'] * 0.20 +
            baseline_score * 0.25
        )
    else:
        final = (
            scores['keyboard_rhythm'] * 0.40 +
            scores['mouse_naturalness'] * 0.35 +
            scores['session_behavior'] * 0.25
        )

    scores['human_probability'] = round(min(max(final, 0), 1), 4)
    scores['verified'] = scores['human_probability'] >= 0.5

    # RAW SIGNALS for debug panel
    scores['raw'] = {
        'dwell_mean_ms': round(dwell_mean, 2),
        'dwell_std_ms': round(dwell_std, 2),
        'flight_mean_ms': round(flight_mean, 2),
        'flight_std_ms': round(float(kb.get('flight_std', 0)), 2),
        'backspace_ratio': round(backspace_ratio, 4),
        'total_keys': int(total_keys),
        'mouse_speed_mean': round(speed_mean, 2),
        'mouse_speed_std': round(speed_std, 2),
        'mouse_jerk_mean': round(jerk_mean, 2),
        'mouse_curvature_mean': round(curvature_mean, 6),
        'mouse_pauses': int(pause_count),
        'session_duration_s': round(duration, 2),
        'words_per_second': round(
            content_length / duration if duration > 0 else 0, 3
        )
    }

    return scores

if __name__ == '__main__':
    # Test with fake human signals
    human = {
        'keyboard': {
            'dwell_mean': 115, 'dwell_std': 35,
            'flight_mean': 180, 'flight_std': 55,
            'backspace_ratio': 0.08, 'total_keys': 120,
            'edit_count': 10, 'flight_std': 55
        },
        'mouse': {
            'speed_mean': 320, 'speed_std': 180,
            'accel_mean': 8000, 'jerk_mean': 45000,
            'curvature_mean': 0.003, 'pause_count': 4,
            'pause_mean': 1200
        },
        'session': {'duration_s': 85},
        'content_length': 80
    }
    bot = {
        'keyboard': {
            'dwell_mean': 8, 'dwell_std': 1,
            'flight_mean': 10, 'flight_std': 0.5,
            'backspace_ratio': 0.0, 'total_keys': 200,
            'edit_count': 0, 'flight_std': 0.5
        },
        'mouse': {
            'speed_mean': 0, 'speed_std': 0,
            'accel_mean': 0, 'jerk_mean': 0,
            'curvature_mean': 0, 'pause_count': 0,
            'pause_mean': 0
        },
        'session': {'duration_s': 0.3},
        'content_length': 200
    }

    print('HUMAN TEST:')
    r = score_signals(human)
    print(f"  Keyboard Rhythm:    {r['keyboard_rhythm']:.2%}")
    print(f"  Mouse Naturalness:  {r['mouse_naturalness']:.2%}")
    print(f"  Session Behavior:   {r['session_behavior']:.2%}")
    print(f"  Human Probability:  {r['human_probability']:.2%}")
    print(f"  Verdict:            {'HUMAN' if r['verified'] else 'BOT'}")

    print('\nBOT TEST:')
    r = score_signals(bot)
    print(f"  Keyboard Rhythm:    {r['keyboard_rhythm']:.2%}")
    print(f"  Mouse Naturalness:  {r['mouse_naturalness']:.2%}")
    print(f"  Session Behavior:   {r['session_behavior']:.2%}")
    print(f"  Human Probability:  {r['human_probability']:.2%}")
    print(f"  Verdict:            {'HUMAN' if r['verified'] else 'BOT'}")
