import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.scorer import score_signals

def verify(signals, label):
    """Test verification with realistic behavioral signals"""
    result = score_signals(signals)
    
    print(f'\n{label}')
    print(f'Human score: {result["human_score"]:.1f}%')
    print(f'Verdict:     {result["verdict"]}')
    return result

# Test 1 — Real human typing (should be high: 88-95%)
human_signals = {
    'keyboard': {
        'dwell_mean': 120, 'dwell_std': 35,
        'flight_mean': 180, 'flight_std': 55,
        'backspace_ratio': 0.08, 'total_keys': 120,
        'edit_count': 10
    },
    'mouse': {
        'speed_mean': 320, 'speed_std': 180,
        'accel_mean': 8000, 'jerk_mean': 45000,
        'curvature_mean': 0.003, 'pause_count': 4
    },
    'session': {'duration_s': 85},
    'content_length': 80,
    'device_score': 0.9
}

# Test 2 — AI bot submission (should be low: 28-38%)
ai_bot_signals = {
    'keyboard': {
        'dwell_mean': 5, 'dwell_std': 0.5,
        'flight_mean': 8, 'flight_std': 0.3,
        'backspace_ratio': 0.0, 'total_keys': 200,
        'edit_count': 0
    },
    'mouse': {
        'speed_mean': 0, 'speed_std': 0,
        'accel_mean': 0, 'jerk_mean': 0,
        'curvature_mean': 0, 'pause_count': 0
    },
    'session': {'duration_s': 0.2},
    'content_length': 200,
    'device_score': 0.1
}

# Test 3 — Suspicious bot (should be low-mid: 32-42%)
suspicious_bot_signals = {
    'keyboard': {
        'dwell_mean': 15, 'dwell_std': 1.5,
        'flight_mean': 20, 'flight_std': 1.2,
        'backspace_ratio': 0.005, 'total_keys': 150,
        'edit_count': 1
    },
    'mouse': {
        'speed_mean': 20, 'speed_std': 3,
        'accel_mean': 500, 'jerk_mean': 200,
        'curvature_mean': 0.0001, 'pause_count': 0
    },
    'session': {'duration_s': 3},
    'content_length': 280,
    'device_score': 0.2
}

# Test 4 — Careful human writer (should be high-mid: 78-88%)
careful_human_signals = {
    'keyboard': {
        'dwell_mean': 220, 'dwell_std': 45,
        'flight_mean': 280, 'flight_std': 70,
        'backspace_ratio': 0.12, 'total_keys': 95,
        'edit_count': 35
    },
    'mouse': {
        'speed_mean': 280, 'speed_std': 150,
        'accel_mean': 6000, 'jerk_mean': 35000,
        'curvature_mean': 0.004, 'pause_count': 8
    },
    'session': {'duration_s': 480},
    'content_length': 450,
    'device_score': 0.95
}

# Test 5 — Pure paste bot (should be very low: 18-30%)
paste_bot_signals = {
    'keyboard': {
        'dwell_mean': 2, 'dwell_std': 0.2,
        'flight_mean': 3, 'flight_std': 0.1,
        'backspace_ratio': 0.0, 'total_keys': 500,
        'edit_count': 0
    },
    'mouse': {
        'speed_mean': 0, 'speed_std': 0,
        'accel_mean': 0, 'jerk_mean': 0,
        'curvature_mean': 0, 'pause_count': 0
    },
    'session': {'duration_s': 0.3},
    'content_length': 500,
    'device_score': 0.05
}

print('=== VERITY VERIFICATION TEST SUITE ===')
print('Testing with realistic behavioral signals...\n')

# Run all tests
results = []
results.append(verify(human_signals, 'Test 1 — Real human typing'))
results.append(verify(ai_bot_signals, 'Test 2 — AI bot submission'))
results.append(verify(suspicious_bot_signals, 'Test 3 — Suspicious bot'))
results.append(verify(careful_human_signals, 'Test 4 — Careful human writer'))
results.append(verify(paste_bot_signals, 'Test 5 — Pure paste bot'))

# Summary
print('\n' + '='*50)
print('TEST SUMMARY')
print('='*50)

for i, result in enumerate(results, 1):
    # Check if this is expected to be a bot test (tests 2, 3, 5)
    is_bot_test = i in [2, 3, 5]
    correctly_detected = (is_bot_test and not result['verified']) or (not is_bot_test and result['verified'])
    status = "✓" if correctly_detected else "✗"
    print(f"Test {i}: {result['human_score']:6.1f}% {result['verdict']} {status}")

# Validation
human_tests = [results[0], results[3]]  # Tests 1 and 4 should be human
bot_tests = [results[1], results[2], results[4]]  # Tests 2, 3, 5 should be bot

human_avg = sum(r['human_score'] for r in human_tests) / len(human_tests)
bot_avg = sum(r['human_score'] for r in bot_tests) / len(bot_tests)

print(f'\nAverage human score: {human_avg:.1f}%')
print(f'Average bot score:  {bot_avg:.1f}%')

# Check if scores are realistic
if human_avg > 75 and bot_avg < 45:
    print('✅ Scoring system is working correctly!')
    print(f'   Human/Bot separation: {human_avg - bot_avg:.1f}% difference')
else:
    print('⚠️  Scoring may need adjustment')
