import numpy as np

class VerityEnv:
    """
    Verity RL Environment
    A human verification environment where an agent learns
    to distinguish real human behavioral signals from bots.
    
    State space: 8-dimensional behavioral signal vector
    Action space: Binary (0 = bot, 1 = human)
    Reward: +1 correct, -1 wrong, +0.5 high confidence correct
    """

    def __init__(self):
        self.state_dim = 8
        self.action_dim = 2
        self.current_state = None
        self.current_label = None
        self.steps = 0
        self.max_steps = 100
        self.correct = 0
        self.total = 0
        self.reset()

    def reset(self):
        self.steps = 0
        self.correct = 0
        self.total = 0
        self.current_state, self.current_label = self._generate_sample()
        return self.current_state.copy()

    def step(self, action, confidence=1.0):
        correct = int(action) == int(self.current_label)
        
        if correct and confidence >= 0.9:
            reward = 1.5
        elif correct:
            reward = 1.0
        elif not correct and confidence >= 0.9:
            reward = -1.5
        else:
            reward = -1.0

        self.steps += 1
        self.total += 1
        if correct:
            self.correct += 1

        done = self.steps >= self.max_steps
        self.current_state, self.current_label = self._generate_sample()

        info = {
            "correct": correct,
            "label": int(self.current_label),
            "action": int(action),
            "accuracy": self.correct / self.total if self.total > 0 else 0,
            "confidence": confidence
        }

        return self.current_state.copy(), reward, done, info

    def _generate_sample(self):
        is_human = np.random.randint(0, 2)

        if is_human:
            state = np.array([
                np.random.normal(187, 45),      # avg keystroke gap ms
                np.random.normal(272, 60),      # session duration seconds
                np.random.normal(21, 8),        # edit/delete count
                np.random.normal(0.82, 0.1),    # cursor entropy 0-1
                np.random.normal(312, 80),      # content length words
                np.random.normal(0.91, 0.05),   # device attestation score
                np.random.normal(4.2, 1.1),     # typing speed variance
                np.random.normal(0.78, 0.12),   # natural pause ratio
            ])
        else:
            state = np.array([
                np.random.normal(12, 8),        # bots type near instantly
                np.random.normal(3, 2),         # very short sessions
                np.random.normal(1, 1),         # almost no edits
                np.random.normal(0.15, 0.08),   # low cursor entropy
                np.random.normal(280, 60),      # content length similar
                np.random.normal(0.3, 0.15),    # low device score
                np.random.normal(0.1, 0.05),    # no typing variance
                np.random.normal(0.05, 0.03),   # no natural pauses
            ])

        state = np.clip(state, 0, None)
        return state.astype(np.float32), float(is_human)

    def get_state_labels(self):
        return [
            "keystroke_gap_ms",
            "session_duration_s",
            "edit_count",
            "cursor_entropy",
            "content_length",
            "device_score",
            "typing_variance",
            "pause_ratio"
        ]

    def render(self, info=None):
        print(f"\n--- Verity Environment Step {self.steps} ---")
        labels = self.get_state_labels()
        for i, label in enumerate(labels):
            print(f"  {label}: {self.current_state[i]:.4f}")
        if info:
            print(f"  Label: {'Human' if info['label'] == 1 else 'Bot'}")
            print(f"  Action: {'Human' if info['action'] == 1 else 'Bot'}")
            print(f"  Correct: {info['correct']}")
            print(f"  Accuracy: {info['accuracy']:.2%}")