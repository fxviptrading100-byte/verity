import requests
import sys

BASE = "http://localhost:8001"

def test_health():
    print("Testing /health...")
    r = requests.get(BASE + "/health", timeout=5)
    data = r.json()
    assert r.status_code == 200
    assert data["status"] == "ok"
    assert data["openenv_compatible"] == True
    print("  PASSED - state_dim=" + str(data["state_dim"]))

def test_reset():
    print("Testing /reset...")
    r = requests.post(BASE + "/reset", timeout=5)
    data = r.json()
    assert r.status_code == 200
    assert len(data["observation"]) == 8
    print("  PASSED - observation length 8")
    return data["observation"]

def test_step(obs):
    print("Testing /step...")
    action = 1 if obs[0] > 50 else 0
    r = requests.post(BASE + "/step",
        json={"action": action, "confidence": 0.9},
        timeout=5)
    data = r.json()
    assert r.status_code == 200
    assert "reward" in data
    assert "observation" in data
    assert "done" in data
    assert "info" in data
    print("  PASSED - reward=" + str(data["reward"]))

def test_state():
    print("Testing /state...")
    r = requests.get(BASE + "/state", timeout=5)
    data = r.json()
    assert r.status_code == 200
    assert "accuracy" in data
    assert "steps" in data
    assert "state_labels" in data
    print("  PASSED - accuracy=" + str(round(data["accuracy"], 4)))

def test_episode():
    print("Testing full episode...")
    requests.post(BASE + "/reset")
    total_reward = 0
    correct = 0
    for i in range(100):
        state = requests.get(BASE + "/state").json()
        obs = state["current_state"]
        action = 1 if obs[0] > 50 else 0
        r = requests.post(BASE + "/step",
            json={"action": action, "confidence": 0.85}).json()
        total_reward += r["reward"]
        if r["info"]["correct"]:
            correct += 1
        if r["done"]:
            break
    print("  PASSED - reward=" + str(round(total_reward, 1)) + " accuracy=" + str(round(correct/100, 2)))

def main():
    print("Verity OpenEnv - Full Test Suite")
    print("=" * 40)
    try:
        test_health()
        obs = test_reset()
        test_step(obs)
        test_state()
        test_episode()
        print("")
        print("ALL TESTS PASSED")
        print("Verity OpenEnv is fully compliant")
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to " + BASE)
        print("Run: python env/verity_openenv.py")
        sys.exit(1)
    except AssertionError as e:
        print("ASSERTION FAILED: " + str(e))
        sys.exit(1)
    except Exception as e:
        print("ERROR: " + str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()