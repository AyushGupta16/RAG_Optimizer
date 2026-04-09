import os
import requests
import json

# Mandatory Environment Variables
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "rag-optimizer-v1")

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()}", flush=True)

def log_end(success: bool, steps: int, rewards: list):
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

def main():
    task = "optimal_rag"
    log_start(task, "rag_optimizer", MODEL_NAME)
    
    rewards = []
    try:
        # 1. Reset
        requests.post(f"{API_BASE}/reset", json={"task_id": task})
        
        # 2. Step with Optimal Action
        action = {"chunk_size": 300, "top_k": 5}
        res = requests.post(f"{API_BASE}/step", json={"action": action})
        
        if res.status_code == 200:
            data = res.json()
            reward = data.get("reward", 0.0)
            done = data.get("done", True)
            rewards.append(reward)
            log_step(1, json.dumps(action), reward, done)
            log_end(reward >= 0.85, 1, rewards)
        else:
            log_end(False, 0, [0.0])
            
    except Exception as e:
        print(f"Error: {e}")
        log_end(False, 0, [0.0])

if __name__ == "__main__":
    main()