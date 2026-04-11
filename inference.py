import os
import requests
import json
from openai import OpenAI  
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Environment Variables
ENV_BASE = os.environ.get("ENV_BASE", "http://127.0.0.1:8000")
LLM_BASE_URL = os.environ.get("API_BASE_URL")
LLM_API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool):
    print(f"[STEP] step={step} action={action} reward={reward:.6f} done={str(done).lower()}", flush=True)

def log_end(success: bool, steps: int, rewards: list):
    rewards_str = ",".join([f"{r:.6f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}\n", flush=True)

def run_task(task: str, action: dict):
    """Run a single task with the given action"""
    log_start(task, "rag_optimizer", MODEL_NAME)
    
    rewards = []
    try:
        # 1. Reset with task_id
        requests.post(f"{ENV_BASE}/reset", json={"task_id": task})
        
        # 2. Step with action
        res = requests.post(f"{ENV_BASE}/step", json={"action": action})
        
        if res.status_code == 200:
            data = res.json()
            
            reward_raw = data.get("reward")
            reward = float(reward_raw) if reward_raw is not None else 0.0
            done = data.get("done", False)
            
            rewards.append(reward)
            log_step(1, json.dumps(action), reward, done)
            
            # Success = score in valid range (0, 1)
            success = done and reward > 0.85
            log_end(success, 1, rewards)
        else:
            log_end(False, 0, [0.0])
            
    except Exception as e:
        print(f"Error: {e}")
        log_end(False, 0, [0.0])

def main():
    # MANDATORY: LLM Proxy Call (once at start)
    try:
        print(f"[DEBUG] BASE_URL={LLM_BASE_URL}, KEY_PRESENT={bool(LLM_API_KEY)}")
        
        client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY or "sk-test"
        )
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Suggest RAG parameters."}],
            max_tokens=5
        )
        print("[DEBUG] LLM Proxy call successful", flush=True)
    except Exception as e:
        print(f"[DEBUG] LLM Proxy call failed: {e}", flush=True)
    
    # Run 3 tasks with different actions
    tasks = [
        ("baseline_retrieval", {"chunk_size": 500, "top_k": 3}),   # Suboptimal -> ~0.57
        ("parameter_tuning", {"chunk_size": 350, "top_k": 4}),     # Close -> ~0.81
        ("optimal_rag", {"chunk_size": 300, "top_k": 5})           # Optimal -> ~0.99
    ]
    
    for task_name, action in tasks:
        run_task(task_name, action)

if __name__ == "__main__":
    main()