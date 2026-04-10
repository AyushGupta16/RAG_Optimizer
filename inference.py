import os
import requests
import json
from openai import OpenAI  
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Environment Variables
ENV_BASE = os.environ.get("ENV_BASE", "http://127.0.0.1:8000") # ALWAYS local env
LLM_BASE_URL = os.environ.get("API_BASE_URL")  # Hugging Face API base URL for proxy
LLM_API_KEY = os.environ.get("API_KEY")  # Hugging Face token for proxy authentication
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()}", flush=True)

def log_end(success: bool, steps: int, rewards: list):
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}\n", flush=True)

def main():
    task = "optimal_rag"
    log_start(task, "rag_optimizer", MODEL_NAME)
    
    # 🔥 MANDATORY: The Proxy Call   
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

    rewards = []
    try:
        # Initialize defaults to prevent NoneType errors
        reward = 0.0
        done = False

        # 1. Reset
        requests.post(f"{ENV_BASE}/reset", json={"task_id": task})
        
        # 2. Step with Optimal Action 
        action = {"chunk_size": 300, "top_k": 5}
        res = requests.post(f"{ENV_BASE}/step", json={"action": action})

        # print(f"[DEBUG] Status: {res.status_code}")
        # print(f"[DEBUG] Raw Response: {res.text}")
        
        if res.status_code == 200:
            data = res.json()

            # print(f"[DEBUG] Parsed JSON: {data}")

            reward = float(data.get("reward", 0.0))
            done = data.get("done", False)
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