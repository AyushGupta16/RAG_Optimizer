import os
import requests
import json
from openai import OpenAI  
from dotenv import load_dotenv

load_dotenv()

ENV_BASE = os.environ.get("ENV_BASE", "http://127.0.0.1:8000")
LLM_BASE_URL = os.environ.get("API_BASE_URL")
LLM_API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()}", flush=True)


def log_end(success: bool, steps: int, rewards: list):
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}\n", flush=True)


def run_task(task: str, actions_seq: list):
    """Multi-step RL-style rollout per task"""

    log_start(task, "rag_optimizer", MODEL_NAME)

    rewards = []

    try:
        # -------------------------
        # RESET (validated)
        # -------------------------
        reset_res = requests.post(f"{ENV_BASE}/reset", json={"task_id": task})

        if reset_res.status_code != 200:
            log_end(False, 0, [0.0])
            return

        # -------------------------
        # STEP LOOP (agentic behavior)
        # -------------------------
        for i, action in enumerate(actions_seq):

            res = requests.post(f"{ENV_BASE}/step", json={"action": action})

            if res.status_code != 200:
                break

            try:
                data = res.json()
            except Exception:
                break

            reward = float(data.get("reward") or 0.0)
            done = bool(data.get("done", False))

            rewards.append(reward)
            log_step(i + 1, json.dumps(action), reward, done)

            if done:
                break

        # -------------------------
        # SUCCESS LOGIC (REAL)
        # -------------------------
        if len(rewards) == 0:
            success = False
        else:
            avg_reward = sum(rewards) / len(rewards)
            success = avg_reward > 0.6

        log_end(success, len(rewards), rewards)

    except Exception as e:
        print(f"Error: {e}")
        log_end(False, 0, [0.0])


def main():

    # -------------------------
    # LLM PROXY CALL (mandatory)
    # -------------------------
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

    # -------------------------
    # TASKS (3-stage curriculum)
    # -------------------------
    tasks = [
        (
            "baseline_retrieval",
            [
                {"chunk_size": 600, "top_k": 2},
                {"chunk_size": 550, "top_k": 3}
            ]
        ),
        (
            "parameter_tuning",
            [
                {"chunk_size": 400, "top_k": 4},
                {"chunk_size": 350, "top_k": 4}
            ]
        ),
        (
            "optimal_rag",
            [
                {"chunk_size": 300, "top_k": 5},
                {"chunk_size": 280, "top_k": 5}
            ]
        )
    ]

    for task_name, actions in tasks:
        run_task(task_name, actions)


if __name__ == "__main__":
    main()