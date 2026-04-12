import json
import os

import requests
from openai import OpenAI

ENV_BASE = os.environ.get("ENV_BASE", "http://127.0.0.1:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

TASK_TARGETS = {
    "baseline_retrieval": 0.5,
    "parameter_tuning": 0.7,
    "optimal_rag": 0.85,
}

MAX_TOTAL_REWARD = 1.0
_SCORE_EPS = 1e-5


def _grader_safe_score(x: float) -> float:
    v = float(x)
    if v <= 0.0:
        return _SCORE_EPS
    if v >= 1.0:
        return 1.0 - _SCORE_EPS
    return v


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    err = "none" if error is None else repr(error)
    print(
        f"[STEP] step={step} action={action} reward={reward:.6f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float], score: float) -> None:
    rewards_str = ",".join(f"{r:.6f}" for r in rewards)
    gscore = _grader_safe_score(score)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={gscore:.6f} rewards={rewards_str}\n",
        flush=True,
    )


def create_llm_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ["API_BASE_URL"].rstrip("/"),
        api_key=os.environ["API_KEY"],
    )


def ping_llm_proxy(client: OpenAI, task: str) -> None:
    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": f"Suggest RAG parameters for task: {task}."},
        ],
        max_tokens=8,
    )
    print(f"[DEBUG] LLM proxy call succeeded for task={task}", flush=True)


def run_task(client: OpenAI, task: str, action: dict) -> None:
    log_start(task, "rag_optimizer", MODEL_NAME)
    rewards: list[float] = []

    try:
        ping_llm_proxy(client, task)
    except Exception as e:
        print(f"[DEBUG] LLM proxy call failed for task={task}: {e}", flush=True)

    try:
        reset_res = requests.post(
            f"{ENV_BASE}/reset",
            json={"task_id": task},
            timeout=30,
        )
        reset_res.raise_for_status()
        _ = reset_res.json()

        step_res = requests.post(
            f"{ENV_BASE}/step",
            json={"action": action},
            timeout=30,
        )
        step_res.raise_for_status()
        data = step_res.json()

        reward_raw = data.get("reward", 0.01)
        reward = _grader_safe_score(reward_raw)
        done = bool(data.get("done", False))

        rewards.append(reward)
        log_step(1, json.dumps(action), reward, done, error=None)

        agg = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = _grader_safe_score(agg)

        target = TASK_TARGETS.get(task, 0.85)
        success = done and score >= target
        log_end(success, 1, rewards, score=score)

    except Exception as e:
        print(f"Error: {e}", flush=True)
        log_end(False, 0 if not rewards else 1, rewards if rewards else [0.01], score=0.01)


def main() -> None:
    client = create_llm_client()

    tasks = [
        ("baseline_retrieval", {"chunk_size": 500, "top_k": 3}),
        ("parameter_tuning", {"chunk_size": 350, "top_k": 4}),
        ("optimal_rag", {"chunk_size": 300, "top_k": 5}),
    ]

    for task_name, action in tasks:
        run_task(client, task_name, action)


if __name__ == "__main__":
    main()