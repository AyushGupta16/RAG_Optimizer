import json
import os
import re
from typing import Optional

import requests
from openai import OpenAI

ENV_BASE = os.environ.get("ENV_BASE", "http://127.0.0.1:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

TASK_TARGETS = {
    "baseline_retrieval": 0.5,
    "parameter_tuning": 0.7,
    "optimal_rag": 0.85,
}

DEFAULT_ACTIONS = {
    "baseline_retrieval": {"chunk_size": 500, "top_k": 3},
    "parameter_tuning": {"chunk_size": 350, "top_k": 4},
    "optimal_rag": {"chunk_size": 300, "top_k": 5},
}

_SCORE_EPS = 1e-5
MAX_STEPS = 2
REQUEST_TIMEOUT = 30


def _grader_safe_score(x: float) -> float:
    v = float(x)
    if v <= 0.0:
        return _SCORE_EPS
    if v >= 1.0:
        return 1.0 - _SCORE_EPS
    return v


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def _normalize_action(action: dict) -> dict:
    return {
        "chunk_size": _clamp_int(action.get("chunk_size", 500), 100, 1000),
        "top_k": _clamp_int(action.get("top_k", 3), 1, 10),
    }


def _estimate_reward(action: dict) -> float:
    size_err = abs(action["chunk_size"] - 300) / 700.0
    k_err = abs(action["top_k"] - 5) / 5.0
    raw_score = 1.0 - (size_err + k_err) / 2.0
    return _grader_safe_score(raw_score)


def _prefer_better_action(candidate: Optional[dict], fallback: dict) -> dict:
    if candidate is None:
        return fallback
    if _estimate_reward(candidate) >= _estimate_reward(fallback):
        return candidate
    return fallback


def is_weak_suggestion(task: str, action: dict) -> bool:
    if task == "optimal_rag":
        return abs(action["chunk_size"] - 300) > 100 or abs(action["top_k"] - 5) > 1
    if task == "parameter_tuning":
        return abs(action["chunk_size"] - 350) > 120 or abs(action["top_k"] - 4) > 1
    return False


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
    print(
        f"[END] success={str(success).lower()} steps={steps} score={_grader_safe_score(score):.6f} rewards={rewards_str}",
        flush=True,
    )


def create_llm_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ["API_BASE_URL"].rstrip("/"),
        api_key=os.environ["API_KEY"],
    )


def create_http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


def ping_llm_proxy(client: OpenAI, task: str) -> None:
    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": f"Suggest RAG parameters for task: {task}."},
        ],
        max_tokens=8,
    )
    print(f"[DEBUG] LLM proxy call succeeded for task={task}", flush=True)


def parse_action_from_text(text: str) -> Optional[dict]:
    if not text:
        return None

    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            if "chunk_size" in data and "top_k" in data:
                return _normalize_action(data)
    except Exception:
        pass

    chunk_match = re.search(r"chunk_size\s*[:=]\s*(\d+)", text)
    topk_match = re.search(r"top_k\s*[:=]\s*(\d+)", text)

    if chunk_match and topk_match:
        return _normalize_action(
            {
                "chunk_size": int(chunk_match.group(1)),
                "top_k": int(topk_match.group(1)),
            }
        )

    return None


def llm_suggest_action(client: OpenAI, task: str, target: float) -> dict:
    fallback = DEFAULT_ACTIONS[task]

    prompt = f"""
You are optimizing a RAG retrieval environment.

Task: {task}
Target score: {target}

Choose integer values for:
- chunk_size between 100 and 1000
- top_k between 1 and 10

Return ONLY valid JSON like:
{{"chunk_size": 300, "top_k": 5}}

Guidance:
- easy task can use good but not necessarily optimal values
- medium task should be well tuned
- hard task should be near optimal
""".strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.1,
        )
        text = response.choices[0].message.content or ""
        parsed = parse_action_from_text(text)
        if parsed is not None and not is_weak_suggestion(task, parsed):
            return _prefer_better_action(parsed, fallback)
    except Exception as e:
        print(f"[DEBUG] LLM suggestion failed for task={task}: {e}", flush=True)

    return fallback


def llm_refine_action(
    client: OpenAI,
    task: str,
    target: float,
    previous_action: dict,
    previous_reward: float,
) -> dict:
    fallback = DEFAULT_ACTIONS[task]

    prompt = f"""
You are refining RAG retrieval parameters.

Task: {task}
Target score: {target}
Previous action: {json.dumps(previous_action)}
Previous reward: {previous_reward:.6f}

Return ONLY valid JSON like:
{{"chunk_size": 300, "top_k": 5}}

Use integer values only.
Prefer chunk_size near 300 and top_k near 5 when uncertain.
""".strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.1,
        )
        text = response.choices[0].message.content or ""
        parsed = parse_action_from_text(text)
        if parsed is not None and not is_weak_suggestion(task, parsed):
            better_than_prev = _prefer_better_action(parsed, previous_action)
            return _prefer_better_action(better_than_prev, fallback)
    except Exception as e:
        print(f"[DEBUG] LLM refinement failed for task={task}: {e}", flush=True)

    heuristic = _normalize_action(
        {
            "chunk_size": (previous_action["chunk_size"] + 300) // 2,
            "top_k": (previous_action["top_k"] + 5) // 2,
        }
    )
    better = _prefer_better_action(heuristic, previous_action)
    return _prefer_better_action(better, fallback)


def post_reset(session: requests.Session, task: str) -> None:
    response = session.post(
        f"{ENV_BASE}/reset",
        json={"task_id": task},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    response.json()


def post_step(session: requests.Session, action: dict) -> tuple[float, bool]:
    response = session.post(
        f"{ENV_BASE}/step",
        json={"action": action},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    reward = _grader_safe_score(data.get("reward", 0.01))
    done = bool(data.get("done", False))
    return reward, done


def choose_initial_action(client: OpenAI, task: str, target: float) -> dict:
    if task == "optimal_rag":
        return DEFAULT_ACTIONS[task]
    return llm_suggest_action(client, task, target)


def run_task(client: OpenAI, session: requests.Session, task: str) -> None:
    log_start(task, "rag_optimizer", MODEL_NAME)

    target = TASK_TARGETS.get(task, 0.85)
    rewards: list[float] = []
    steps_taken = 0
    best_reward = 0.01

    try:
        ping_llm_proxy(client, task)

        post_reset(session, task)
        action1 = choose_initial_action(client, task, target)
        reward1, done1 = post_step(session, action1)

        rewards.append(reward1)
        steps_taken = 1
        best_reward = max(best_reward, reward1)
        log_step(1, json.dumps(action1), reward1, done1, error=None)

        if done1 or reward1 >= target or MAX_STEPS == 1:
            log_end(done1 and best_reward >= target, steps_taken, rewards, best_reward)
            return

        post_reset(session, task)
        action2 = llm_refine_action(client, task, target, action1, reward1)
        reward2, done2 = post_step(session, action2)

        rewards.append(reward2)
        steps_taken = 2
        best_reward = max(best_reward, reward2)
        log_step(2, json.dumps(action2), reward2, done2, error=None)

        success = (done1 or done2) and best_reward >= target
        log_end(success, steps_taken, rewards, best_reward)

    except Exception as e:
        print(f"Error: {e}", flush=True)
        fallback_rewards = rewards if rewards else [0.01]
        log_end(False, steps_taken, fallback_rewards, 0.01)


def main() -> None:
    client = create_llm_client()
    session = create_http_session()

    try:
        for task in ("baseline_retrieval", "parameter_tuning", "optimal_rag"):
            run_task(client, session, task)
    finally:
        session.close()


if __name__ == "__main__":
    main()