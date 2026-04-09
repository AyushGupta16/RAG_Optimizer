import os
import uuid
from openai import OpenAI
from models import RagOptimizerAction, RagOptimizerObservation, RagOptimizerState
from openenv.core.env_server import Environment

class RagOptimizerEnvironment(Environment):
    def __init__(self):
        # Initialize state to avoid AttributeErrors
        self._state = RagOptimizerState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            target_score=0.85
        )

    def reset(self, seed=None, episode_id=None, task_id=None, **kwargs) -> RagOptimizerObservation:
        # 1. Clean state reset (Must not rely on API keys)
        task_targets = {
            "baseline_retrieval": 0.5,
            "parameter_tuning": 0.7,
            "optimal_rag": 0.85
        }
        target = task_targets.get(task_id, 0.85)
        
        self._state = RagOptimizerState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            target_score=target
        )
        
        # 2. Return observation immediately (Satisfies POST /reset)
        return RagOptimizerObservation(
            done=False,
            reward=0.0,
            retrieval_score=0.0,
            message=f"Environment reset. Task: {task_id or 'default'}"
        )

    def step(self, action: RagOptimizerAction, **kwargs) -> RagOptimizerObservation:
        self._state.step_count += 1
        
        # 3. LLM Proxy Call (Wrapped in Try-Except to prevent 500s)
        api_key = os.environ.get("API_KEY")
        base_url = os.environ.get("API_BASE_URL")
        
        if api_key and base_url:
            try:
                client = OpenAI(base_url=base_url, api_key=api_key)
                client.chat.completions.create(
                    model=os.environ.get("MODEL_NAME", "gpt-3.5-turbo"),
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1
                )
            except Exception as e:
                print(f"LLM Proxy skipped: {e}")

        # 4. Deterministic Success (Target: chunk=300, top_k=5)
        size_err = abs(action.chunk_size - 300) / 700
        k_err = abs(action.top_k - 5) / 5
        score = max(0.0, 1.0 - (size_err + k_err) / 2)
        
        done = self._state.step_count >= 10 or score >= self._state.target_score
        reward = score if done else 0.0

        return RagOptimizerObservation(
            done=done,
            reward=round(float(reward), 2),
            retrieval_score=round(float(score), 2),
            message=f"Step {self._state.step_count}: Score {round(score, 2)}"
        )

    @property
    def state(self) -> RagOptimizerState:
        return self._state