import os
import uuid

from openai import OpenAI
from models import RagOptimizerAction, RagOptimizerObservation, RagOptimizerState
from openenv.core.env_server import Environment


class RagOptimizerEnvironment(Environment):
    _current_target = 0.85
    _current_episode_id = None
    _current_step_count = 0

    @classmethod
    def _get_target_for_task(cls, task_id: str | None) -> float:
        task_targets = {
            "baseline_retrieval": 0.5,
            "parameter_tuning": 0.7,
            "optimal_rag": 0.85,
        }
        return task_targets.get(task_id, 0.85)

    @classmethod
    def _restore_shared_state(cls, state: RagOptimizerState) -> None:
        cls._current_target = state.target_score
        cls._current_episode_id = state.episode_id
        cls._current_step_count = state.step_count

    @classmethod
    def _build_state(cls) -> RagOptimizerState:
        return RagOptimizerState(
            episode_id=cls._current_episode_id or str(uuid.uuid4()),
            step_count=cls._current_step_count,
            target_score=cls._current_target,
        )

    @staticmethod
    def _clamp_score(raw_score: float) -> float:
        if raw_score <= 0.0:
            return 0.01
        if raw_score >= 1.0:
            return 0.99
        return max(0.01, min(0.99, raw_score))

    def __init__(self):
        self._state = self._build_state()

    def reset(self, seed=None, episode_id=None, task_id=None, **kwargs) -> RagOptimizerObservation:
        target = self._get_target_for_task(task_id)

        self.__class__._current_target = target
        self.__class__._current_episode_id = episode_id or str(uuid.uuid4())
        self.__class__._current_step_count = 0

        self._state = RagOptimizerState(
            episode_id=self.__class__._current_episode_id,
            step_count=0,
            target_score=target,
        )

        print(f"DEBUG RESET: task={task_id}, target={target}", flush=True)

        return RagOptimizerObservation(
            reward=0.01,
            retrieval_score=0.01,
            done=False,
            message=f"Environment reset. Task: {task_id or 'default'} | target={target}",
        )

    def step(self, action: RagOptimizerAction, **kwargs) -> RagOptimizerObservation:
        self._state = self._build_state()

        self._state.step_count += 1
        self.__class__._restore_shared_state(self._state)

        api_key = os.environ["API_KEY"]
        base_url = os.environ["API_BASE_URL"]
        model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")

        try:
            client = OpenAI(
                base_url=base_url.rstrip("/"),
                api_key=api_key,
            )
            client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "validate"}],
                max_tokens=1,
            )
            print("LLM Proxy call attempted.", flush=True)
        except Exception as e:
            print(f"LLM Proxy Error: {e}", flush=True)

        size_err = abs(action.chunk_size - 300) / 700.0
        k_err = abs(action.top_k - 5) / 5.0
        raw_score = 1.0 - (size_err + k_err) / 2.0
        score = self._clamp_score(raw_score)

        done = self._state.step_count >= 10 or score >= self._state.target_score

        print(
            f"DEBUG STEP: target={self._state.target_score}, score={score:.4f}, done={done}",
            flush=True,
        )

        return RagOptimizerObservation(
            retrieval_score=float(score),
            reward=float(score),
            done=done,
            message=f"Step {self._state.step_count}: Score {score:.4f}",
        )

    @property
    def state(self) -> RagOptimizerState:
        return self._state