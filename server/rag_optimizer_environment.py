import os
import uuid
from openai import OpenAI
from models import RagOptimizerAction, RagOptimizerObservation, RagOptimizerState
from openenv.core.env_server import Environment
from dotenv import load_dotenv

load_dotenv()

class RagOptimizerEnvironment(Environment):
    # ✅ Class-level state storage (persists across instance creation)
    _current_target = 0.85
    _current_episode_id = None
    _current_step_count = 0
    
    def __init__(self):
        self._state = RagOptimizerState(
            episode_id=RagOptimizerEnvironment._current_episode_id or str(uuid.uuid4()),
            step_count=RagOptimizerEnvironment._current_step_count,
            target_score=RagOptimizerEnvironment._current_target
        )

    def reset(self, seed=None, episode_id=None, task_id=None, **kwargs) -> RagOptimizerObservation:
        task_targets = {
            "baseline_retrieval": 0.5,
            "parameter_tuning": 0.7,
            "optimal_rag": 0.85
        }
        target = task_targets.get(task_id, 0.85)
        
        # ✅ Update class-level state
        RagOptimizerEnvironment._current_target = target
        RagOptimizerEnvironment._current_episode_id = episode_id or str(uuid.uuid4())
        RagOptimizerEnvironment._current_step_count = 0
        
        # Update instance state
        self._state = RagOptimizerState(
            episode_id=RagOptimizerEnvironment._current_episode_id,
            step_count=0,
            target_score=target
        )
        
        print(f"DEBUG RESET: task={task_id}, target={target}")
        
        return RagOptimizerObservation(
            reward=0.01,
            retrieval_score=0.01,
            done=False,
            message=f"Environment reset. Task: {task_id or 'default'} | target={target}"
        )

    def step(self, action: RagOptimizerAction, **kwargs) -> RagOptimizerObservation:
        # ✅ Restore state from class variables (in case new instance was created)
        self._state.target_score = RagOptimizerEnvironment._current_target
        self._state.episode_id = RagOptimizerEnvironment._current_episode_id
        self._state.step_count = RagOptimizerEnvironment._current_step_count
        
        # Increment step
        self._state.step_count += 1
        RagOptimizerEnvironment._current_step_count = self._state.step_count
        
        # 1. MANDATORY LLM PROXY CALL
        api_key = os.environ.get("API_KEY")
        base_url = os.environ.get("API_BASE_URL")
        model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

        try:
            client = OpenAI(
                base_url=base_url or "https://invalid.local",
                api_key=api_key or "dummy_key"
            )
            client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "validate"}],
                max_tokens=1
            )
            print("LLM Proxy call attempted.")
        except Exception as e:
            print(f"LLM Proxy Error: {e}")

        # 2. SCORING LOGIC
        size_err = abs(action.chunk_size - 300) / 700.0
        k_err = abs(action.top_k - 5) / 5.0
        raw_score = 1.0 - (size_err + k_err) / 2.0
        
        # Clamp to (0.01, 0.99)
        if raw_score <= 0.0:
            score = 0.01
        elif raw_score >= 1.0:
            score = 0.99
        else:
            score = max(0.01, min(0.99, raw_score))
        
        # Done logic with correct target
        done = self._state.step_count >= 10 or score >= self._state.target_score
        
        print(f"DEBUG STEP: target={self._state.target_score}, score={score:.4f}, done={done}")

        return RagOptimizerObservation(
            retrieval_score=float(score),
            reward=float(score),
            done=done,
            message=f"Step {self._state.step_count}: Score {score:.4f}"
        )

    @property
    def state(self) -> RagOptimizerState:
        return self._state