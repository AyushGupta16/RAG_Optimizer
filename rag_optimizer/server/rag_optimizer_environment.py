import uuid
from models import RagOptimizerAction, RagOptimizerObservation, RagOptimizerState
from openenv.core.env_server import Environment

class RagOptimizerEnvironment(Environment):
    def __init__(self):
        self._state = RagOptimizerState()

    def reset(self, seed=None, episode_id=None, **kwargs) -> RagOptimizerObservation:
        self._state = RagOptimizerState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0
        )
        return RagOptimizerObservation(
            done=False,
            reward=0.0,
            retrieval_score=0.0,
            message="Find the optimal chunk_size and top_k for the dataset."
        )

    def step(self, action: RagOptimizerAction, **kwargs) -> RagOptimizerObservation:
        self._state.step_count += 1
        
        # Simulated logic: Chunk size 300 and Top-K 5 is "Perfect"
        # We calculate how close the agent is to these values
        size_err = abs(action.chunk_size - 300) / 700
        k_err = abs(action.top_k - 5) / 5
        
        # Score ranges from 0.0 to 1.0
        score = max(0.0, 1.0 - (size_err + k_err) / 2)
        
        done = self._state.step_count >= 10 or score >= self._state.target_score
        reward = score if done else 0.0

        return RagOptimizerObservation(
            done=done,
            reward=reward,
            retrieval_score=score,
            message=f"Step {self._state.step_count}: Current score is {round(score, 2)}"
        )

    @property
    def state(self) -> RagOptimizerState:
        return self._state