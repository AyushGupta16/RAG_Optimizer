from openenv.core.env_server import Action, Observation, State


class RagOptimizerAction(Action):
    chunk_size: int
    top_k: int


class RagOptimizerObservation(Observation):
    retrieval_score: float
    message: str
    reward: float = 0.0
    done: bool = False


class RagOptimizerState(State):
    current_chunk_size: int = 500
    current_top_k: int = 3
    target_score: float = 0.85