from typing import List, Optional
from openenv.core.env_server import Action, Observation, State

class RagOptimizerAction(Action):
    chunk_size: int      # Agent chooses size (e.g., 100-1000)
    top_k: int           # Agent chooses how many docs to retrieve

class RagOptimizerObservation(Observation):
    retrieval_score: float 
    message: str
    reward: float = 0.0      
    done: bool = False 

class RagOptimizerState(State):
    current_chunk_size: int = 500
    current_top_k: int = 3
    target_score: float = 0.85