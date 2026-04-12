# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rag Optimizer Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import RagOptimizerAction, RagOptimizerObservation


class RagOptimizerEnv(
    EnvClient[RagOptimizerAction, RagOptimizerObservation, State]
):
    def _step_payload(self, action: RagOptimizerAction) -> Dict:
        return {
            "chunk_size": action.chunk_size,
            "top_k": action.top_k,
        }

    def _parse_result(self, payload: Dict) -> StepResult[RagOptimizerObservation]:
        obs_data = payload.get("observation", {})
        observation = RagOptimizerObservation(
            retrieval_score=obs_data.get("retrieval_score", 0.0),
            message=obs_data.get("message", ""),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )