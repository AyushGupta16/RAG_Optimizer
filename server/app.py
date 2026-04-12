# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Rag Optimizer Environment.

This module creates an HTTP server that exposes the RagOptimizerEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import os
import sys

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

# Add project root to path so root-level imports work consistently
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from models import RagOptimizerAction, RagOptimizerObservation
from server.rag_optimizer_environment import RagOptimizerEnvironment

app = create_app(
    RagOptimizerEnvironment,
    RagOptimizerAction,
    RagOptimizerObservation,
    env_name="rag_optimizer",
    max_concurrent_envs=1,
)


@app.get("/")
def root():
    return {"status": "ok", "env": "rag_optimizer"}


def main():
    """Run the FastAPI server."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Rag Optimizer server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    uvicorn.run("server.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()