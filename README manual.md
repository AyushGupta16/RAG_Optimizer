---
title: Rag Optimizer Environment Server
emoji: server
colorFrom: red
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
    - openenv
---
<!-- markdownlint-disable MD025 -->

# Rag Optimizer Environment

A reinforcement learning environment for optimizing Retrieval-Augmented Generation (RAG) parameters such as `<span>chunk_size</span>` and `<span>top_k</span>`.

This environment simulates retrieval performance and provides a reward signal, enabling agents to learn optimal configurations through interaction.

## Key Features

* RL-style environment (state, action, reward, done)
* FastAPI-based HTTP + WebSocket interface
* Supports Docker deployment and Hugging Face Spaces
* Designed for experimentation with:
  * Hyperparameter optimization
  * RL agents (Q-learning, PPO, etc.)
  * LLM-based decision agents

## Quick Start

The simplest way to use the Rag Optimizer environment is through the `RagOptimizerEnv` class:

```python
from rag_optimizer 
import RagOptimizerAction, RagOptimizerEnv 
try: 
    env = RagOptimizerEnv.from_docker_image("rag_optimizer-env:latest") 
    result = env.reset()
    print(result.observation.message)
    for _ in range(5): 
        action = RagOptimizerAction(chunk_size=300, top_k=5) 
        result = env.step(action) 
        print(result.observation.message) 
        print("Reward:", result.reward) 
finally: 
    env.close()
```

## Environment Details

### Action

**RagOptimizerAction**: The agent optimizes retrieval by adjusting:

* `chunk_size` (int): Size of text segments (Optimal: 300)
* `top_k` (int): Number of documents to retrieve (Optimal: 5)

### Observation

**RagOptimizerObservation**: Returns the result of the retrieval strategy:

* `retrieval_score` (float): Accuracy score from 0.0 to 1.0.
* `message` (str): Feedback on the current step.

### Reward

The reward is a proximity-based score. A perfect match of (300, 5) returns a reward of **1.0** and terminates the episode.

The reward is based on how close the action is to a hidden optimal configuration.

#### Enhancements

* Dynamic optimal values per episode (not fixed)
* Noise added to simulate real retrieval uncertainty
* Penalization for large parameter jumps
* Stateful updates across steps

#### Example Logic

```python
size_err = abs(action.chunk_size - optimal_chunk) / 700
k_err = abs(action.top_k - optimal_k) / 5

score = 1.0 - (size_err + k_err) / 2

# Add noise
score += random.uniform(-0.05, 0.05)

# Clip score
score = max(0.0, min(score, 1.0))
```

### Episode Termination

An episode ends when:

* Target score is reached (`<span>score >= target_score</span>`)
* OR max steps reached (`<span>step_count >= 10</span>`)

## Running the Server

### Recommended (Development)

```bash
uvicorn server.app:app --reload
```

Access:

* API Docs -> [http://localhost:8000/docs](http://localhost:8000/docs)
* Web UI -> [http://localhost:8000/web](http://localhost:8000/web)

### Important Note

Do **NOT** use:

```text
http://0.0.0.0:8000
```

Use:

```text
http://localhost:8000
```

## API Usage

### Reset

```http
POST /reset
```

Response:

```json
{
  "observation": {
    "retrieval_score": 0,
    "message": "Find the optimal chunk_size and top_k for the dataset."
  },
  "reward": 0,
  "done": false
}
```

### Step

```http
POST /step
```

Request:

```json
{
  "action": {
    "chunk_size": 300,
    "top_k": 5
  }
}
```

That's it! The `RagOptimizerEnv.from_docker_image()` method handles:

* Starting the Docker container
* Waiting for the server to be ready
* Connecting to the environment
* Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t rag_optimizer-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:

1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

* Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

* `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
* `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
* `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
* `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:

* **Web Interface** at `/web` - Interactive UI for exploring the environment
* **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
* **Health Check** at `/health` - Container health monitoring
* **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Advanced Usage

### Connecting to an Existing Server

If you already have a Rag Optimizer environment server running, you can connect directly:

```python
from rag_optimizer import RagOptimizerEnv

# Connect to existing server
rag_optimizerenv = RagOptimizerEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = rag_optimizerenv.reset()
result = rag_optimizerenv.step(RagOptimizerAction(message="Hello!"))
```

Note: When connecting to an existing server, `rag_optimizerenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from rag_optimizer import RagOptimizerAction, RagOptimizerEnv

# Connect with context manager (auto-connects and closes)
with RagOptimizerEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(RagOptimizerAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
```

The client uses WebSocket connections for:

* **Lower latency**: No HTTP connection overhead per request
* **Persistent session**: Server maintains your environment state
* **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    RagOptimizerEnvironment,  # Pass class, not instance
    RagOptimizerAction,
    RagOptimizerObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from rag_optimizer import RagOptimizerAction, RagOptimizerEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with RagOptimizerEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(RagOptimizerAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/rag_optimizer_environment.py
```

This verifies that:

* Environment resets correctly
* Step executes actions properly
* State tracking works
* Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

OR

```bash
python server/rag_optimizer_environment.py
```

## Project Structure

```text
rag_optimizer/
|-- .dockerignore         # Docker build exclusions
|-- __init__.py            # Module exports
|-- README.md              # This file
|-- openenv.yaml           # OpenEnv manifest
|-- pyproject.toml         # Project metadata and dependencies
|-- uv.lock                # Locked dependencies (generated)
|-- client.py              # RagOptimizerEnv client
|-- models.py              # Action and Observation models
\-- server/
    |-- __init__.py        # Server module exports
    |-- rag_optimizer_environment.py  # Core environment logic
    |-- app.py             # FastAPI application (HTTP + WebSocket endpoints)
    \-- Dockerfile         # Container image definition
```

## Future Improvements

* Plug into real vector DB (FAISS / Chroma)
* Use real retrieval metrics (Recall@k, MRR)
* Train RL agents (PPO, DQN)
* Integrate LLM-based tuning agents

## Summary

This project is no longer a simple echo environment.

It is now a  **mini RL environment for RAG optimization** , suitable for:

* Research experiments
* ML system design practice
* Real-world retrieval tuning

<!-- markdownlint-enable MD025 -->
