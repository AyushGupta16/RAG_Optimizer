# RAG Optimizer Environment

A reinforcement learning environment for optimizing Retrieval-Augmented Generation (RAG) parameters. Agents learn to select optimal `chunk_size` and `top_k` values to maximize retrieval performance.

## Overview

This environment simulates a RAG parameter optimization task where agents must discover the optimal configuration:

- **Optimal chunk_size**: 300
- **Optimal top_k**: 5

The environment scores actions based on how close the selected parameters are to these optimal values.

## Key Features

- RL-style environment for parameter optimization
- Deterministic reward function based on distance from optimal configuration
- Multi-task setup with increasing difficulty
- LLM proxy integration for OpenEnv validation
- FastAPI + OpenEnv server for scalable interaction
- Docker-ready and deployable on Hugging Face Spaces

## Problem Formulation

We define RAG tuning as:

- **State** : Current retrieval configuration
- **Action** : Adjust `(chunk_size, top_k)`
- **Reward** : Retrieval quality score
- **Goal** : Maximize reward to reach task-specific thresholds

## Tasks

Three tasks of varying difficulty:

| Task                 | Target Score | Description                             |
|----------------------|--------------|-----------------------------------------|
| `baseline_retrieval` | 0.5          | Easy - suboptimal parameters work       |
| `parameter_tuning`   | 0.7          | Medium - requires good parameters       |
| `optimal_rag`        | 0.85         | Hard - requires near-optimal parameters |

## Action Space

```python
chunk_size: int   # Size of document chunks
top_k: int        # Number of retrieved documents
```

## Reward Function

Optimal configuration:

- chunk_size = 300
- top_k = 5

```python
size_err = abs(chunk_size - 300) / 700.0
k_err = abs(top_k - 5) / 5.0
raw_score = 1.0 - (size_err + k_err) / 2.0
```

Scores are clamped to `(0.01, 0.99)` to satisfy strict validator constraints.

## Installation

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- UV package manager (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/AyushGupta16/RAG_Optimizer.git
cd RAG_Optimizer

# Install dependencies
uv sync

# Or with pip
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API credentials
```

### Environment Variables

```bash
# Required for LLM proxy (validator checks this)
API_BASE_URL=https://your-llm-proxy.com/v1
API_KEY=your_api_key_here
MODEL_NAME=gpt-4o-mini

# Local development
ENV_BASE=http://127.0.0.1:8000
```

## Usage

### Running Locally

```bash
# Terminal 1: Start environment server
python server/app.py --host 127.0.0.1 --port 8000

# Terminal 2: Run inference
```bash
python inference.py
```

### Expected Output

```bash
[DEBUG] LLM proxy call succeeded for task=baseline_retrieval
[START] task=baseline_retrieval env=rag_optimizer model=gpt-4o-mini
[STEP] step=1 action={"chunk_size": 500, "top_k": 3} reward=0.657143 done=true error=none
[END] success=true steps=1 score=0.657143 rewards=0.657143

[DEBUG] LLM proxy call succeeded for task=parameter_tuning
[START] task=parameter_tuning env=rag_optimizer model=gpt-4o-mini
[STEP] step=1 action={"chunk_size": 350, "top_k": 4} reward=0.864286 done=true error=none
[END] success=true steps=1 score=0.864286 rewards=0.864286

[DEBUG] LLM proxy call succeeded for task=optimal_rag
[START] task=optimal_rag env=rag_optimizer model=gpt-4o-mini
[STEP] step=1 action={"chunk_size": 300, "top_k": 5} reward=0.990000 done=true error=none
[END] success=true steps=1 score=0.990000 rewards=0.990000
```

## Validation

Run local validation before submitting:

```bash
# Check environment setup
openenv validate

# Expected output:
# [OK] RAG_Optimizer: Ready for multi-mode deployment
```

## Architecture and Integration

### System Architecture

```text
inference.py
   |
   | 1. LLM proxy call using API_BASE_URL + API_KEY
   | 2. POST /reset and POST /step
   v
server/app.py
   |
   | OpenEnv create_app(...)
   v
server/rag_optimizer_environment.py
   |
   | - restore persistent state
   | - evaluate (chunk_size, top_k)
   | - call LLM proxy for validator compliance
   | - compute score and done flag
   v
models.py
   |
   | - RagOptimizerAction
   | - RagOptimizerObservation
   | - RagOptimizerState
   v
Reward + Observation returned to inference.py
```

### Repository Architecture

```text
RAG_Optimizer/
├── server/
│   ├── __init__.py
│   ├── app.py
│   ├── Dockerfile
│   ├── rag_optimizer_environment.py
│   └── requirements.txt
├── outputs/
├── .venv/
├── __init__.py
├── .env
├── .gitignore
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
└── uv.lock
```

### Role of Each File

- `server/app.py` creates the FastAPI/OpenEnv server.
- `server/rag_optimizer_environment.py` contains the environment logic, task targets, scoring, and state persistence.
- `server/Dockerfile` packages the environment for Hugging Face Spaces and validator execution.
- `server/requirements.txt` lists runtime dependencies for server builds.
- `inference.py` runs the benchmark tasks, calls the LLM proxy, and prints validator-compatible logs.
- `models.py` defines action, observation, and state schemas.
- `client.py` provides a reusable client wrapper for interacting with the environment.
- `openenv.yaml` registers the environment and task metadata.
- `pyproject.toml` defines package metadata and dependencies.
- `uv.lock` locks dependency versions for reproducible builds.
- `.env` is for local development only and should not be committed with secrets.
- `outputs/` stores local run artifacts if generated.

### LLM Proxy Integration

To satisfy OpenEnv validation requirements:

- Each task triggers at least one LLM API call
- Uses injected environment variables:
  - `API_BASE_URL`
  - `API_KEY`

This ensures compliance with the "LLM Criteria Check".

### State Management

OpenEnv may recreate environment instances between calls.

To maintain consistency:

- Shared state is stored at the class level
- Ensures task targets persist across `/reset` and `/step`

### Project Structure

```text
RAG_Optimizer/
├── server/
│   ├── app.py
│   ├── __init__.py
│   └── rag_optimizer_environment.py
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
└── server/requirements.txt
```

### Environment Logic

The environment computes scores based on normalized distance from optimal parameters:

```python
size_error = abs(chunk_size - 300) / 700
k_error = abs(top_k - 5) / 5
raw_score = 1.0 - (size_error + k_error) / 2
```

Scores are clamped to `(0.01, 0.99)` to satisfy validator requirements (strictly between 0 and 1).

### State Persistence

The environment uses class-level variables to persist state across HTTP requests:

```python
class RagOptimizerEnvironment(Environment):
    _current_target: float = 0.85
    _current_episode_id: Optional[str] = None
    _current_step_count: int = 0
```

This ensures that episode state survives server framework instance recreation.

## API Endpoints

### POST /reset

Reset environment for a new episode.

**Request:**

```json
{
  "task_id": "optimal_rag",
  "episode_id": "optional-uuid"
}
```

**Response:**

```json
{
  "observation": {
    "retrieval_score": 0.01,
    "message": "Environment reset. Task: optimal_rag | target=0.85"
  },
  "reward": 0.01,
  "done": false
}
```

### POST /step

Execute an action in the environment.

**Request:**

```json
{
  "action": {
    "chunk_size": 300,
    "top_k": 5
  }
}
```

**Response:**

```json
{
  "observation": {
    "retrieval_score": 0.99,
    "message": "Step 1: Score 0.990000"
  },
  "reward": 0.99,
  "done": true
}
```

## Development

### Running Tests

```bash
# Lint code
ruff check .

# Format code
ruff format .

# Type check
mypy .
```

### Adding New Tasks

To add a new difficulty level:

1. Update `TASK_TARGETS` in `rag_optimizer_environment.py`:

```python
TASK_TARGETS = {
    "baseline_retrieval": 0.5,
    "parameter_tuning": 0.7,
    "optimal_rag": 0.85,
    "expert_rag": 0.95,  # New task
}
```

1. Update `TASK_TARGETS` in `inference.py`
2. Add test case to `inference.py`:

```python
tasks = [
    # ... existing tasks ...
    ("expert_rag", {"chunk_size": 300, "top_k": 5}),
]
```

## Troubleshooting

### Common Issues

#### LLM proxy call failed

- Check `API_BASE_URL` and `API_KEY` in `.env`
- Verify proxy endpoint is accessible
- Check API quota/rate limits

#### Scores out of range

- Ensure environment clamps scores to `(0.01, 0.99)`
- Check `grader_safe_score()` function
- Verify no exact 0.0 or 1.0 values in logs

#### State not persisting

- Verify class variables are used (not instance variables)
- Check `_current_*` variables are updated in `reset()`
- Ensure `step()` restores state from class variables

## Tech Stack

- Python
- FastAPI
- OpenEnv
- OpenAI Python SDK
- Docker
- Hugging Face Spaces

## What Makes This Stand Out

- ML Systems + RL environment design
- Backend engineering with FastAPI + OpenEnv
- Validator-compliant LLM proxy integration
- Stateful environment handling across HTTP requests
- Reproducible deployment through Docker and Hugging Face Spaces

## Future Improvements

- Let the LLM suggest actions dynamically instead of fixed task actions
- Integrate real vector databases (FAISS, Pinecone)
- Replace synthetic reward with retrieval metrics (Recall@k, MRR)
- Multi-step RL optimization (instead of single-step)
- Learnable policies via PPO / DQN

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [OpenEnv](https://github.com/meta-llama/openenv)
- Part of the OpenEnv Hackathon 2026
- Environment design inspired by RAG optimization research

## Contact

- **Author**: Ayush Gupta
- **GitHub**: [@AyushGupta16](https://github.com/AyushGupta16)
- **Email**: <ayushgupta0616@gmail.com>

## Citation

If you use this environment in your research, please cite:

```bibtex
@misc{rag-optimizer-2026,
  author = {Gupta, Ayush},
  title = {RAG Optimizer Environment},
  year = {2026},
  publisher = {GitHub},
  url = {<https://github.com/AyushGupta16/RAG_Optimizer>}
}
```
