# ShotGraph

AI Cinematic Video Generation Pipeline - Convert long-form stories into cinematic videos using open-source AI models.

## Overview

ShotGraph is a modular, multi-agent AI pipeline that transforms text stories into cinematic videos. The system uses:

- **LLMs** (Together.ai with cost-optimized per-stage routing) for scene breakdown and shot planning
- **Diffusion Models** (Stable Video Diffusion) for video generation
- **TTS Models** (Edge TTS) for narration
- **MusicGen** for background music
- **MoviePy/FFmpeg** for video composition

## Features

- **Cost-Optimized LLM Routing**: Per-stage model selection (Gemma 3N for compression, Llama 3.1-8B for scenes, Maverick for shots) reduces costs by 30-40%
- **TOON Format Support**: Token-efficient format saves ~40% tokens on output-heavy stages
- **Structured Outputs**: JSON schema validation with automatic repair using Together.ai structured outputs
- **Safety Moderation**: Content safety checks before video generation using VirtueGuard
- **Dual Execution Profiles**: DEBUG_CPU for local development, PROD_GPU for production
- **Modular Agent Architecture**: Each pipeline stage is a separate, testable agent
- **Automatic GPU Detection**: Falls back gracefully to CPU mode
- **Parallel LLM Processing**: Optional parallel shot planning for faster processing
- **Visual Continuity**: Last-frame initialization for smooth shot transitions
- **Mock Asset Generation**: Fast testing without AI models
- **Cost Tracking**: Automatic logging and reporting of token usage and estimated costs per stage

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/shotgraph.git
cd shotgraph

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install development dependencies (see "Debugging Installation" section below)
pip install -r requirements-dev.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Debugging Installation

ShotGraph provides three requirements files for different use cases:

1. **`requirements-dev.txt`** - Minimal dependencies for debugging
   - Lightweight installation (~100MB)
   - No ML libraries (torch, diffusers, etc.)
   - Perfect for testing pipeline flow on CPU-only systems
   - Use with `EXECUTION_PROFILE=debug_cpu` in `.env`

2. **`requirements-dev-full.txt`** - Full dependencies with CPU-only PyTorch
   - All dependencies including ML libraries
   - CPU-only PyTorch (smaller, no CUDA)
   - Suitable for debugging on 16GB RAM systems
   - Installation steps:
     ```bash
     # Install CPU-only PyTorch first
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
     
     # Then install remaining dependencies
     pip install -r requirements-dev-full.txt
     ```
   - Use with `EXECUTION_PROFILE=debug_cpu` and `GPU_ENABLED=false` in `.env`

3. **`requirements.txt`** - Full dependencies with GPU PyTorch
   - Complete production dependencies
   - GPU-enabled PyTorch (requires CUDA)
   - For production deployment with GPU
   - Use with `EXECUTION_PROFILE=prod_gpu` and `GPU_ENABLED=true` in `.env`

**Recommendation**: Start with `requirements-dev.txt` for quick debugging. Use `requirements-dev-full.txt` if you need to test with all dependencies installed but on a CPU-only system.

### Running the API

```bash
# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Open API docs
# http://localhost:8000/docs
```

### Generate a Video

```bash
# Using curl
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"story": "Once upon a time in a magical kingdom...", "title": "The Quest"}'

# Check status
curl "http://localhost:8000/status/{job_id}"

# Download video when complete
curl -O "http://localhost:8000/video/{job_id}"
```

## Configuration

All configuration is via environment variables or `.env` file:

```bash
# Execution Profile
EXECUTION_PROFILE=debug_cpu  # or prod_gpu
GPU_ENABLED=false

# LLM Configuration
LLM_PROVIDER=together  # together, groq
LLM_API_KEY=your_key
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3  # Fallback model (per-stage models configured below)
LLM_PARALLEL=false  # Enable parallel shot planning

# Per-Stage Model Configuration (Cost-Optimized Routing)
# Step A - Story compression (cheapest: $0.02/$0.04 per 1M tokens)
LLM_MODEL_STORY_COMPRESS=google/gemma-3n-E4B-it
LLM_MODEL_STORY_COMPRESS_FALLBACK=meta-llama/Llama-3.2-3B-Instruct-Turbo

# Step B - Scene breakdown ($0.18/$0.18 per 1M tokens)
LLM_MODEL_SCENE_DRAFT=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
LLM_MODEL_SCENE_DRAFT_LARGE=meta-llama/Llama-4-Scout-17B-16E-Instruct

# Step C - Shot planning ($0.27/$0.85 per 1M tokens)
LLM_MODEL_SHOT_FINAL=meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
LLM_MODEL_SHOT_FINAL_FALLBACK=meta-llama/Llama-3.3-70B-Instruct-Turbo

# Step D - JSON repair ($0.06/$0.06 per 1M tokens)
LLM_MODEL_JSON_REPAIR=meta-llama/Llama-3.2-3B-Instruct-Turbo

# Safety/moderation
LLM_SAFETY_MODEL=meta-llama/Llama-Guard-4-12B

# Cost control thresholds
LLM_SKIP_SUMMARIZATION_THRESHOLD=2000  # Skip summarization if story < N tokens
LLM_USE_LARGE_CONTEXT_THRESHOLD=8000   # Use large context model if input > N tokens

# TOON Format (saves ~40% tokens)
USE_TOON_FORMAT=true

# Video Generation
VIDEO_MODEL=stable-video-diffusion
VIDEO_RESOLUTION=1024x576
VIDEO_FPS=24

# Storage
STORAGE_PATH=./output
ASSETS_PATH=./assets
```

## Project Structure

```
ShotGraph/
├── app/                 # FastAPI application
│   ├── main.py         # Endpoints
│   ├── schemas.py      # Request/response models
│   └── dependencies.py # Dependency injection
├── core/               # Core domain logic
│   ├── models.py       # Pydantic models
│   ├── orchestrator.py # Pipeline coordinator
│   ├── agents/         # Pipeline agents
│   ├── protocols/      # Service interfaces
│   └── services/       # Service implementations
├── config/             # Configuration
│   ├── settings.py     # Pydantic Settings
│   └── prompts/        # LLM prompt templates
├── tests/              # Test suite
├── Dockerfile          # Production container
└── requirements.txt    # Dependencies
```

## Pipeline Architecture

```
Story Input
    │
    ▼
┌───────────────────────┐
│ NLP Preprocessing     │ ← Entity extraction, summarization (Step A: Gemma 3N)
└───────────────────────┘
    │
    ▼
┌───────────────────────┐
│ Scene Breakdown       │ ← LLM splits story (Step B: Llama 3.1-8B or Scout)
└───────────────────────┘
    │
    ▼
┌───────────────────────┐
│ Shot Planning         │ ← LLM creates shots (Step C: Maverick, can run parallel)
└───────────────────────┘
    │
    ▼
┌───────────────────────┐
│ JSON Repair (if needed)│ ← Fix malformed JSON (Step D: Llama 3.2-3B)
└───────────────────────┘
    │
    ▼
┌───────────────────────┐
│ Safety Check          │ ← Content moderation (VirtueGuard)
└───────────────────────┘
    │
    ▼
┌───────────────────────┐
│ Video Generation      │ ← Diffusion model generates clips
└───────────────────────┘
    │
    ▼
┌───────────────────────┐
│ Audio Generation      │ ← TTS + Background music
└───────────────────────┘
    │
    ▼
┌───────────────────────┐
│ Video Composition     │ ← Stitch clips, add audio, subtitles
└───────────────────────┘
    │
    ▼
Final Video (MP4)
```

### Cost-Optimized Routing

ShotGraph uses intelligent model routing to minimize costs:

- **Step A (Story Compression)**: Uses ultra-cheap Gemma 3N ($0.02/$0.04 per 1M tokens) for summarization
- **Step B (Scene Breakdown)**: Uses Llama 3.1-8B ($0.18/$0.18) for most cases, Scout for large contexts
- **Step C (Shot Planning)**: Uses Maverick ($0.27/$0.85) for quality, falls back to 70B for hard cases
- **Step D (JSON Repair)**: Uses cheap Llama 3.2-3B ($0.06/$0.06) only when needed

**Expected Savings**: 30-40% reduction in LLM costs compared to using a single model throughout.

**TOON Format**: Enabled by default, reduces token usage by ~40% on output-heavy stages (B/C).

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=app

# Run specific test file
pytest tests/unit/test_models.py -v
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Fix lint issues
ruff check --fix .
```

## Production Deployment

### Docker

```bash
# Build image
docker build -t shotgraph:latest .

# Run container
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -e LLM_API_KEY=your_key \
  -v $(pwd)/output:/app/output \
  shotgraph:latest
```

### RunPod

1. Push Docker image to registry
2. Create RunPod pod with GPU (RTX 3090 or A100 recommended)
3. Deploy container with environment variables
4. Access API via RunPod URL

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/generate` | Start video generation |
| GET | `/status/{job_id}` | Get job status |
| GET | `/jobs` | List all jobs |
| GET | `/video/{job_id}` | Download completed video |

## Cost Optimization

ShotGraph implements intelligent cost optimization through:

1. **Per-Stage Model Routing**: Uses cheaper models for simple tasks (Gemma 3N for summarization) and more powerful models only when needed (Maverick for shot planning)
2. **TOON Format**: Reduces token usage by ~40% on output-heavy stages
3. **Smart Summarization**: Skips summarization for short stories to avoid unnecessary costs
4. **JSON Repair**: Uses cheap models to fix malformed responses instead of expensive retries
5. **Cost Tracking**: Automatic logging of token usage and estimated costs per stage

**Expected Cost Savings**: 30-40% reduction compared to using a single model throughout the pipeline.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Together.ai](https://together.ai) - LLM API with cost-optimized routing
- [Stability AI](https://stability.ai) - Stable Video Diffusion
- [Edge TTS](https://github.com/rany2/edge-tts) - Cloud-based TTS
- [Meta AI](https://ai.meta.com) - MusicGen
