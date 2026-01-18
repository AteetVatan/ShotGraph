# ShotGraph

AI Cinematic Video Generation Pipeline - Convert long-form stories into cinematic videos using open-source AI models.

## Overview

ShotGraph is a modular, multi-agent AI pipeline that transforms text stories into cinematic videos. The system uses:

- **LLMs** (Mistral, via Together.ai/Groq) for scene breakdown and shot planning
- **Diffusion Models** (Stable Video Diffusion) for video generation
- **TTS Models** (AI4Bharat, Coqui) for narration
- **MusicGen** for background music
- **MoviePy/FFmpeg** for video composition

## Features

- **Dual Execution Profiles**: DEBUG_CPU for local development, PROD_GPU for production
- **Modular Agent Architecture**: Each pipeline stage is a separate, testable agent
- **Automatic GPU Detection**: Falls back gracefully to CPU mode
- **Parallel LLM Processing**: Optional parallel shot planning for faster processing
- **Visual Continuity**: Last-frame initialization for smooth shot transitions
- **Mock Asset Generation**: Fast testing without AI models

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
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3
LLM_PARALLEL=false  # Enable parallel shot planning

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
┌───────────────────┐
│ Scene Breakdown   │ ← LLM splits story into scenes
└───────────────────┘
    │
    ▼
┌───────────────────┐
│ Shot Planning     │ ← LLM creates cinematic shots (can run parallel)
└───────────────────┘
    │
    ▼
┌───────────────────┐
│ Video Generation  │ ← Diffusion model generates clips
└───────────────────┘
    │
    ▼
┌───────────────────┐
│ Audio Generation  │ ← TTS + Background music
└───────────────────┘
    │
    ▼
┌───────────────────┐
│ Video Composition │ ← Stitch clips, add audio, subtitles
└───────────────────┘
    │
    ▼
Final Video (MP4)
```

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

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Together.ai](https://together.ai) - LLM API
- [Stability AI](https://stability.ai) - Stable Video Diffusion
- [AI4Bharat](https://ai4bharat.org) - Indic TTS
- [Meta AI](https://ai.meta.com) - MusicGen
