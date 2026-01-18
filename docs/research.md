# Agentic AI Cinematic Video Generation System Design

## Overview and Goals

The goal is to build an agentic, multi-stage AI pipeline that converts a long-form story (text) into a high-quality cinematic video of 5–60 minutes. The system will use only open-source models (Plan A) for all AI tasks. It will accept a story as input and produce a cohesive video composed of multiple AI-generated clips (5–10 seconds each), stitched together with subtitles, background music, and text-to-speech audio (in English and Hindi). The design emphasizes modularity (multiple specialized agents), configurability (swappable models and prompts), and efficiency (capable of running on a 16GB RAM CPU for debugging, and leveraging GPUs for production on RunPod or similar). The system will be exposed as a FastAPI web service for triggering video generation jobs and tracking their status.

## Key Requirements and Constraints:

- **Open-Source Models Only**: All AI models (LLMs for text, generative image/video, TTS, music) must be open-source. (Plan B could integrate proprietary models, but here we focus on Plan A). For example, we will use the open Mistral-7B series for language tasks (via APIs like Together.ai or Groq), open diffusion models for video (e.g. Stable Diffusion-based text-to-video), and open TTS models.
- **Agentic Pipeline**: The system is broken into multiple agents, each responsible for a specific stage (story parsing, scene planning, video generation, audio generation, etc.). This follows the multi-agent design pattern where specialized agents collaborate in a pipeline. Each agent will adhere to SOLID principles (single responsibility, decoupled via clear interfaces, etc.).
- **Configuration & Modularity**: All configuration (model file paths, API endpoints, prompt templates, etc.) will be provided via environment variables (env file), enabling easy swapping of models or services without code changes. Hardcoded strings in logic are avoided; we use enums/constants for repeatable keys or labels to ensure maintainability.
- **Data Modeling with Pydantic**: Every agent's input and output is defined with Pydantic models for validation and clarity. For example, a Scene model contains fields like id, text, shots: List[Shot], etc., and a Shot model might include description, duration, video_file_path, etc. Using Pydantic enforces a structured JSON schema for communication between agents and for the final output format (e.g. a shot list or video manifest).
- **Performance Considerations**: The pipeline design considers speed and cost. Wherever possible, heavy computations can be offloaded to GPU (e.g. RunPod cloud) while the orchestrator and lighter logic run on CPU. The system will allow running smaller/faster models or reduced settings for quick iterations (MVP or debugging), and higher-quality models for final production. We assume a production environment with at least one GPU (e.g. 24 GB VRAM) available for diffusion models and possibly high-end LLMs, while local debugging may use CPU-friendly models or stubs.
- **Output Quality**: Despite focusing on speed, we aim for high-quality outputs. The design includes retry and re-roll mechanisms – if an agent produces unsatisfactory output (e.g. an LLM returns malformed JSON or a video frame is off-style), the system can trigger a retry or an alternate approach (with adjusted prompts or seeds). We will note assumptions like maximum retry counts (e.g. 2 attempts per agent by default) and quality-check criteria.

With these goals, the following sections detail the agent pipeline, the system architecture, the data models, and the implementation plan.

## Agent Pipeline Design

The video generation process is divided into stages handled by different agents or modules. Below is an end-to-end pipeline outline, with each agent's responsibilities, inputs, and outputs (modeled via Pydantic):

### 1. Story Ingestion & Preparation

- **Responsibility**: Accept the raw story text (which could be several pages long) and prepare it for scene splitting. If needed, pre-process the text (e.g. remove unwanted formatting).
- **Input**: Full story text (plain string, possibly with markdown or screenplay format).
- **Output**: A sanitized or standardized text format (Pydantic model: StoryInput with fields like text, title, etc.).
- **Notes**: This is a simple step, possibly just handled within the API layer or orchestrator. It ensures the story is ready for LLM processing (e.g. truncated if extremely long, or chunked if needed due to LLM context limits).

### 2. Scene Breakdown Agent (LLM-based)

- **Responsibility**: Split the story into scenes. Each scene would be a logical segment of the narrative (e.g. a change of location, time, or major event) suitable for a distinct video segment. The agent should produce a summary or description for each scene.
- **Input**: The full story text (or StoryInput from previous step). Possibly a prompt template guiding the LLM to output a JSON list of scenes.
- **Output**: A list of scenes with descriptions. For example, a Pydantic Scene model with fields: id (scene number), text (the story segment for that scene), summary (LLM-generated brief summary or setting), and possibly shots: List[Shot] (initially empty, to be filled by next agent). This could be wrapped in a SceneList model.
- **Implementation**: Use an open LLM (e.g. Mistral-7B or a similar capable model) via an API. We might use the Together.ai API to call a Mistral model (which is cost-effective and fast). The prompt will instruct the model to break the story into numbered scenes, perhaps with a brief title or summary for each.
- **Example**: If the story involves multiple chapters or acts, the LLM might output JSON like `{"scenes": [ {"id": 1, "summary": "Hero meets mentor in village...", "text": "...original text..."}, ... ]}`.
- **Retry Logic**: If the LLM output is not valid JSON or misses the format, the orchestrator will detect it (via Pydantic parsing). It can then re-prompt the LLM (e.g. adding a system message: "Output only valid JSON.") and retry. We assume up to 2 retries for well-formed output. If still failing, the job can be aborted with an error status reported.

### 3. Shot Planning Agent (LLM-based)

- **Responsibility**: For each scene, break it further into shots or camera scenes of ~5–10 seconds. Each shot will have a detailed description to guide video generation. The agent effectively storyboards the scene into a sequence of visual moments.
- **Input**: One scene's text or summary (from Scene Breakdown), plus any context (like overall style).
- **Output**: A list of Shot models for that scene. Each Shot may include: id (shot number within scene), description (text prompt describing the visuals), duration_seconds, and maybe metadata like type (e.g. wide shot, close-up, if we want to classify shot types). This could be structured as JSON per scene: e.g. `{"scene_id":1, "shots":[ {"id":1, "description":"A wide shot of the village at sunrise, birds chirping...", "duration":5}, {...}]}`.
- **Implementation**: Another LLM prompt (possibly a more creative one) will be used. For example, using a slightly larger or more creative model (if available via API, e.g. Qwen-14B via Together for complex reasoning, or still Mistral with a higher temperature). The prompt template might say: "You are a film director. Given the scene description: <scene_summary>, break it into a sequence of cinematic shots. Provide 5–10 second shots, each described vividly. Output as JSON with fields: id, description, duration."
- **Considerations**: This agent ensures the story is translated from narrative text to visual descriptions. It might also extract any dialogue lines to be spoken in that shot or note which characters are present (we could extend the Shot model with optional dialogue or subtitle_text). That way, we know what subtitle or voiceover to generate for that shot. For example, if the story text in a scene includes dialogue, the agent might attach it to the corresponding shot.
- **Retry & Re-roll**: Similar JSON validation as above. Additionally, if a description is too vague or not visual enough, a heuristic or even a second LLM check could flag it for improvement. In MVP, we likely rely on prompt quality to get good results initially, but later we could have a review agent check consistency (this is noted as a potential future improvement).

### 4. Visual Asset Generation Agent

- **Responsibility**: Generate video clips for each shot. This is the most computationally intensive stage. For each shot description, the agent will:
  - Optionally generate a keyframe image (using text-to-image, e.g. Stable Diffusion XL or similar)
  - Generate a short video clip (5–10 seconds) from the image or directly from text (using text-to-video models like Stable Video Diffusion, ModelScope, or other open models)
  - Save the video file to disk (or cloud storage) and record the path in the Shot model
- **Input**: Shot description (text prompt), optional style parameters, and possibly a reference image if we want consistency across shots.
- **Output**: A video file path (or URL) stored in the Shot model's video_file_path field. The agent may also produce metadata like resolution, frame rate, etc.
- **Implementation**: Use open-source text-to-video models. Examples include:
  - **Stable Video Diffusion** (by Stability AI): Can generate 4–25 frame videos from an image. Requires ~24GB VRAM for full quality, but can run with lower settings.
  - **ModelScope Text-to-Video**: Open model that can generate videos directly from text prompts.
  - **Other options**: As the field evolves, newer models like HunyuanVideo, Mochi, or Wan2.2 (mentioned in Modal's blog) may become available and can be swapped in via configuration.
- **Considerations**: 
  - This stage will be the bottleneck in terms of GPU time. We may need to queue shots and process them in batches or parallelize across multiple GPUs if available.
  - Quality vs. speed trade-offs: For MVP, we might use faster/lower-quality settings, then re-run with higher quality for production.
  - Consistency: To maintain visual consistency across shots in a scene, we might reuse a base image or use a seed/guidance mechanism. Some models support conditioning on previous frames.
- **Retry Logic**: If a generated video is corrupted, too short, or visually off (detected via heuristics or manual review), the agent can regenerate with adjusted parameters (e.g. different seed, stronger prompt, or fallback to a simpler model).

### 5. Audio Generation Agent

- **Responsibility**: Generate audio components for the video:
  - **Text-to-Speech (TTS)**: Convert dialogue or narration text (from shots or scenes) into speech audio in English and Hindi.
  - **Background Music**: Generate or select background music that matches the mood/tone of each scene or shot.
- **Input**: 
  - For TTS: Text strings (dialogue/narration), language (English/Hindi), voice preference.
  - For Music: Scene mood/description, duration needed.
- **Output**: 
  - TTS audio files (one per shot or scene, or combined) with paths stored in the Shot/Scene model.
  - Background music file(s) with paths.
- **Implementation**:
  - **TTS**: Use open-source TTS models:
    - **AI4Bharat Indic TTS**: Provides high-quality voices for Hindi and other Indic languages (Apache 2.0 license). Also supports English.
    - **Coqui TTS**: Rich repository of open voices, supports multiple languages.
    - **Other options**: Models like Kokoro, Orpheus (mentioned in Modal's blog) offer multi-lingual support.
  - **Music**: Use open-source music generation:
    - **MusicGen by Meta**: Can generate ~12 second clips from text prompts. Requires ~16GB GPU. Code is MIT, weights are CC BY-NC (non-commercial use).
    - **Alternatives**: Community forks or other open music generators as they become available.
- **Considerations**:
  - TTS should match the pacing of the video (e.g. if a shot is 7 seconds, the speech should fit within that duration, or we adjust the shot duration).
  - Music should be loopable or long enough to cover the scene duration. We might generate longer music clips and trim/loop as needed.
  - Audio mixing: The final composition will mix TTS and music at appropriate volumes (music quieter, TTS louder).

### 6. Subtitle Generation Agent

- **Responsibility**: Generate subtitle files (SRT format or embedded) for the video. Subtitles should include:
  - Dialogue text (from shots)
  - Narration text (if any)
  - Timing synchronized with the video/audio
- **Input**: Shot/Scene models containing dialogue text and timing information.
- **Output**: SRT file or subtitle track that can be embedded in the final video.
- **Implementation**: This is primarily a formatting/timing task. We can use libraries like `pysrt` or generate SRT manually from the shot timing and text. The timing should align with when TTS audio plays.
- **Considerations**: Subtitles should be readable (good font, size, positioning) and properly timed. We might support multiple languages (English and Hindi subtitles simultaneously, or user-selectable).

### 7. Video Composition Agent

- **Responsibility**: Stitch together all video clips, add audio tracks (TTS + music), overlay subtitles, and produce the final video file.
- **Input**: 
  - List of video clips (from Visual Asset Generation)
  - Audio files (TTS and music)
  - Subtitle file
  - Composition parameters (resolution, frame rate, transitions between clips, etc.)
- **Output**: Final video file (e.g. MP4) ready for distribution.
- **Implementation**: Use video editing libraries:
  - **MoviePy**: Pure Python library, easy to use for combining clips, adding audio, overlaying text/subtitles. Good for prototyping.
  - **FFmpeg** (via `ffmpeg-python` or direct commands): More efficient for production, handles complex audio mixing, subtitle embedding, etc.
- **Considerations**:
  - Transitions: Smooth transitions between clips (fade, cross-fade) can improve quality.
  - Audio mixing: Balance TTS and background music volumes.
  - Resolution/format: Output in a standard format (e.g. 1080p MP4) suitable for distribution.
  - Performance: For long videos (30–60 minutes), composition can take time. We should show progress and allow the job to run asynchronously.

### 8. Orchestrator/Coordinator

- **Responsibility**: Coordinate all agents, manage the pipeline flow, handle errors and retries, track job status, and expose the API endpoints.
- **Input**: Story text (from API request)
- **Output**: Job ID and status updates, final video file path/URL
- **Implementation**: 
  - FastAPI application with endpoints:
    - `POST /generate` - Start a video generation job (accepts story text, returns job_id)
    - `GET /status/{job_id}` - Get job status (pending, processing, completed, failed)
    - `GET /video/{job_id}` - Download or stream the final video (if completed)
  - Job queue/state management: Use in-memory dict for MVP, or a database (SQLite/PostgreSQL) for persistence. For production, consider using a task queue like Celery or Temporal.
  - Error handling: Catch exceptions from each agent, log them, update job status, and optionally retry failed stages.
- **Considerations**:
  - The orchestrator should be stateless where possible (or store state in a database) so it can be restarted without losing jobs.
  - Progress tracking: Report which stage is currently running (e.g. "Generating scene 3 of 10", "Rendering shot 15 of 50").
  - Timeout handling: Set reasonable timeouts for each stage (e.g. if video generation takes >5 minutes per shot, something might be wrong).

## System Architecture

### High-Level Architecture

```
┌─────────────────┐
│  FastAPI Server │  (Orchestrator)
│  /generate      │
│  /status/{id}   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Agent Pipeline (Sequential/Parallel)│
│  ┌──────────┐  ┌──────────┐         │
│  │ Scene    │→ │ Shot     │         │
│  │ Breakdown│  │ Planning │         │
│  └──────────┘  └────┬─────┘         │
│                     │                │
│         ┌───────────┴───────────┐   │
│         ▼                       ▼   │
│  ┌──────────────┐      ┌──────────────┐
│  │ Visual Asset │      │ Audio        │
│  │ Generation   │      │ Generation   │
│  └──────┬───────┘      └──────┬───────┘
│         │                     │
│         └───────────┬─────────┘
│                     ▼
│            ┌─────────────────┐
│            │ Video Composition│
│            └─────────────────┘
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Storage        │  (Local disk or cloud)
│  - Video clips  │
│  - Audio files  │
│  - Final video  │
└─────────────────┘
```

### Component Details

1. **API Layer (FastAPI)**: Handles HTTP requests, validates input, returns job IDs and status.
2. **Orchestrator**: Manages the pipeline execution, calls agents in sequence (or parallel where possible), handles retries.
3. **Agent Modules**: Each agent is a Python module/class with a clear interface (input Pydantic model → output Pydantic model).
4. **Model Services**: Wrappers around LLM APIs, diffusion model pipelines, TTS engines, etc. These can be swapped via configuration.
5. **Storage Layer**: File system or cloud storage (S3, etc.) for intermediate and final outputs.
6. **Configuration**: Environment variables or config files defining model paths, API keys, prompts, etc.

### Data Flow

1. User submits story → FastAPI receives it → Creates job record → Returns job_id
2. Orchestrator starts pipeline:
   - Story Ingestion → StoryInput
   - Scene Breakdown Agent → SceneList
   - For each Scene:
     - Shot Planning Agent → List[Shot]
     - For each Shot:
       - Visual Asset Generation → video_file_path
       - Audio Generation (TTS) → audio_file_path
   - Audio Generation (Music) per Scene → music_file_path
   - Subtitle Generation → subtitle_file_path
   - Video Composition → final_video_path
3. Job status updated to "completed", final video available via API

## Data Models (Pydantic)

Below are example Pydantic models that define the data structures used throughout the pipeline:

```python
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class Language(str, Enum):
    ENGLISH = "en"
    HINDI = "hi"

class StoryInput(BaseModel):
    text: str
    title: Optional[str] = None
    language: Language = Language.ENGLISH

class Shot(BaseModel):
    id: int
    description: str
    duration_seconds: float
    video_file_path: Optional[str] = None
    audio_file_path: Optional[str] = None
    subtitle_text: Optional[str] = None
    shot_type: Optional[str] = None  # e.g. "wide", "close-up"

class Scene(BaseModel):
    id: int
    text: str
    summary: str
    shots: List[Shot] = []
    music_file_path: Optional[str] = None

class SceneList(BaseModel):
    scenes: List[Scene]

class VideoJob(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    story_input: StoryInput
    scenes: Optional[SceneList] = None
    final_video_path: Optional[str] = None
    error_message: Optional[str] = None
    progress: Optional[str] = None  # e.g. "Generating scene 3 of 10"
```

## Implementation Plan

### Phase 1: MVP (Minimum Viable Product)

1. **Setup**: FastAPI skeleton, basic job tracking (in-memory dict), Pydantic models
2. **Story Ingestion**: Simple text sanitization
3. **Scene Breakdown**: Integrate LLM API (Together.ai Mistral-7B), prompt for JSON scene list, parse with Pydantic
4. **Shot Planning**: Similar LLM integration, generate shot descriptions
5. **Visual Asset Generation**: Integrate one text-to-video model (e.g. Stable Video Diffusion via HuggingFace diffusers), generate clips for a few test shots
6. **Audio Generation**: Integrate one TTS model (e.g. AI4Bharat), generate speech for test shots
7. **Video Composition**: Use MoviePy to stitch 2–3 test clips with audio
8. **End-to-End Test**: Generate a 30-second video from a short story

### Phase 2: Production Readiness

1. **Error Handling & Retries**: Add retry logic for all agents, proper error messages
2. **Music Generation**: Integrate MusicGen for background music
3. **Subtitle Generation**: Add subtitle overlay
4. **Storage**: Move from local files to cloud storage (S3) or organized local structure
5. **Job Persistence**: Use database (SQLite for MVP, PostgreSQL for production) instead of in-memory
6. **Progress Tracking**: Report detailed progress through API
7. **Quality Improvements**: Tune prompts, add consistency mechanisms for visuals
8. **Performance**: Optimize video generation (batch processing, GPU utilization)

### Phase 3: Scaling & Polish

1. **Multi-GPU Support**: Parallelize video generation across multiple GPUs
2. **Advanced Features**: Transitions, better audio mixing, multiple language support
3. **Monitoring & Logging**: Structured logging, metrics collection
4. **API Documentation**: OpenAPI/Swagger docs, example requests
5. **Testing**: Unit tests for each agent, integration tests for pipeline
6. **Deployment**: Docker containerization, deployment scripts for RunPod/cloud

## Configuration Management

All configuration should be externalized via environment variables or a `.env` file. Example configuration:

```bash
# LLM Configuration
LLM_API_PROVIDER=together  # or "groq", "local"
LLM_API_KEY=your_key_here
LLM_MODEL=mistral-7b-instruct

# Video Generation
VIDEO_MODEL=stable-video-diffusion
VIDEO_MODEL_PATH=/path/to/model
VIDEO_RESOLUTION=1024x576
VIDEO_FPS=24
GPU_DEVICE=cuda:0

# TTS Configuration
TTS_MODEL=ai4bharat-indic-parler
TTS_MODEL_PATH=/path/to/tts/model
TTS_VOICE_EN=en-female
TTS_VOICE_HI=hi-female

# Music Generation
MUSIC_MODEL=musicgen-medium
MUSIC_MODEL_PATH=/path/to/musicgen

# Storage
STORAGE_TYPE=local  # or "s3"
STORAGE_PATH=/path/to/videos
# If S3:
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
# S3_BUCKET=...

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

## Reusable Tools and References

In building this system, we can take advantage of existing open-source projects and tools to avoid reinventing the wheel. Below is a list of some useful resources and libraries that align with our design:

- **ComfyUI Workflows**: ComfyUI is a powerful node-based UI for diffusion models. It has community-developed workflows for text-to-video (e.g., nodes for ModelScope or Stable Video Diffusion). While our system doesn't use a UI, we can reuse the underlying workflow logic or even run a ComfyUI server in headless mode to generate video clips. This could simplify handling multi-step diffusion pipelines (e.g., using a ComfyUI workflow that takes an initial image and returns a video, which we trigger programmatically).
- **Hugging Face Diffusers**: The HuggingFace `diffusers` library provides pipelines for text-to-video (e.g., StableVideoDiffusionPipeline and others) and is continually updated by the community. Using diffusers would give us a standard interface to many models and handle details like scheduler, frame interpolation, etc. We should leverage these rather than writing our own low-level diffusion code.
- **Pydantic AI / LangGraph**: As discussed, if we wanted to formalize the agent orchestration, the Pydantic AI framework (by the Pydantic team) could be used to define agents with type-checked input/outputs, and LangGraph to connect them in a flow. Since our manual approach is similar in spirit, we can draw ideas from these frameworks. For example, Pydantic AI suggests patterns for quick agent definition and ensuring one agent's output matches the next's expected schema. Even if not using them directly, their concepts reinforce our design.
- **Temporal.io (Workflow Orchestration)**: For a highly reliable production system, one might integrate with a workflow engine like Temporal to handle long-running processes (ensuring they can resume on failure, etc.). In fact, Pydantic AI has integration with Temporal for durable execution. This could be overkill for MVP, but it's an option for scaling (especially if job needs to survive server restarts or be distributed).
- **AI4Bharat Indic TTS and Coqui TTS**: The AI4Bharat models provide high-quality voices for Hindi and other Indian languages and are open-source (Apache 2.0). Coqui TTS has a rich repository of voices and tools to train or fine-tune. We might use pre-trained voices from these sources for good results (e.g., a pleasant English narrator voice and a Hindi voice). Reusing these models saves us from training our own TTS.
- **MusicGen / AudioCraft by Meta**: The open-source release of MusicGen (part of the AudioCraft toolkit) can be directly used. There are also community forks that allow longer generation or better quality (some use diffusion for music as mentioned in the TechCrunch article where multi-band diffusion was an upgrade). We should monitor those for improvements. Additionally, Open Music libraries like Suno's Bark (which is more for speech but can do music-like sounds) might be interesting, though Bark is more for multi-lingual speech.
- **MoviePy and FFmpeg**: These will be the backbone of video editing. MoviePy is pure Python and easy to use, but for final production, directly using FFmpeg commands might be more efficient (MoviePy ultimately uses ffmpeg under the hood). We can reuse lots of community recipes for adding audio streams, subtitles, etc., using ffmpeg. Also, tools like ffmpeg-python provide a Pythonic way to compose ffmpeg pipelines.
- **Example Projects**: We have inspiration from Thierry Moreau's recipe video pipeline, which chained LLM (Mixtral 8x7B) to JSON formatting, SDXL for images, and Stable Video Diffusion for 4s clips, then MoviePy to stitch. Also Giulio Sistilli's multi-agent video system (used BeautifulSoup, LangChain, OpenCV, MoviePy). We can reuse patterns from these, such as using OpenCV if we need any frame post-processing, or how they managed memory by switching to local models to cut API costs. Both examples reinforce that our chosen tools are viable in practice.
- **GitHub Repos**: We should identify and bookmark repositories for the models we use: e.g. the official Mistral repo (for any prompt format specifics), ModelScope text2video GitHub (which might have sample code), HuggingFace spaces or examples for MusicGen or TTS usage. These will provide code snippets and help debug any model integration issues. If possible, we'll use stable APIs (like huggingface's inference API for some parts during prototyping) and then swap to local to ensure we remain open-source and cost-effective.

Finally, we will document all assumptions and findings. For instance, if we assume Shot durations ~5s, but find out in editing that speech takes 7s, we note to adjust duration dynamically. If a certain open model isn't producing desired quality, we may list a backup (Plan B: use a paid API like ElevenLabs for TTS if open model fails, etc., keeping it optional). The architecture is designed to be robust, extensible, and transparent about these choices, enabling continuous improvement towards the goal of full-length AI-generated cinematic videos.

## Sources

- Moreau, T. (2024). *GenAI video generation pipeline (recipe example)* – Used multiple open models (Mistral 7B, SDXL, Stable Video Diffusion) and stitched results with MoviePy.
- Giulio Sistilli (2025). *Multi-Agent Video Processing System* – Demonstrated a similar multi-agent approach (crawler, scriptwriter, asset creator, etc.) with LangChain and MoviePy on local GPU.
- Reddit AI Agents Community – Discussed using *Groq* for fast open-model inference and *Together.ai* for reasoning (Qwen-14B), and frameworks like LangGraph/AutoGen for orchestration. Also highlighted combining Pydantic for agent I/O with LangGraph for complex workflows.
- Stability AI & Modal Blog – Provided insight into state-of-art open video models (HunyuanVideo, Mochi, Wan2.2) and their resource requirements, and open TTS models (Higgs, Kokoro, Orpheus with multi-lingual support).
- AI4Bharat Indic TTS – Open-sourced models for Hindi and 20 Indic languages (plus English) under Apache-2.0. Suitable for our TTS needs.
- Meta AI (MusicGen) – Open-source music generator (code MIT, weights CC BY-NC) requiring ~16GB GPU, capable of conditioning on text and melody for ~12s clips.

### Reference Links

1. What's the cheapest(good if free) but still useful LLM API in 2025? Also, which one is best for learning agentic AI? : r/AI_Agents
   https://www.reddit.com/r/AI_Agents/comments/1m1ag00/whats_the_cheapestgood_if_free_but_still_useful/

2. Building a Multi-Agent AI System for Video Processing | by Giulio Sistilli | Level Up Coding
   https://levelup.gitconnected.com/building-a-multi-agent-ai-system-for-video-processing-ca629a09e210?gi=87811bb79e83

3. Build your own GenAI video generation pipeline | by Thierry Moreau | Medium
   https://medium.com/@thierryj Moreau/build-your-own-genai-video-generation-pipeline-cdc1515d1db9

4. Top open-source text-to-video AI models
   https://modal.com/blog/text-to-video-ai-article

5. Stable Video — Stability AI
   https://stability.ai/stable-video

6. ai4bharat/indic-parler-tts · Hugging Face
   https://huggingface.co/ai4bharat/indic-parler-tts

7. The Top Open-Source Text to Speech (TTS) Models
   https://modal.com/blog/open-source-tts

8. Meta open sources an AI-powered music generator | TechCrunch
   https://techcrunch.com/2023/06/12/meta-open-sources-an-ai-powered-music-generator/

9. The Most Powerful Way to Build AI Agents: LangGraph + Pydantic AI (Detailed Example) : r/AI_Agents
   https://www.reddit.com/r/AI_Agents/comments/1jorllf/the_most_powerful_way_to_build_ai_agents/

10. Here's how to build durable AI agents with Pydantic and Temporal
    https://temporal.io/blog/build-durable-ai-agents-pydantic-ai-and-temporal

11. AudioCraft - Meta AI
    https://ai.meta.com/resources/models-and-libraries/audiocraft/
