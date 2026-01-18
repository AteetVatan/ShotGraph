# How to Debug ShotGraph

This guide provides step-by-step instructions for debugging the ShotGraph video generation pipeline using the built-in debug script.

## Overview

The debug script (`app/main_debug.py`) allows you to:
1. Start the FastAPI server in the background
2. Directly test the video generation endpoint with sample data
3. Monitor the pipeline execution without making HTTP requests
4. Debug issues in a controlled environment

## Prerequisites

Before starting the debug process, ensure you have:

1. **Environment Setup**
   - Python 3.10+ installed
   - Virtual environment activated (if using one)
   - Dependencies installed (choose one option):
     
     **Option A: Minimal Dependencies (Recommended for quick debugging)**
     ```bash
     pip install -r requirements-dev.txt
     ```
     - Lightweight (~100MB)
     - No ML libraries required
     - Perfect for testing pipeline flow
     
     **Option B: Full Dependencies with CPU-only PyTorch (For debugging with all libraries)**
     ```bash
     # Install CPU-only PyTorch first (required before other dependencies)
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
     
     # Then install remaining dependencies
     pip install -r requirements-dev-full.txt
     ```
     - All dependencies including ML libraries
     - CPU-only PyTorch (smaller, no CUDA)
     - Suitable for 16GB RAM systems
     - Use when you want to test with all dependencies but without GPU
     
     **Option C: Full Dependencies with GPU PyTorch (For production)**
     ```bash
     pip install -r requirements.txt
     ```
     - Complete production dependencies
     - Requires GPU and CUDA
     - Use with `EXECUTION_PROFILE=prod_gpu`
   
   - Environment variables configured (see `env.example`)

2. **Configuration Check**
   - Verify your `.env` file is properly configured
   - Check that `EXECUTION_PROFILE` is set appropriately:
     - `debug_cpu` - Uses mock services (fast, no GPU required)
     - `prod_gpu` - Uses real AI models (requires GPU)
   - Ensure API keys are set if using external services (Together.ai, Groq, etc.)

## Step-by-Step Debug Process

### Step 1: Navigate to Project Root

Open your terminal and navigate to the ShotGraph project root directory:

```bash
cd /path/to/ShotGraph
```

### Step 2: Activate Virtual Environment (if applicable)

If you're using a virtual environment, activate it:

```bash
# On Windows
.\venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

Choose the appropriate installation method based on your needs:

**For Quick Debugging (Minimal Dependencies):**
```bash
pip install -r requirements-dev.txt
```

**For Full Dependencies with CPU-only PyTorch (16GB RAM systems):**
```bash
# Install CPU-only PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Then install remaining dependencies
pip install -r requirements-dev-full.txt
```

**Note:** The CPU-only PyTorch installation must be done BEFORE installing other requirements to avoid conflicts.

### Step 4: Verify Environment Configuration

Check that your environment variables are loaded correctly. You can verify by checking the settings:

```bash
python -c "from config.settings import Settings; s = Settings(); print(f'Profile: {s.execution_profile.value}')"
```

### Step 5: Run the Debug Script

Execute the debug script using one of these methods:

**Method 1: Run as Python module (Recommended)**
```bash
python -m app.main_debug
```

**Method 2: Run directly**
```bash
python app/main_debug.py
```

### Step 6: Observe the Output

The script will perform the following actions in sequence:

1. **Server Startup**
   ```
   [Step 1] Starting FastAPI server in background...
   Server starting in background thread...
   Server will be available at: http://localhost:8000
   API docs will be available at: http://localhost:8000/docs
   ```

2. **Initialization Wait**
   ```
   Waiting for server to initialize...
   ```
   (Waits 2 seconds for server to fully start)

3. **Test Request Creation**
   ```
   [Step 2] Testing generate_video function directly...
   ============================================================
   Testing generate_video endpoint directly
   ============================================================
   Created GenerateRequest:
     Title: The Hero's Journey
     Language: en
     Story length: 345 characters
     API Key Auth: disabled
   ```

4. **Function Call**
   ```
   Calling generate_video function...
   ```

5. **Response Display**
   ```
   ============================================================
   Response received:
     Job ID: abc123def456...
     Status: pending
     Message: Video generation started. Use /status/{job_id} to track progress.
   ============================================================
   ```

6. **Background Task Execution**
   ```
   Executing background tasks...
   Test completed successfully!
   ```

### Step 7: Monitor Job Progress

After the debug script completes, you can monitor the job progress in several ways:

**Option A: Check Status via API**
```bash
# Get job status (replace JOB_ID with actual job ID from output)
curl http://localhost:8000/status/JOB_ID
```

**Option B: Use the Interactive API Docs**
1. Open your browser and navigate to: `http://localhost:8000/docs`
2. Use the `/status/{job_id}` endpoint to check job progress
3. Use the `/jobs` endpoint to list all jobs

**Option C: Check Logs**
Monitor the console output for detailed logging from the pipeline execution.

### Step 8: Stop the Server

When you're done debugging, stop the server by pressing:

```
Ctrl+C
```

The script will gracefully shut down and display:
```
Shutting down...
```

## Customizing Test Data

You can modify the test data in `app/main_debug.py` to test different scenarios:

### Changing the Story

Edit the `SAMPLE_STORY` constant:

```python
SAMPLE_STORY = (
    "Your custom story text here. "
    "Make sure it's at least 50 characters long "
    "and contains at least 10 words to pass validation."
)
```

**Validation Requirements:**
- Minimum 50 characters
- Minimum 10 words
- Maximum 100,000 characters
- Maximum 500KB in size

### Changing the Title

Edit the `SAMPLE_TITLE` constant:

```python
SAMPLE_TITLE = "Your Custom Title"
```

### Changing the Language

Edit the `SAMPLE_LANGUAGE` constant:

```python
from core.models import Language

SAMPLE_LANGUAGE = Language.ENGLISH  # or Language.HINDI
```

## Debugging Different Scenarios

### Scenario 1: Testing with Mock Services (Fast)

Set in your `.env` file:
```env
EXECUTION_PROFILE=debug_cpu
```

This uses mock services that return quickly without actual AI processing. Useful for:
- Testing the pipeline flow
- Debugging orchestration logic
- Verifying data structures
- Fast iteration during development

### Scenario 2: Testing with Real AI Models (Slow)

Set in your `.env` file:
```env
EXECUTION_PROFILE=prod_gpu
```

This uses real AI models. Useful for:
- End-to-end testing
- Performance profiling
- Verifying model integrations
- Testing with actual video generation

**Note:** Requires GPU and proper API keys configured.

### Scenario 3: Testing API Key Authentication

1. Enable API key in `.env`:
   ```env
   API_KEY_ENABLED=true
   API_KEY=your-secret-key-here
   ```

2. The debug script will automatically use the configured API key.

### Scenario 4: Testing Different Story Lengths

Modify `SAMPLE_STORY` to test edge cases:

**Short Story (Minimum Valid):**
```python
SAMPLE_STORY = "This is a short story with exactly ten words to test minimum validation requirements."
```

**Long Story (Test Performance):**
```python
SAMPLE_STORY = "Your very long story here..." * 100
```

## Common Issues and Solutions

### Issue 1: Import Errors

**Error:** `ModuleNotFoundError: No module named 'app'`

**Solution:**
- Ensure you're running from the project root directory
- Use `python -m app.main_debug` instead of `python app/main_debug.py`
- Verify your `PYTHONPATH` includes the project root

### Issue 2: Server Won't Start

**Error:** `Address already in use`

**Solution:**
- Check if port 8000 is already in use: `netstat -ano | findstr :8000` (Windows) or `lsof -i :8000` (Linux/Mac)
- Kill the process using the port or change the port in `run_server()` function

### Issue 3: Pipeline Initialization Fails

**Error:** `Pipeline initialization failed`

**Solution:**
- Check your `.env` file configuration
- Verify all required environment variables are set
- Check that API keys are valid (if using external services)
- Review the error logs for specific failure points

### Issue 4: Background Tasks Not Executing

**Symptom:** Job stays in "pending" status

**Solution:**
- Check that `await background_tasks()` is being called
- Verify the pipeline's `execute()` method is working
- Check logs for exceptions in background task execution
- Ensure the pipeline instance is properly initialized

### Issue 5: API Key Authentication Errors

**Error:** `Invalid API key` or `API key is required`

**Solution:**
- Check `API_KEY_ENABLED` setting in `.env`
- Verify `API_KEY` matches your configuration
- The debug script should handle this automatically, but verify `get_api_key_for_debug()` is working

## Advanced Debugging

### Adding Breakpoints

You can add breakpoints in the debug script or in the main pipeline code:

```python
import pdb; pdb.set_trace()  # Python debugger
```

Or use your IDE's debugger:
1. Set breakpoints in `app/main_debug.py` or `app/main.py`
2. Run the script in debug mode from your IDE
3. Step through the code execution

### Enabling Verbose Logging

Modify the logging level in `app/main_debug.py`:

```python
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
```

### Inspecting Pipeline State

Add inspection code after pipeline initialization:

```python
pipeline = get_pipeline()
print(f"Pipeline settings: {pipeline.settings}")
print(f"Pipeline agents: {pipeline.scene_splitter}, {pipeline.shot_planner}, ...")
```

### Testing Individual Components

You can test individual pipeline components separately:

```python
from app.dependencies import get_pipeline

pipeline = get_pipeline()
# Test scene splitting
scenes = await pipeline.scene_splitter.run(story_input)
print(f"Scenes: {scenes}")
```

## Best Practices

1. **Start with Mock Services**: Always test with `debug_cpu` profile first to verify the flow works
2. **Check Logs**: Monitor console output for detailed error messages
3. **Verify Environment**: Double-check your `.env` configuration before debugging
4. **Test Incrementally**: Test one component at a time when debugging complex issues
5. **Use API Docs**: Leverage the interactive API docs at `/docs` for manual testing
6. **Keep Test Data Simple**: Start with simple test stories before testing complex scenarios

## Next Steps

After successful debugging:

1. **Review Generated Output**: Check the generated video files in your storage path
2. **Analyze Performance**: Review execution times and resource usage
3. **Test Edge Cases**: Test with various story lengths and content types
4. **Integration Testing**: Test the full pipeline with real-world stories
5. **Production Deployment**: Once verified, deploy to production environment

## Additional Resources

- **API Documentation**: `http://localhost:8000/docs` (when server is running)
- **Code Implementation**: See `docs/code_implementation.md`
- **Research Notes**: See `docs/research.md`
- **Environment Configuration**: See `env.example`

## Troubleshooting Checklist

Before reporting issues, verify:

- [ ] Python version is 3.10+
- [ ] All dependencies are installed
- [ ] `.env` file is properly configured
- [ ] Virtual environment is activated (if using one)
- [ ] Running from project root directory
- [ ] Port 8000 is available
- [ ] API keys are valid (if using external services)
- [ ] GPU is available (if using `prod_gpu` profile)
- [ ] Logs show no obvious errors
- [ ] Test data meets validation requirements

---

For additional help, check the project's main README or open an issue on the repository.
