# Mock Assets Directory

This directory contains placeholder media files used in DEBUG_CPU mode for fast pipeline testing without running actual AI models.

## Contents

- `placeholder.png` - Generated dynamically on first use
- `bg_music.mp3` - Silent or simple loop audio for testing
- Generated mock videos are cached here for reuse

## Purpose

In debug mode, the pipeline generates placeholder assets instead of using expensive AI models:

1. **Mock Videos**: Simple images with text overlay converted to short video clips
2. **Mock Audio**: Silent audio files or simple tones matching expected duration
3. **Mock Music**: Looped background track or silence

These assets enable developers to:
- Test the full pipeline flow on CPU-only machines
- Validate data passing between agents
- Debug orchestration logic without GPU requirements
- Iterate quickly on integration tests

## Caching

Generated mock assets are cached by content hash. Subsequent runs with the same input will reuse cached files, making repeated tests faster.
