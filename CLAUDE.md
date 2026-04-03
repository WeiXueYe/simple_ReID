# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a video person Re-identification (ReID) system that automatically extracts and tracks people in videos, providing precise appearance timing information. The system uses YOLOv8 for person detection and deep learning models for feature extraction and person matching across frames.

## Common Commands

### Running the System

**Process a single video:**
```bash
python main.py --single video/test.mp4
```

**Process all videos in default directory:**
```bash
python main.py
```

**Process videos with custom parameters:**
```bash
python main.py -i ./my_videos -o ./results --sample-rate 2 --confidence-threshold 0.6
```

**Use CPU mode (disable GPU):**
```bash
python main.py --no-gpu
```

**Debug mode with detailed logging:**
```bash
python main.py --single video/test.mp4 --log-level DEBUG --log-file debug.log
```

### Testing

**Run basic functionality tests:**
```bash
python simple_test.py
```

**Test video processing pipeline:**
```bash
python test_video_processing.py
```

**Test shot transition detection:**
```bash
python test_shot_transition.py
```

**Run complete solution verification:**
```bash
python verify_solution.py
```

### Environment Setup

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Install GPU-enabled PyTorch (recommended):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Code Architecture

### Core Module Structure

The system follows a modular pipeline architecture:

1. **VideoProcessor** (`src/video_processor.py`) - Handles video I/O and frame extraction
2. **PersonDetector** (`src/person_detector.py`) - YOLOv8-based person detection
3. **FeatureExtractor** (`src/feature_extractor.py`) - Deep feature extraction for ReID
4. **PersonTracker** (`src/person_tracker.py`) - Multi-person tracking with ID management
5. **AppearanceAnalyzer** (`src/appearance_analyzer.py`) - Analyzes and formats appearance timing
6. **MainController** (`src/main_controller.py`) - Orchestrates the entire pipeline

### Key Features

**Shot Transition Handling:**
- Automatic detection of camera cuts/scene changes
- Dynamic similarity threshold adjustment (0.75 normal → 0.68 post-transition)
- Enhanced feature update weights during transitions (15% → 35%)
- Emergency ID merging to resolve split identities

**Advanced Tracking:**
- Cosine similarity-based person matching
- Time continuity scoring to improve matching accuracy
- Trajectory history management (50-frame history)
- Configurable feature update strategies

**Performance Optimizations:**
- GPU acceleration support with CUDA
- Batch processing for feature extraction
- Frame sampling rate control
- Feature caching system

### Configuration System

Main configuration in `src/config.py` includes:

- **Detection**: YOLO model size, confidence thresholds, image sizes
- **Tracking**: Similarity thresholds, missed frame tolerance, feature weights
- **Shot Transitions**: Detection sensitivity, recovery parameters
- **Performance**: GPU usage, batch sizes, cache settings

### Data Flow

```
Video Input → Frame Extraction → Person Detection → Feature Extraction →
Identity Matching → Track Management → Appearance Analysis → JSON Output
```

### Output Format

Results are exported as structured JSON with:
- Video metadata (FPS, duration, frame count)
- Per-person tracking data with appearance time ranges
- Millisecond-precision timing information
- Formatted timecodes for easy reference

## Key Files and Their Purposes

- `main.py` - Command-line interface and program entry point
- `src/config.py` - Centralized configuration management
- `src/main_controller.py` - Pipeline orchestration
- `src/person_tracker.py` - Core tracking logic with shot transition handling
- `src/utils.py` - Helper functions including shot transition detection
- `requirements.txt` - Python dependencies
- `README.md` - User documentation and usage guide
- `SOLUTION_SUMMARY.md` - Technical details of shot transition solutions

## Development Patterns

### Adding New Features
1. Create new module in `src/` directory
2. Update `MainController` to integrate the module
3. Add configuration parameters to `Config` class
4. Update command-line interface in `main.py` if needed

### Configuration Changes
1. Add parameters to `Config` class in `src/config.py`
2. Update validation logic in `validate_config()` method
3. Add CLI parameter support in `main.py` if needed

### Testing Strategy
- Unit tests for individual modules (import and functionality tests)
- Integration tests for complete video processing pipeline
- Shot transition detection verification
- Performance and accuracy validation

## Performance Considerations

- Use GPU acceleration when available for 4-8x speed improvement
- Adjust `FRAME_SAMPLE_RATE` for speed vs. accuracy trade-offs
- Larger YOLO models ('s', 'm', 'l') provide better accuracy but slower processing
- Higher resolution inputs improve detection but increase memory usage
- Feature cache size affects memory usage and matching performance