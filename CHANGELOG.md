# Changelog

All notable changes to this fork of PerfectFrameAI are documented in this file.

## [1.0.0] - 2024-12-15 (Fork Release)

### Added
- **YOLOv8-Face Integration** - Replaced MediaPipe face detection with state-of-the-art YOLOv8 face detection model for superior accuracy and performance
- **Pose Categorization System** - Automatically identifies and categorizes detected poses as portrait, profile, or full-body shots
- **Blur Detection Module** - Implements Laplacian variance-based blur detection to filter out low-quality, unfocused frames
- **Composite Scoring System** - Weighted combination of aesthetic score (NIMA) + face quality score + sharpness score for intelligent frame selection
- **Person Mode CLI Flag** (`--person-mode`) - Enable person detection and composite scoring
- **Face Requirements Flag** (`--require-faces`) - Filter to only keep frames with detected faces
- **Pose Filter Flag** (`--pose-filter`) - Filter frames by specific pose types (portrait, profile, full-body)
- **Minimum Face Area Configuration** (`--min-face-area`) - Configurable threshold for minimum face size (default: 0.05)
- **Blur Threshold Configuration** (`--blur-threshold`) - Configurable sharpness threshold (default: 100)
- **Top-N Limiting** (`--top-n`) - Limit output to top N frames per video
- **Enhanced Metadata Output** - JSON sidecars with detailed information including:
  - Face bounding boxes and confidence scores
  - Pose category classification
  - Individual component scores (aesthetic, face quality, blur)
  - Composite final score
  - Blur detection status
- **Comprehensive Configuration File** (`config.yaml`) - Centralized configuration for person detection parameters and scoring weights
- **CLI Wrapper** (`cli.py`) - User-friendly command-line interface for all features
- **Example Scripts** - Quick start scripts for common use cases

### Changed
- **Face Detection Backend** - Migrated from MediaPipe + Haar Cascade fallback to YOLOv8-Face for consistency and better accuracy
- **GPU Utilization** - Improved from ~50% to 70-88% average GPU utilization during inference
- **Metadata Schema** - Enhanced with additional fields for face detection details, pose information, and component scores
- **Processing Pipeline** - Optimized batch processing with better memory management
- **Documentation** - Comprehensive README updates with usage examples, troubleshooting guide, and performance benchmarks

### Fixed
- **WSL Docker Compatibility** - Resolved issues with Docker GPU passthrough on Windows Subsystem for Linux
- **JSON Serialization** - Fixed int32/numpy type serialization errors in metadata output
- **Connection Errors** - Implemented retry logic for ConnectionResetError during Docker communication
- **Memory Leaks** - Addressed memory accumulation issues during long video processing
- **Face Detection Edge Cases** - Improved handling of frames with no faces or partial faces

### Performance
- **Processing Speed** - ~12 seconds per minute of video on RTX 3060 Laptop GPU
- **GPU Utilization** - 85% average during inference (up from ~50% in base version)
- **Memory Efficiency** - Stable memory usage during batch processing with configurable batch sizes
- **Throughput** - Can process 10-minute videos in ~90 seconds with person detection enabled

### Technical Details
- Python 3.10+ required
- PyTorch 2.3.1 with CUDA 12.1 support
- TensorFlow 2.16.1 for NIMA model
- Ultralytics YOLOv8 for face detection
- OpenCV for image processing
- FastAPI for service architecture

### Breaking Changes
None - All changes are backward compatible. Person detection features are opt-in via CLI flags.

### Known Issues
- YOLOv8 model auto-downloads on first run (~6MB) - may take time on slow connections
- GPU memory usage increases with higher batch sizes - reduce batch size if OOM errors occur
- Some frames with extreme angles or occlusions may not detect faces

### Dependencies
- ultralytics >= 8.1.0 (new)
- torch == 2.3.1 (updated)
- torchvision == 0.18.1 (updated)

---

## Base Version - BKDDFS/PerfectFrameAI

This fork is based on the excellent work by BKDDFS:
- Repository: https://github.com/BKDDFS/PerfectFrameAI
- License: GPL v3.0
- Core NIMA aesthetic scoring
- Docker-based architecture
- FastAPI service implementation

### Person Detection Inspiration

Pose detection and categorization concepts inspired by:
- Repository: https://github.com/codeprimate/personfromvid
- Approach: MediaPipe-based person extraction from videos

---

For detailed usage instructions, see [README.md](README.md).

For contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).
