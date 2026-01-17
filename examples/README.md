# PerfectFrameAI Examples

This directory contains examples and documentation to help you get started with PerfectFrameAI Enhanced.

## Contents

### 📜 Scripts

- **`quick_start.sh`** - Interactive demo script showing different usage modes
  - Basic NIMA-only mode
  - Smart person detection mode
  - Portrait filter mode
  
  **Usage:**
  ```bash
  chmod +x examples/quick_start.sh
  ./examples/quick_start.sh
  ```

### 📊 Documentation

- **`before_after_comparison.md`** - Detailed comparison between NIMA-only and person detection modes
  - Performance metrics
  - Use case recommendations
  - Feature comparison table
  
- **`metadata_example.json`** - Annotated example of JSON metadata output
  - All scoring fields explained
  - Face detection data structure
  - Pose categorization format

### 🖼️ Sample Output

The `sample_output/` directory is reserved for example extracted frames and their metadata. After running the quick start script, you'll find example outputs here showing:
- High-quality extracted frames
- Corresponding JSON metadata files
- Examples of different pose categories

## Quick Start

1. **Add video files** to the `input_directory/` folder in the project root
2. **Run the quick start script:**
   ```bash
   ./examples/quick_start.sh
   ```
3. **Check the output** in `output_directory/`

## Common Usage Patterns

### Extract Best Frames (Basic)
```bash
python cli.py --input videos/ --output frames/ --top-n 10
```

### Smart Person Detection
```bash
python cli.py \
  --input videos/ \
  --output frames/ \
  --person-mode \
  --require-faces \
  --blur-threshold 120 \
  --top-n 5
```

### Portrait Photography Mode
```bash
python cli.py \
  --input videos/ \
  --output frames/ \
  --person-mode \
  --pose-filter portrait \
  --min-face-area 0.08 \
  --blur-threshold 150 \
  --top-n 3
```

### Profile Shots Only
```bash
python cli.py \
  --input videos/ \
  --output frames/ \
  --person-mode \
  --pose-filter profile \
  --min-face-area 0.06 \
  --top-n 5
```

## Understanding the Output

### Image Files
- Format: JPEG (`.jpg`)
- Naming: `{video_name}_frame_{number}.jpg`
- Location: Specified `--output` directory

### Metadata Files (Person Mode Only)
- Format: JSON (`.json`)
- Naming: Same as image file with `.json` extension
- Contains: Scores, face data, pose info, timestamps

### Example Output Structure
```
output_directory/
├── family_video_frame_0042.jpg
├── family_video_frame_0042.json
├── family_video_frame_0087.jpg
├── family_video_frame_0087.json
└── ...
```

## Tips & Tricks

### Getting More Frames
- Increase `--top-n` value
- Lower `--blur-threshold` (more lenient)
- Lower `--min-face-area` (smaller faces accepted)

### Getting Higher Quality
- Increase `--blur-threshold` (e.g., 150-200)
- Increase `--min-face-area` (e.g., 0.08-0.10)
- Use `--pose-filter portrait` for close-ups

### Troubleshooting

**No frames extracted?**
- Check that input directory contains video files
- Lower `--min-face-area` to 0.02
- Remove `--require-faces` flag
- Check video file format is supported (mp4, avi, mov, mkv)

**Too many blurry frames?**
- Increase `--blur-threshold` to 150 or 200
- Videos may be inherently low quality

**Processing too slow?**
- Use `--cpu` flag if GPU issues occur
- Reduce batch size in `config.yaml`
- Process shorter videos first

## More Information

- See main [README.md](../README.md) for full documentation
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for development setup
- See [CHANGELOG.md](../CHANGELOG.md) for version history
