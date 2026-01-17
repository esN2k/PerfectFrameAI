#!/bin/bash
# Quick start example for PerfectFrameAI Enhanced
# This script demonstrates basic usage with person detection

set -e  # Exit on error

echo "🎬 PerfectFrameAI Enhanced - Quick Start Demo"
echo "=============================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.10+ first."
    exit 1
fi

# Create input directory if it doesn't exist
if [ ! -d "input_directory" ]; then
    echo "📁 Creating input directory..."
    mkdir -p input_directory
fi

# Create output directory if it doesn't exist
if [ ! -d "output_directory" ]; then
    echo "📁 Creating output directory..."
    mkdir -p output_directory
fi

# Check if there are any video files in input
VIDEO_COUNT=$(find input_directory -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" \) 2>/dev/null | wc -l)

if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo "⚠️  No video files found in input_directory/"
    echo ""
    echo "Please add video files to input_directory/ and run this script again."
    echo ""
    echo "Supported formats: .mp4, .avi, .mov, .mkv"
    echo ""
    echo "Example:"
    echo "  cp /path/to/your/video.mp4 input_directory/"
    exit 1
fi

echo "✅ Found $VIDEO_COUNT video file(s) in input_directory/"
echo ""

# Example 1: Basic mode (NIMA-only, no person detection)
echo "Example 1: Basic Mode (NIMA aesthetic scoring only)"
echo "----------------------------------------------------"
echo "Command: python cli.py --input input_directory/ --output output_directory/basic/ --top-n 5"
echo ""
read -p "Run this example? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python cli.py --input input_directory/ --output output_directory/basic/ --top-n 5
    echo "✅ Basic mode complete! Check output_directory/basic/"
    echo ""
fi

# Example 2: Person mode with face requirements
echo "Example 2: Smart Person Mode (with face detection)"
echo "---------------------------------------------------"
echo "Command: python cli.py --input input_directory/ --output output_directory/person_mode/ \\"
echo "         --person-mode --require-faces --blur-threshold 120 --top-n 5"
echo ""
read -p "Run this example? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python cli.py \
        --input input_directory/ \
        --output output_directory/person_mode/ \
        --person-mode \
        --require-faces \
        --blur-threshold 120 \
        --top-n 5
    echo "✅ Person mode complete! Check output_directory/person_mode/"
    echo "📊 Metadata JSON files are saved alongside each image"
    echo ""
fi

# Example 3: Portrait filter mode
echo "Example 3: Portrait Filter Mode (only portrait shots)"
echo "------------------------------------------------------"
echo "Command: python cli.py --input input_directory/ --output output_directory/portraits/ \\"
echo "         --person-mode --pose-filter portrait --min-face-area 0.08 --blur-threshold 150 --top-n 3"
echo ""
read -p "Run this example? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python cli.py \
        --input input_directory/ \
        --output output_directory/portraits/ \
        --person-mode \
        --pose-filter portrait \
        --min-face-area 0.08 \
        --blur-threshold 150 \
        --top-n 3
    echo "✅ Portrait filter mode complete! Check output_directory/portraits/"
    echo ""
fi

echo ""
echo "🎉 Demo complete!"
echo ""
echo "📁 Output directories:"
echo "  - output_directory/basic/       - NIMA-only results"
echo "  - output_directory/person_mode/ - Smart person detection results"
echo "  - output_directory/portraits/   - Portrait-filtered results"
echo ""
echo "💡 Tips:"
echo "  - Check the .json files for detailed frame metadata"
echo "  - Adjust --blur-threshold (default 100) for sharper images"
echo "  - Adjust --min-face-area (default 0.05) to filter small faces"
echo "  - Use --pose-filter to select specific shot types"
echo ""
echo "📖 For more options, run: python cli.py --help"
