"""
Convenience CLI wrapper for running the best frames extractor.
"""
from __future__ import annotations

import argparse
import logging

from config import Config
from service_manager.docker_manager import DockerManager
from service_manager.service_initializer import ServiceInitializer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI wrapper for PerfectFrameAI best frames extraction."
    )
    parser.add_argument("--input", "-i", dest="input_dir",
                        default=Config.input_directory,
                        help="Full path to the input directory.")
    parser.add_argument("--output", "-o", dest="output_dir",
                        default=Config.output_directory,
                        help="Full path to the output directory.")
    parser.add_argument("--port", "-p", type=int, default=Config.port,
                        help="Port to expose the service on the host.")
    parser.add_argument("--build", "-b", action="store_true",
                        help="Force Docker image rebuild.")
    parser.add_argument("--all-frames", action="store_true",
                        help="Return all frames without filtering.")
    parser.add_argument("--person-mode", action="store_true",
                        help="Enable person detection scoring.")
    parser.add_argument("--require-faces", action="store_true",
                        help="Only keep frames with detected faces.")
    parser.add_argument("--min-face-area", type=float, default=0.05,
                        help="Minimum face area ratio (default: 0.05).")
    parser.add_argument("--blur-threshold", type=float, default=100.0,
                        help="Minimum sharpness score (default: 100).")
    parser.add_argument("--pose-filter", type=str, default=None,
                        help="Filter by pose type: portrait,profile,full-body.")
    parser.add_argument("--top-n", type=int, default=0,
                        help="Limit to top N frames per video (default: 0, disabled).")
    parser.add_argument("--save-metadata", action="store_true",
                        help="No-op flag for compatibility with legacy CLI.")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only mode.")
    parser.add_argument("--extractor", default="best_frames_extractor",
                        choices=["best_frames_extractor", "top_images_extractor"],
                        help="Extractor to run (default: best_frames_extractor).")
    args = parser.parse_args()
    args.extractor_name = args.extractor
    return args


def main() -> None:
    user_input = parse_args()
    service = ServiceInitializer(user_input)
    docker = DockerManager(
        Config.service_name,
        user_input.input_dir,
        user_input.output_dir,
        user_input.port,
        user_input.build,
        user_input.cpu
    )
    docker.build_image(Config.dockerfile)
    docker.deploy_container(
        Config.port,
        Config.volume_input_directory,
        Config.volume_output_directory
    )
    service.run_extractor()
    docker.follow_container_logs()
    logger.info("Process stopped.")


if __name__ == "__main__":
    main()
