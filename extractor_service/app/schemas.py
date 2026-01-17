"""
This module defines Pydantic models and validators.
Models:
    - ExtractorConfig: Model containing the extractors configuration parameters.
    - Message: Model for encapsulating messages returned by the application.
    - ExtractorStatus: Model representing the status of the working extractor in the system.
LICENSE
=======
Copyright (C) 2024  Bartłomiej Flis

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import logging
import os
from pathlib import Path

from pydantic import BaseModel, DirectoryPath, Field

logger = logging.getLogger(__name__)


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid %s value: %s", name, value)
        return default


class PersonDetectionConfig(BaseModel):
    """
    A Pydantic model containing person detection configuration parameters.

    Attributes:
        enabled (bool): Enable person detection scoring.
        require_faces (bool): Require at least one face to keep a frame.
        min_face_area (float): Minimum face area ratio (0-1) to keep a face.
        min_face_confidence (float): Minimum face confidence (0-1) to keep a face.
        blur_threshold (float): Minimum Laplacian variance for sharpness scoring.
        pose_filter (list[str]): Pose categories to keep; empty means no filtering.
    """
    enabled: bool = False
    require_faces: bool = False
    min_face_area: float = 0.05
    min_face_confidence: float = 0.7
    blur_threshold: float = 100.0
    pose_filter: list[str] = Field(default_factory=list)


class ScoringWeights(BaseModel):
    """
    A Pydantic model containing scoring weights for composite scoring.

    Attributes:
        aesthetic (float): Weight for NIMA aesthetic score.
        face_quality (float): Weight for face quality score.
        sharpness (float): Weight for sharpness score.
    """
    aesthetic: float = 0.5
    face_quality: float = 0.35
    sharpness: float = 0.15


class ExtractorConfig(BaseModel):
    """
    A Pydantic model containing the extractors configuration parameters.

    Attributes:
        input_directory (DirectoryPath): Input directory path containing entries for extraction.
            By default, it sets value for docker container volume.
        output_directory (DirectoryPath): Output directory path for extraction results.
            By default, it sets value for docker container volume.
        video_extensions (tuple[str]): Supported videos' extensions in service for reading videos.
        images_extensions (tuple[str]): Supported images' extensions in service for reading images.
        processed_video_prefix (str): Prefix will be added to processed video after extraction.
        batch_size (int): Maximum number of images processed in a single batch.
        compering_group_size (int): Images group number to compare for finding the best one.
        top_n (int): Optional limit for top frames per video (0 means disabled).
        top_images_percent (float): Percentage threshold to determine the top images.
        images_output_format (str): Format for saving output images, e.g., '.jpg', '.png'.
        target_image_size (tuple[int, int]): Images will be normalized to this size.
        weights_directory (Path | str): Directory path where model weights are stored.
        weights_filename (str): The filename of the model weights file to be loaded.
        weights_repo_url (str): URL to the repository where model weights can be downloaded.
        all_frames (bool): It changes best_frames_extractor -> frames_extractor.
            If Ture best_frames_extractor returns all frames without filtering/evaluation.
        person_detection (PersonDetectionConfig): Settings for optional person detection scoring.
        scoring_weights (ScoringWeights): Weights for composite scoring.
    """
    input_directory: DirectoryPath = Path("/app/input_directory")
    output_directory: DirectoryPath = Path("/app/output_directory")
    video_extensions: tuple[str] = (".mp4", ".mov", ".webm", ".mkv", ".avi")  # add more containers here
    images_extensions: tuple[str] = (".jpg", ".jpeg", ".png", ".webp")  # add more containers here
    processed_video_prefix: str = "frames_extracted_"
    batch_size: int = _get_env_int("PERFECTFRAMEAI_BATCH_SIZE", 100)
    compering_group_size: int = 5
    top_n: int = Field(default=0, ge=0)
    top_images_percent: float = 90.0
    images_output_format: str = ".jpg"
    target_image_size: tuple[int, int] = (224, 224)
    weights_directory: Path | str = Path.home() / ".cache" / "huggingface"
    weights_filename: str = "weights.h5"
    weights_repo_url: str = "https://huggingface.co/BKDDFS/nima_weights/resolve/main/"
    all_frames: bool = False
    person_detection: PersonDetectionConfig = PersonDetectionConfig()
    scoring_weights: ScoringWeights = ScoringWeights()


class Message(BaseModel):
    """
    A Pydantic model for encapsulating messages returned by the application.

    Attributes:
        message (str): The message content.
    """
    message: str


class ExtractorStatus(BaseModel):
    """
    A Pydantic model representing the status of the currently working extractor in the system.

    Attributes:
        active_extractor (str): The name of the currently active extractor.
    """
    active_extractor: str | None
