"""
This module provides:
    - Extractor: Abstract class for creating extractors.
    - ExtractorFactory: Factory for getting extractors by their names.
    - Extractors:
        - BestFramesExtractor: For extracting best frames from all videos from any directory.
        - TopImagesExtractor: For extracting images with top percent evaluating from any directory.
LICENSE
=======
Copyright (C) 2024  Bartłomiej Flis

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
import gc
import heapq
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread
from dataclasses import dataclass
from pathlib import Path
from typing import Type

import numpy as np

from .dependencies import ExtractorDependencies
from .image_evaluators import ImageEvaluator
from .image_processors import ImageProcessor
from .modules.person_detector import (FaceDetection, calculate_blur_score,
                                      categorize_pose,
                                      compute_face_quality_score_from_faces,
                                      detect_faces)
from .perf import (gpu_log_batch_enabled, log_gpu_stats, log_perf_metric,
                   perf_context, perf_timer)
from .schemas import ExtractorConfig
from .video_processors import VideoProcessor

logger = logging.getLogger(__name__)


@dataclass
class FrameEvaluation:
    """Evaluation details for a single frame."""
    frame: np.ndarray
    aesthetic_score: float
    face_quality_score: float
    blur_score: float
    composite_score: float
    faces: list[FaceDetection]
    pose_category: str
    is_blurry: bool
    frame_id: str | None
    timestamp: float | None
    eligible: bool


class Extractor(ABC):
    """Abstract class for creating extractors."""

    class EmptyInputDirectoryError(Exception):
        """Error appear when extractor can't get any input to extraction."""

    def __init__(self, config: ExtractorConfig,
                 image_processor: Type[ImageProcessor],
                 video_processor: Type[VideoProcessor],
                 image_evaluator_class: Type[ImageEvaluator]) -> None:
        """
        Initializes the manager with the given extractor configuration.

        Args:
            config (ExtractorConfig): A Pydantic model with configuration
                parameters for the extractor.
            image_processor (Type[ImageProcessor]): The class for processing images.
            video_processor (Type[VideoProcessor]): The class for processing videos.
            image_evaluator_class (Type[ImageEvaluator]): The class for evaluating images.
        """
        self._config = config
        self._image_processor = image_processor
        self._video_processor = video_processor
        self._image_evaluator_class = image_evaluator_class
        self._image_evaluator = None

    @abstractmethod
    def process(self) -> None:
        """Abstract main method for extraction process implementation."""

    def _get_image_evaluator(self) -> ImageEvaluator:
        """
        Initializes one of image evaluators (currently NIMA) and
            adds it to extractor instance parameters.

        Returns:
            PyIQA: Image evaluator class instance for evaluating images.
        """
        self._image_evaluator = self._image_evaluator_class(self._config)
        return self._image_evaluator

    @staticmethod
    def _prefetch_batches(generator, max_prefetch: int):
        """Prefetch batches from generator on a background thread."""
        if max_prefetch <= 0:
            yield from generator
            return
        queue: Queue = Queue(maxsize=max_prefetch)
        sentinel = object()
        error_holder: dict[str, Exception | None] = {"exc": None}

        def _producer() -> None:
            try:
                for item in generator:
                    queue.put(item)
            except Exception as exc:  # pragma: no cover - defensive path
                error_holder["exc"] = exc
            finally:
                queue.put(sentinel)

        thread = Thread(target=_producer, daemon=True)
        thread.start()
        while True:
            item = queue.get()
            if item is sentinel:
                break
            yield item
        if error_holder["exc"] is not None:
            raise error_holder["exc"]

    @staticmethod
    def _get_prefetch_batches() -> int:
        """Return number of prefetched batches based on environment tuning."""
        value = os.getenv("PERFECTFRAMEAI_PREFETCH_BATCHES", "1")
        try:
            return max(0, int(value))
        except ValueError:
            logger.warning("Invalid PERFECTFRAMEAI_PREFETCH_BATCHES value: %s", value)
            return 0

    def _list_input_directory_files(self, extensions: tuple[str, ...],
                                    prefix: str | None = None) -> list[Path]:
        """
        List all files with given extensions except files with given filename prefix form
            config input directory.

        Args:
            extensions (tuple): Searched files extensions.
            prefix (str | None): Excluded files filename prefix. Default is None.

        Returns:
            list[Path]: All matching files list.
        """
        directory = self._config.input_directory
        entries = directory.iterdir()
        files = [
            entry for entry in entries
            if entry.is_file()
               and entry.suffix in extensions
               and (prefix is None or not entry.name.startswith(prefix))
        ]
        if not files:
            prefix = prefix if prefix else "Prefix not provided"
            error_massage = (
                f"Files with extensions '{extensions}' and without prefix '{prefix}' "
                f"not found in folder: {directory}."
                f"\n-->HINT: You probably don't have input or you haven't changed prefixes. "
                f"\nCheck input directory."
            )
            logger.error(error_massage)
            raise self.EmptyInputDirectoryError(error_massage)
        logger.info("Directory '%s' files listed.", str(directory))
        logger.debug("Listed file paths: %s", files)
        return files

    def _evaluate_images(self, normalized_images: np.ndarray) -> np.array:
        """
        Rating all images in provided images batch using already initialized image evaluator.

        Args:
            normalized_images (list[np.ndarray]): Already normalized images for evaluating.

        Returns:
            np.array: Array with images scores in given images order.
        """
        scores = np.array(self._image_evaluator.evaluate_images(normalized_images))
        return scores

    def _read_images(self, paths: list[Path]) -> list[np.ndarray]:
        """
        Read all images from given paths synonymously.

        Args:
            paths (list[Path]): List of images paths.

        Returns:
            list[np.ndarray]: List of images in numpy ndarrays.
        """
        with ThreadPoolExecutor() as executor:
            images = []
            futures = [executor.submit(
                self._image_processor.read_image, path,
            ) for path in paths]
            for future in futures:
                image = future.result()
                if image is not None:
                    images.append(image)
            return images

    def _read_images_with_paths(self, paths: list[Path]) -> list[tuple[np.ndarray, Path]]:
        """
        Read all images from given paths and return images with paths.

        Args:
            paths (list[Path]): List of images paths.

        Returns:
            list[tuple[np.ndarray, Path]]: List of images and their paths.
        """
        with ThreadPoolExecutor() as executor:
            images_with_paths = []
            futures = {
                executor.submit(self._image_processor.read_image, path): path
                for path in paths
            }
            for future, path in futures.items():
                image = future.result()
                if image is not None:
                    images_with_paths.append((image, path))
            return images_with_paths

    def _save_images(self, images: list[np.ndarray]) -> list[Path | None]:
        """
        Save all images in config output directory synonymously.

        Args:
            images (list[np.ndarray]): List of images in numpy ndarrays.

        Returns:
            list[Path | None]: List of paths where images were saved.
        """
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._image_processor.save_image, image,
                    self._config.output_directory,
                    self._config.images_output_format
                )
                for image in images
            ]
            saved_paths = [future.result() for future in futures]
            return saved_paths

    def _normalize_images(self, images: list[np.ndarray],
                          target_size: tuple[int, int]) -> np.ndarray:
        """
        Normalize all images in given list to target size for further operations.

        Args:
            images (list[np.ndarray]): List of np.ndarray images to normalize.
            target_size (tuple[int, int]): Images will be normalized to this size.

        Returns:
            np.ndarray: All images as a one numpy array.
        """
        normalized_images = self._image_processor.normalize_images(images, target_size)
        return normalized_images

    def _save_metadata(self, image_paths: list[Path | None], metadata: list[dict]) -> None:
        """
        Save metadata JSON sidecars alongside saved images.

        Args:
            image_paths (list[Path]): Paths where images were saved.
            metadata (list[dict]): Metadata entries for each saved image.
        """
        if len(image_paths) != len(metadata):
            logger.warning(
                "Metadata count (%s) does not match image count (%s).",
                len(metadata),
                len(image_paths),
            )
            return
        for image_path, entry in zip(image_paths, metadata):
            if image_path is None:
                continue
            metadata_path = image_path.with_suffix(".json")
            with metadata_path.open("w", encoding="utf-8") as metadata_file:
                json.dump(entry, metadata_file, indent=2)
            logger.debug("Metadata saved at '%s'.", metadata_path)

    @staticmethod
    def _normalize_sharpness_score(blur_score: float, blur_threshold: float) -> float:
        """Normalize blur score to 0-10 sharpness scale."""
        if blur_threshold <= 0:
            return 0.0
        return min(blur_score / blur_threshold, 1.0) * 10.0

    def _calculate_composite_score(self, aesthetic_score: float,
                                   face_quality_score: float,
                                   sharpness_score: float) -> float:
        """Calculate composite score using configured weights."""
        weights = self._config.scoring_weights
        total = weights.aesthetic + weights.face_quality + weights.sharpness
        if total <= 0:
            return aesthetic_score
        return (
            (aesthetic_score * weights.aesthetic) +
            (face_quality_score * weights.face_quality) +
            (sharpness_score * weights.sharpness)
        ) / total

    def _filter_faces(self, faces: list[FaceDetection],
                      frame_shape: tuple[int, int]) -> list[FaceDetection]:
        """Filter faces based on config thresholds."""
        config = self._config.person_detection
        frame_height, frame_width = frame_shape[:2]
        frame_area = frame_height * frame_width
        if frame_area == 0:
            return []
        filtered = []
        for face in faces:
            if face.confidence < config.min_face_confidence:
                continue
            face_area_ratio = (face.bbox[2] * face.bbox[3]) / float(frame_area)
            if face_area_ratio < config.min_face_area:
                continue
            filtered.append(face)
        return filtered

    @staticmethod
    def _select_primary_face(faces: list[FaceDetection]) -> FaceDetection | None:
        """Select the most prominent face by area."""
        if not faces:
            return None
        return max(faces, key=lambda face: face.area)

    def _get_pose_category(self, faces: list[FaceDetection],
                           frame_shape: tuple[int, int]) -> str:
        """Infer pose category from the most prominent face."""
        primary_face = self._select_primary_face(faces)
        if primary_face is None:
            return "unknown"
        return categorize_pose(primary_face.bbox, frame_shape)

    def _is_frame_eligible(self, faces: list[FaceDetection], pose_category: str) -> bool:
        """Check if frame meets face and pose filtering criteria."""
        config = self._config.person_detection
        if config.require_faces and not faces:
            return False
        pose_filters = [
            pose.strip().lower()
            for pose in config.pose_filter
            if pose.strip()
        ]
        if pose_filters and pose_category.lower() not in pose_filters:
            return False
        return True

    def _evaluate_frames_with_person_detection(
            self,
            frames: list[np.ndarray],
            frame_ids: list[str] | None = None,
            timestamps: list[float | None] | None = None
    ) -> list[FrameEvaluation]:
        """
        Evaluate frames using NIMA plus person-aware scoring.

        Args:
            frames (list[np.ndarray]): Frames to evaluate.
            frame_ids (list[str] | None): Optional frame identifiers.
            timestamps (list[float | None] | None): Optional frame timestamps.

        Returns:
            list[FrameEvaluation]: Evaluated frames with metadata.
        """
        normalized_images = self._normalize_images(frames, self._config.target_image_size)
        aesthetic_scores = self._evaluate_images(normalized_images)
        del normalized_images
        config = self._config.person_detection

        evaluations: list[FrameEvaluation] = []
        face_detection_time = 0.0
        for index, frame in enumerate(frames):
            detect_start = time.perf_counter()
            faces = detect_faces(frame)
            face_detection_time += time.perf_counter() - detect_start
            faces = self._filter_faces(faces, frame.shape)
            blur_score = calculate_blur_score(frame)
            is_blurry = blur_score < config.blur_threshold
            face_quality_score = compute_face_quality_score_from_faces(
                frame,
                faces,
                clarity_normalization=max(config.blur_threshold, 1.0),
            )
            sharpness_score = self._normalize_sharpness_score(
                blur_score,
                config.blur_threshold
            )
            composite_score = self._calculate_composite_score(
                float(aesthetic_scores[index]),
                face_quality_score,
                sharpness_score
            )
            pose_category = self._get_pose_category(faces, frame.shape)
            eligible = self._is_frame_eligible(faces, pose_category)
            frame_id = frame_ids[index] if frame_ids and index < len(frame_ids) else None
            timestamp = timestamps[index] if timestamps and index < len(timestamps) else None
            evaluations.append(
                FrameEvaluation(
                    frame=frame,
                    aesthetic_score=float(aesthetic_scores[index]),
                    face_quality_score=face_quality_score,
                    blur_score=blur_score,
                    composite_score=composite_score,
                    faces=faces,
                    pose_category=pose_category,
                    is_blurry=is_blurry,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    eligible=eligible,
                )
            )
        log_perf_metric("face_detection", face_detection_time, f"frames={len(frames)}")
        return evaluations

    @staticmethod
    def _build_metadata_entry(evaluation: FrameEvaluation) -> dict:
        """Build metadata JSON entry for a frame."""
        face_bboxes = [
            [int(face.bbox[0]), int(face.bbox[1]), int(face.bbox[2]), int(face.bbox[3])]
            for face in evaluation.faces
        ]
        timestamp = float(evaluation.timestamp) if evaluation.timestamp is not None else None
        return {
            "frame_id": evaluation.frame_id or "unknown",
            "timestamp": timestamp,
            "aesthetic_score": float(round(evaluation.aesthetic_score, 3)),
            "face_quality_score": float(round(evaluation.face_quality_score, 3)),
            "blur_score": float(round(evaluation.blur_score, 3)),
            "composite_score": float(round(evaluation.composite_score, 3)),
            "faces_detected": int(len(evaluation.faces)),
            "face_bounding_boxes": face_bboxes,
            "pose_category": evaluation.pose_category,
            "is_blurry": bool(evaluation.is_blurry),
        }

    @staticmethod
    def _add_prefix(prefix: str, path: Path) -> Path:
        """
        Adds prefix to file filename.
        
        Args:
            prefix (str): Prefix that will be added.
            path (Path): Path to file that filename will be changed.

        Returns:
            Path: Path of the file with new filename.
        """
        new_path = path.parent / f"{prefix}{path.name}"
        path.rename(new_path)
        logger.debug("Prefix '%s' added to file '%s'. New path: %s",
                     prefix, path, new_path)
        return new_path

    @staticmethod
    def _signal_readiness_for_shutdown() -> None:
        """
        Contains the logic for sending a signal externally that the service has completed
        the process and can be safely shut down.
        """
        logger.info("Service ready for shutdown")


class ExtractorFactory:
    """Extractor factory for getting extractors class by their names."""

    @staticmethod
    def create_extractor(extractor_name: str, config: ExtractorConfig,
                         dependencies: ExtractorDependencies) -> Extractor:
        """
        Match extractor class by its name and return its class.

        Args:
            extractor_name (str): Name of the extractor.
            config (ExtractorConfig): A Pydantic model with extractor configuration.
            dependencies(ExtractorDependencies): Dependencies that will be used in extractor.

        Returns:
            Extractor: Chosen extractor class.
        """
        match extractor_name:
            case "best_frames_extractor":
                return BestFramesExtractor(config, dependencies.image_processor,
                                           dependencies.video_processor, dependencies.evaluator)
            case "top_images_extractor":
                return TopImagesExtractor(config, dependencies.image_processor,
                                          dependencies.video_processor, dependencies.evaluator)
            case _:
                error_massage = f"Provided unknown extractor name: {extractor_name}"
                logger.error(error_massage)
                raise ValueError(error_massage)


class BestFramesExtractor(Extractor):
    """Extractor for extracting best frames from videos in any input directory."""

    def process(self) -> None:
        """
        Rate all videos in given config input directory and
        extract best visually frames from every video.
        """
        logger.info("Starting frames extraction process from '%s'.",
                    self._config.input_directory)
        videos_paths = self._list_input_directory_files(self._config.video_extensions,
                                                        self._config.processed_video_prefix)
        if self._config.all_frames is False:  # evaluator won't be used if all frames
            self._get_image_evaluator()
        for video_path in videos_paths:
            self._extract_best_frames(video_path)
            self._add_prefix(self._config.processed_video_prefix, video_path)
            logger.info("Frames extraction has finished for video: %s", video_path)
        logger.info("Extraction process finished. All frames extracted.")
        self._signal_readiness_for_shutdown()

    def _extract_best_frames(self, video_path: Path) -> None:
        """
        Extract best visually frames from given video.

        Args:
            video_path (Path): Path of the video that will be extracted.
        """
        with perf_timer("video_total", f"video={video_path.name}"):
            with perf_context(video=video_path.name):
                log_gpu_stats("before_video", f"video={video_path.name}")
                if self._config.person_detection.enabled and not self._config.all_frames:
                    frames_batch_generator = self._video_processor.get_next_frames_with_metadata(
                        video_path, self._config.batch_size
                    )
                    frames_batch_generator = self._prefetch_batches(
                        frames_batch_generator, self._get_prefetch_batches()
                    )
                    top_n = max(0, int(self._config.top_n))
                    if top_n > 0:
                        heap: list[tuple[float, int, FrameEvaluation]] = []
                        counter = 0
                        batch_index = 0
                        for frames_with_metadata in frames_batch_generator:
                            if not frames_with_metadata:
                                continue
                            logger.debug("Frames batch generated.")
                            frames = [frame.image for frame in frames_with_metadata]
                            frame_ids = [
                                f"frame_{int(round(frame.timestamp)):04d}"
                                for frame in frames_with_metadata
                            ]
                            timestamps = [frame.timestamp for frame in frames_with_metadata]
                            evaluations = self._evaluate_frames_with_person_detection(
                                frames,
                                frame_ids=frame_ids,
                                timestamps=timestamps
                            )
                            for evaluation in evaluations:
                                if not evaluation.eligible:
                                    continue
                                entry = (evaluation.composite_score, counter, evaluation)
                                if len(heap) < top_n:
                                    heapq.heappush(heap, entry)
                                elif entry[0] > heap[0][0]:
                                    heapq.heapreplace(heap, entry)
                                counter += 1
                            if batch_index == 0 or gpu_log_batch_enabled():
                                log_gpu_stats("during_batch",
                                              f"video={video_path.name} batch={batch_index}")
                            batch_index += 1
                            del frames_with_metadata, frames, evaluations
                            gc.collect()
                        if heap:
                            selected = [
                                entry[2]
                                for entry in sorted(heap, key=lambda item: item[0], reverse=True)
                            ]
                            saved_paths = self._save_images(
                                [evaluation.frame for evaluation in selected]
                            )
                            metadata = [
                                self._build_metadata_entry(evaluation)
                                for evaluation in selected
                            ]
                            self._save_metadata(saved_paths, metadata)
                    else:
                        batch_index = 0
                        for frames_with_metadata in frames_batch_generator:
                            if not frames_with_metadata:
                                continue
                            logger.debug("Frames batch generated.")
                            frames = [frame.image for frame in frames_with_metadata]
                            frame_ids = [
                                f"frame_{int(round(frame.timestamp)):04d}"
                                for frame in frames_with_metadata
                            ]
                            timestamps = [frame.timestamp for frame in frames_with_metadata]
                            evaluations = self._evaluate_frames_with_person_detection(
                                frames,
                                frame_ids=frame_ids,
                                timestamps=timestamps
                            )
                            selected = self._get_best_frames_with_person_detection(evaluations)
                            saved_paths = self._save_images(
                                [evaluation.frame for evaluation in selected]
                            )
                            metadata = [
                                self._build_metadata_entry(evaluation)
                                for evaluation in selected
                            ]
                            self._save_metadata(saved_paths, metadata)
                            if batch_index == 0 or gpu_log_batch_enabled():
                                log_gpu_stats("during_batch",
                                              f"video={video_path.name} batch={batch_index}")
                            batch_index += 1
                            del frames_with_metadata, frames, evaluations, selected
                            gc.collect()
                else:
                    frames_batch_generator = self._video_processor.get_next_frames(
                        video_path, self._config.batch_size
                    )
                    frames_batch_generator = self._prefetch_batches(
                        frames_batch_generator, self._get_prefetch_batches()
                    )
                    top_n = max(0, int(self._config.top_n))
                    if top_n > 0 and not self._config.all_frames:
                        heap: list[tuple[float, int, np.ndarray]] = []
                        counter = 0
                        batch_index = 0
                        for frames in frames_batch_generator:
                            if not frames:
                                continue
                            logger.debug("Frames batch generated.")
                            normalized_images = self._normalize_images(
                                frames, self._config.target_image_size
                            )
                            scores = self._evaluate_images(normalized_images)
                            del normalized_images
                            for frame, score in zip(frames, scores):
                                entry = (float(score), counter, frame)
                                if len(heap) < top_n:
                                    heapq.heappush(heap, entry)
                                elif entry[0] > heap[0][0]:
                                    heapq.heapreplace(heap, entry)
                                counter += 1
                            if batch_index == 0 or gpu_log_batch_enabled():
                                log_gpu_stats("during_batch",
                                              f"video={video_path.name} batch={batch_index}")
                            batch_index += 1
                            del frames, scores
                            gc.collect()
                        if heap:
                            selected_frames = [
                                entry[2]
                                for entry in sorted(heap, key=lambda item: item[0], reverse=True)
                            ]
                            self._save_images(selected_frames)
                    else:
                        batch_index = 0
                        for frames in frames_batch_generator:
                            if not frames:
                                continue
                            logger.debug("Frames batch generated.")
                            if not self._config.all_frames:
                                frames = self._get_best_frames(frames)
                            self._save_images(frames)
                            if batch_index == 0 or gpu_log_batch_enabled():
                                log_gpu_stats("during_batch",
                                              f"video={video_path.name} batch={batch_index}")
                            batch_index += 1
                            del frames
                            gc.collect()
                log_gpu_stats("after_video", f"video={video_path.name}")

    def _get_best_frames(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        Splits images batch for comparing groups and select best image for each group.

        Args:
            frames (list[np.ndarray]): Batch of images in numpy ndarray.

        Returns:
            list[np.ndarray]: Best images list.
        """
        normalized_images = self._normalize_images(frames, self._config.target_image_size)
        scores = self._evaluate_images(normalized_images)
        del normalized_images

        best_frames = []
        group_size = self._config.compering_group_size
        groups = np.array_split(scores, np.arange(group_size, len(scores), group_size))
        for index, group in enumerate(groups):
            best_index = np.argmax(group)
            global_index = index * group_size + best_index
            best_frames.append(frames[global_index])
        logger.info("Best frames selected(%s).", len(best_frames))
        return best_frames

    def _get_best_frames_with_person_detection(
            self, evaluations: list[FrameEvaluation]
    ) -> list[FrameEvaluation]:
        """
        Select best frames using composite scores and optional face filters.

        Args:
            evaluations (list[FrameEvaluation]): Evaluated frames with metadata.

        Returns:
            list[FrameEvaluation]: Best frames from each group.
        """
        if not evaluations:
            return []
        group_size = self._config.compering_group_size
        scores = np.array([
            evaluation.composite_score if evaluation.eligible else -np.inf
            for evaluation in evaluations
        ])
        groups = np.array_split(scores, np.arange(group_size, len(scores), group_size))
        best_frames = []
        for index, group in enumerate(groups):
            if np.all(np.isneginf(group)):
                continue
            best_index = int(np.argmax(group))
            global_index = index * group_size + best_index
            best_frames.append(evaluations[global_index])
        logger.info("Best frames selected(%s).", len(best_frames))
        return best_frames


class TopImagesExtractor(Extractor):
    """Images extractor for extracting top percent of images in config input directory."""

    def process(self) -> None:
        """
        Rate all images in given config input directory and
        extract images that are in top percent of images visually.
        """
        images_paths = self._list_input_directory_files(self._config.images_extensions)
        self._get_image_evaluator()
        for batch_index in range(0, len(images_paths), self._config.batch_size):
            batch = images_paths[batch_index:batch_index + self._config.batch_size]
            if self._config.person_detection.enabled:
                images_with_paths = self._read_images_with_paths(batch)
                if not images_with_paths:
                    continue
                images = [image for image, _ in images_with_paths]
                frame_ids = [path.stem for _, path in images_with_paths]
                evaluations = self._evaluate_frames_with_person_detection(
                    images, frame_ids=frame_ids
                )
                eligible = [evaluation for evaluation in evaluations if evaluation.eligible]
                if not eligible:
                    logger.info("No eligible images found in batch.")
                    continue
                scores = np.array([evaluation.composite_score for evaluation in eligible])
                threshold = np.percentile(scores, self._config.top_images_percent)
                top_images = [
                    evaluation for evaluation in eligible
                    if evaluation.composite_score >= threshold
                ]
                saved_paths = self._save_images([evaluation.frame for evaluation in top_images])
                metadata = [self._build_metadata_entry(evaluation) for evaluation in top_images]
                self._save_metadata(saved_paths, metadata)
            else:
                images = self._read_images(batch)
                normalized_images = self._normalize_images(images, self._config.target_image_size)
                scores = self._evaluate_images(normalized_images)
                top_images = self._get_top_percent_images(images, scores,
                                                          self._config.top_images_percent)
                self._save_images(top_images)
        logger.info("Extraction process finished. All top images extracted from directory: %s.",
                    self._config.input_directory)
        self._signal_readiness_for_shutdown()

    @staticmethod
    def _get_top_percent_images(images: list[np.ndarray], scores: np.array,
                                top_percent: float) -> list[np.ndarray]:
        """
        Returns images that have scores in the top percent of all scores.

        Args:
            images (list[np.ndarray]): Batch of images in numpy ndarray.
            scores (np.array): Array with images scores with images batch order.
            top_percent (float): The top percentage of scores to include (e.g. 80 for top 80%).

        Returns:
            list[np.ndarray]: Top images from given images batch.
        """
        threshold = np.percentile(scores, top_percent)
        top_images = [img for img, score in zip(images, scores) if score >= threshold]
        logger.info("Top images selected(%s).", len(top_images))
        return top_images
