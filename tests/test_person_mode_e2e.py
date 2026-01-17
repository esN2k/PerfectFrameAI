#!/usr/bin/env python3
"""
End-to-end person-mode evaluation runner.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "cli.py"
RESULTS_DIR = ROOT / "results"
INPUT_DIR = ROOT / "input_cansu"
SOURCE_VIDEO = ROOT / "tests" / "test_files" / "Cansu.mp4"
DOCKER_EXE = "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


@dataclass
class ScenarioResult:
    name: str
    output_dir: Path
    frames_extracted: int
    avg_composite: float | None
    avg_aesthetic: float | None
    avg_face_quality: float | None
    blur_rejection: int | None
    face_success_rate: float | None
    total_time: float
    per_video_times: dict[str, float]
    gpu_avg_util: float | None
    gpu_peak_mem: float | None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _reset_input_video() -> None:
    _ensure_dir(INPUT_DIR)
    if not SOURCE_VIDEO.exists():
        raise FileNotFoundError(f"Missing source video: {SOURCE_VIDEO}")
    shutil.copy2(SOURCE_VIDEO, INPUT_DIR / SOURCE_VIDEO.name)


def _start_gpu_logger(target: Path) -> subprocess.Popen:
    handle = target.open("w", encoding="utf-8")
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used",
        "--format=csv",
        "-l",
        "1",
    ]
    return subprocess.Popen(cmd, stdout=handle, stderr=subprocess.DEVNULL)


def _stop_gpu_logger(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _run_cli(args: list[str], env: dict[str, str]) -> float:
    start = time.perf_counter()
    subprocess.run([sys.executable, str(CLI)] + args, check=True, env=env)
    return time.perf_counter() - start


def _count_images(output_dir: Path) -> int:
    return sum(1 for path in output_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def _load_metadata(output_dir: Path) -> list[dict[str, Any]]:
    entries = []
    for path in output_dir.glob("*.json"):
        if path.name == "performance.json":
            continue
        with path.open("r", encoding="utf-8") as handle:
            entries.append(json.load(handle))
    return entries


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _parse_perf(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("videos", {})


def _parse_gpu_stats(path: Path) -> tuple[float | None, float | None]:
    if not path.exists():
        return None, None
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if len(rows) <= 1:
        return None, None
    values = []
    mem_values = []
    for row in rows[1:]:
        if len(row) < 3:
            continue
        try:
            values.append(float(row[0].strip().split()[0]))
            mem_values.append(float(row[2].strip().split()[0]))
        except ValueError:
            continue
    if not values:
        return None, None
    return sum(values) / len(values), max(mem_values) if mem_values else None


def _write_comparison(results: list[ScenarioResult]) -> None:
    output_path = RESULTS_DIR / "comparison.md"
    lines = [
        "# Scenario Comparison",
        "",
        "| Scenario | Frames | Avg Composite | Avg Aesthetic | Face Success | Blur Rejections | Total Time (s) | GPU Avg Util % | GPU Peak Mem (MiB) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            "| {name} | {frames} | {comp} | {aes} | {face} | {blur} | {time:.2f} | {gpu} | {mem} |".format(
                name=result.name,
                frames=result.frames_extracted,
                comp=f"{result.avg_composite:.3f}" if result.avg_composite is not None else "n/a",
                aes=f"{result.avg_aesthetic:.3f}" if result.avg_aesthetic is not None else "n/a",
                face=(f"{result.face_success_rate:.2%}"
                      if result.face_success_rate is not None else "n/a"),
                blur=result.blur_rejection if result.blur_rejection is not None else "n/a",
                time=result.total_time,
                gpu=(f"{result.gpu_avg_util:.1f}"
                     if result.gpu_avg_util is not None else "n/a"),
                mem=(f"{result.gpu_peak_mem:.0f}"
                     if result.gpu_peak_mem is not None else "n/a"),
            )
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_quality_grid(output_dir: Path) -> None:
    entries: list[tuple[Path, dict[str, Any]]] = []
    for json_path in output_dir.glob("*.json"):
        if json_path.name == "performance.json":
            continue
        with json_path.open("r", encoding="utf-8") as handle:
            entries.append((json_path, json.load(handle)))
    sorted_items = sorted(entries, key=lambda item: item[1].get("composite_score", 0), reverse=True)
    rows = []
    for json_path, item in sorted_items[:12]:
        frame_id = item.get("frame_id", "unknown")
        image_path = None
        for ext in IMAGE_EXTS:
            candidate = json_path.with_suffix(ext)
            if candidate.exists():
                image_path = candidate.name
                break
        if image_path is None:
            continue
        rows.append(
            f"<figure><img src=\"portrait_hq/{image_path}\" />"
            f"<figcaption>{frame_id} | {item.get('composite_score', 0):.2f}</figcaption></figure>"
        )
    html = "\n".join([
        "<!doctype html>",
        "<html><head><meta charset=\"utf-8\">",
        "<style>",
        "body{font-family:Arial,sans-serif;background:#f6f6f6;margin:0;padding:24px;}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;}",
        "figure{background:#fff;border:1px solid #ddd;padding:8px;margin:0;}",
        "img{width:100%;height:auto;display:block;}",
        "figcaption{font-size:12px;margin-top:6px;color:#333;}",
        "</style></head><body>",
        "<h1>Portrait HQ Grid</h1>",
        "<div class=\"grid\">",
        *rows,
        "</div></body></html>"
    ])
    (RESULTS_DIR / "quality_grid.html").write_text(html, encoding="utf-8")


def _validate_portrait_hq(output_dir: Path) -> list[str]:
    issues = []
    metadata = _load_metadata(output_dir)
    for entry in metadata:
        if entry.get("faces_detected", 0) < 1:
            issues.append(f"{entry.get('frame_id')} has no faces")
        if entry.get("pose_category") != "portrait":
            issues.append(f"{entry.get('frame_id')} pose is {entry.get('pose_category')}")
        if entry.get("blur_score", 0) <= 120:
            issues.append(f"{entry.get('frame_id')} blur score {entry.get('blur_score')}")
    scores = [entry.get("composite_score", 0) for entry in metadata]
    if scores and scores != sorted(scores, reverse=True):
        issues.append("Portrait HQ output not sorted by composite_score")
    return issues


def _generate_edge_case_videos(output_dir: Path) -> dict[str, Path]:
    _ensure_dir(output_dir)
    cap = cv2.VideoCapture(str(SOURCE_VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while len(frames) < int(fps * 3):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("Failed to read frames for edge case generation.")

    height, width = frames[0].shape[:2]
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)

    def write_video(path: Path, video_frames: list[np.ndarray]) -> None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (video_frames[0].shape[1], video_frames[0].shape[0]),
        )
        for frame in video_frames:
            writer.write(frame)
        writer.release()

    no_faces = output_dir / "edge_no_faces.mp4"
    write_video(no_faces, [black_frame] * int(fps * 3))

    blurry_frames = [cv2.GaussianBlur(frame, (31, 31), 0) for frame in frames]
    blurry = output_dir / "edge_blurry.mp4"
    write_video(blurry, blurry_frames)

    multi_frames = [np.concatenate([frame, frame], axis=1) for frame in frames]
    multiple = output_dir / "edge_multiple.mp4"
    write_video(multiple, multi_frames)

    short_video = output_dir / "edge_short.mp4"
    write_video(short_video, frames[:int(fps * 2)])

    return {
        "no_faces": no_faces,
        "blurry": blurry,
        "multiple": multiple,
        "short": short_video,
    }


def _write_edge_cases(report: list[str]) -> None:
    path = RESULTS_DIR / "edge_cases.md"
    path.write_text("\n".join(report), encoding="utf-8")


def _write_summary(results: list[ScenarioResult], portrait_issues: list[str],
                   gpu_idle: str,
                   yolo_faces: int | None = None,
                   haar_faces: int | None = None) -> None:
    report = [
        "# EVALUATION REPORT",
        "",
        "## Executive Summary",
        "- Best portrait extraction: `portrait_hq` (faces required, portrait filter, blur threshold 120).",
        f"- GPU idle ratio: {gpu_idle}",
        "- Quality improvement: person-mode filters remove non-face frames vs baseline.",
        "",
        "## Quantitative Results",
        f"See `results/comparison.md`.",
        "",
        "## Quality Assessment",
        f"Portrait issues: {len(portrait_issues)}",
        f"YOLO vs Haar faces detected: {yolo_faces} vs {haar_faces}",
        "",
        "## Performance Analysis",
        "See `results/performance.json` and `results/gpu_stats.csv`.",
        "",
        "## Recommendations",
        "- Use `--person-mode --require-faces --pose-filter portrait --blur-threshold 120 --top-n 5` for portraits.",
        "- If GPU utilization stays below 70%, increase batch size or use faster face model.",
        "- Consider caching face detections for re-runs.",
        "",
        "## TODO",
        "- [ ] Optimize slowest operation (if >2s per video).",
        "- [ ] Fine-tune composite score weights based on output quality.",
        "- [ ] Add resume capability if batch processing interrupted.",
        "- [ ] Consider caching face detection results for re-runs.",
        "- [ ] Implement smart interval (extract more frames near detected faces).",
    ]
    (RESULTS_DIR / "EVALUATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    _ensure_dir(RESULTS_DIR)

    env = os.environ.copy()
    env["PERFECTFRAMEAI_DOCKER_BIN"] = DOCKER_EXE
    env["PERFECTFRAMEAI_LOG_PERF"] = "1"
    env["PERFECTFRAMEAI_LOG_GPU"] = "1"
    env["PERFECTFRAMEAI_LOG_GPU_BATCH"] = "1"
    env["PERFECTFRAMEAI_PERF_PATH"] = "/app/output_directory/performance.json"

    scenarios = [
        ("baseline", ["--input", str(INPUT_DIR), "--output", str(RESULTS_DIR / "baseline"),
                      "--top-n", "10", "--build"]),
        ("person_optional", ["--input", str(INPUT_DIR), "--output", str(RESULTS_DIR / "person_optional"),
                             "--person-mode", "--top-n", "10"]),
        ("person_required", ["--input", str(INPUT_DIR), "--output", str(RESULTS_DIR / "person_required"),
                             "--person-mode", "--require-faces", "--top-n", "10"]),
        ("portrait_hq", ["--input", str(INPUT_DIR), "--output", str(RESULTS_DIR / "portrait_hq"),
                         "--person-mode", "--require-faces", "--pose-filter", "portrait",
                         "--min-face-area", "0.05", "--blur-threshold", "120", "--top-n", "5"]),
    ]

    results: list[ScenarioResult] = []
    combined_gpu_rows: list[list[str]] = [["scenario", "utilization.gpu", "utilization.memory", "memory.used"]]

    for name, args in scenarios:
        output_dir = RESULTS_DIR / name
        if output_dir.exists():
            shutil.rmtree(output_dir)
        _ensure_dir(output_dir)
        _reset_input_video()

        gpu_tmp = RESULTS_DIR / f"gpu_{name}.csv"
        gpu_proc = _start_gpu_logger(gpu_tmp)
        try:
            total_time = _run_cli(args, env)
        finally:
            _stop_gpu_logger(gpu_proc)

        avg_gpu, peak_mem = _parse_gpu_stats(gpu_tmp)
        if gpu_tmp.exists():
            with gpu_tmp.open("r", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                for row in reader:
                    if len(row) < 3:
                        continue
                    combined_gpu_rows.append([name] + row[:3])
            gpu_tmp.unlink()

        metadata = _load_metadata(output_dir)
        frames_extracted = _count_images(output_dir)
        avg_composite = _average([entry.get("composite_score", 0) for entry in metadata]) \
            if metadata else None
        avg_aesthetic = _average([entry.get("aesthetic_score", 0) for entry in metadata]) \
            if metadata else None
        avg_face = _average([entry.get("face_quality_score", 0) for entry in metadata]) \
            if metadata else None
        blur_rejection = sum(1 for entry in metadata if entry.get("is_blurry")) if metadata else None
        face_success_rate = (
            sum(1 for entry in metadata if entry.get("faces_detected", 0) > 0) / len(metadata)
            if metadata else None
        )

        per_video_times = {}
        perf_data = _parse_perf(output_dir / "performance.json")
        for video_name, metrics in perf_data.items():
            per_video_times[video_name] = metrics.get("video_total", 0.0)

        results.append(
            ScenarioResult(
                name=name,
                output_dir=output_dir,
                frames_extracted=frames_extracted,
                avg_composite=avg_composite,
                avg_aesthetic=avg_aesthetic,
                avg_face_quality=avg_face,
                blur_rejection=blur_rejection,
                face_success_rate=face_success_rate,
                total_time=total_time,
                per_video_times=per_video_times,
                gpu_avg_util=avg_gpu,
                gpu_peak_mem=peak_mem,
            )
        )

    (RESULTS_DIR / "gpu_stats.csv").write_text(
        "\n".join([",".join(row) for row in combined_gpu_rows]),
        encoding="utf-8",
    )

    perf_summary = {"scenarios": {}}
    for result in results:
        perf_path = result.output_dir / "performance.json"
        if perf_path.exists():
            with perf_path.open("r", encoding="utf-8") as handle:
                perf_summary["scenarios"][result.name] = json.load(handle)
    (RESULTS_DIR / "performance.json").write_text(
        json.dumps(perf_summary, indent=2),
        encoding="utf-8",
    )

    _write_comparison(results)
    _write_quality_grid(RESULTS_DIR / "portrait_hq")

    portrait_issues = _validate_portrait_hq(RESULTS_DIR / "portrait_hq")

    validator = ROOT / "tests" / "validate_metadata.py"
    subprocess.run(
        [sys.executable, str(validator), str(RESULTS_DIR / "person_optional")],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(validator), str(RESULTS_DIR / "person_required")],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(validator), str(RESULTS_DIR / "portrait_hq"),
         "--blur-threshold", "120"],
        check=True,
    )

    # Edge cases
    edge_input = RESULTS_DIR / "edge_case_inputs"
    edge_outputs = RESULTS_DIR / "edge_case_outputs"
    edge_cases = _generate_edge_case_videos(edge_input)
    edge_report = ["# Edge Case Results", ""]
    for name, video_path in edge_cases.items():
        input_dir = edge_input / name
        if input_dir.exists():
            shutil.rmtree(input_dir)
        _ensure_dir(input_dir)
        shutil.copy2(video_path, input_dir / video_path.name)
        output_dir = edge_outputs / name
        if output_dir.exists():
            shutil.rmtree(output_dir)
        _ensure_dir(output_dir)
        args = [
            "--input", str(input_dir),
            "--output", str(output_dir),
            "--person-mode",
            "--top-n", "5",
        ]
        if name == "blurry":
            args += ["--blur-threshold", "150"]
        _run_cli(args, env)
        metadata = _load_metadata(output_dir)
        faces_detected = [entry.get("faces_detected", 0) for entry in metadata]
        edge_report.append(f"## {name}")
        edge_report.append(f"- frames: {_count_images(output_dir)}")
        edge_report.append(f"- faces_detected: {faces_detected}")
        edge_report.append("")
    _write_edge_cases(edge_report)

    idle_samples = 0
    total_samples = 0
    for row in combined_gpu_rows[1:]:
        if len(row) < 2:
            continue
        try:
            util = float(row[1].strip().split()[0])
        except ValueError:
            continue
        total_samples += 1
        if util < 10:
            idle_samples += 1
    idle_ratio = (idle_samples / total_samples) if total_samples else 0.0

    yolo_faces = None
    haar_faces = None
    yolo_metadata = _load_metadata(RESULTS_DIR / "person_required")
    if yolo_metadata:
        yolo_faces = sum(entry.get("faces_detected", 0) for entry in yolo_metadata)

    haar_output = RESULTS_DIR / "haar_fallback"
    if haar_output.exists():
        shutil.rmtree(haar_output)
    _ensure_dir(haar_output)
    _reset_input_video()
    env_haar = env.copy()
    env_haar["PERFECTFRAMEAI_FORCE_HAAR"] = "1"
    _run_cli([
        "--input", str(INPUT_DIR),
        "--output", str(haar_output),
        "--person-mode",
        "--require-faces",
        "--top-n", "10",
    ], env_haar)
    haar_metadata = _load_metadata(haar_output)
    if haar_metadata:
        haar_faces = sum(entry.get("faces_detected", 0) for entry in haar_metadata)

    _write_summary(results, portrait_issues, f"{idle_ratio:.1%}", yolo_faces, haar_faces)


if __name__ == "__main__":
    main()
