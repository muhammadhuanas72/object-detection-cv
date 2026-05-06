from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import os
import threading
from pathlib import Path
from typing import Callable

import cv2

ULTRALYTICS_CONFIG_DIR = Path(__file__).resolve().parent / ".ultralytics"
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))

from ultralytics import YOLO

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
BROWSER_VIDEO_CODEC = "mp4v"
TRACK_CONFIDENCE = float(os.environ.get("YOLO_TRACK_CONFIDENCE", "0.45"))
TRACK_NMS_IOU = float(os.environ.get("YOLO_TRACK_NMS_IOU", "0.55"))
MIN_TRACK_FRAMES = int(os.environ.get("YOLO_MIN_TRACK_FRAMES", "5"))
MIN_TRACK_AVG_CONFIDENCE = float(os.environ.get("YOLO_MIN_TRACK_AVG_CONFIDENCE", "0.5"))
MAX_TRACK_MERGE_GAP = int(os.environ.get("YOLO_MAX_TRACK_MERGE_GAP", "12"))
TRACK_MERGE_IOU = float(os.environ.get("YOLO_TRACK_MERGE_IOU", "0.35"))
TRACK_MERGE_CENTER_RATIO = float(os.environ.get("YOLO_TRACK_MERGE_CENTER_RATIO", "0.08"))


@dataclass
class DetectionEntry:
    label: str
    confidence: float
    track_id: int | None
    bbox: tuple[float, float, float, float] | None


@dataclass
class StableTrackState:
    label: str
    first_frame: int
    last_frame: int
    frames_seen: int = 0
    confidence_sum: float = 0.0
    best_confidence: float = 0.0
    last_bbox: tuple[float, float, float, float] | None = None
    source_track_ids: set[int] = field(default_factory=set)

    @property
    def average_confidence(self) -> float:
        if self.frames_seen == 0:
            return 0.0
        return self.confidence_sum / self.frames_seen

    @property
    def is_countable(self) -> bool:
        return (
            self.frames_seen >= MIN_TRACK_FRAMES
            and self.average_confidence >= MIN_TRACK_AVG_CONFIDENCE
        )

    def record(
        self,
        *,
        confidence: float,
        frame_index: int,
        bbox: tuple[float, float, float, float] | None,
        track_id: int | None,
    ) -> None:
        self.frames_seen += 1
        self.confidence_sum += confidence
        self.best_confidence = max(self.best_confidence, confidence)
        self.last_frame = frame_index
        self.last_bbox = bbox
        if track_id is not None:
            self.source_track_ids.add(track_id)


def _label_from_names(names: dict | list, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def extract_detection_entries(result) -> list[DetectionEntry]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return []

    class_ids = [int(value) for value in boxes.cls.tolist()]
    track_ids: list[int | None] | None = None
    if getattr(boxes, "id", None) is not None:
        track_ids = [int(value) for value in boxes.id.tolist()]
    confidences: list[float] | None = None
    if getattr(boxes, "conf", None) is not None:
        confidences = [float(value) for value in boxes.conf.tolist()]
    bboxes: list[list[float]] | None = None
    if getattr(boxes, "xyxy", None) is not None:
        bboxes = [[float(point) for point in points] for points in boxes.xyxy.tolist()]

    entries: list[DetectionEntry] = []
    for index, class_id in enumerate(class_ids):
        label = _label_from_names(result.names, class_id)
        track_id = None
        if track_ids is not None and index < len(track_ids):
            track_id = track_ids[index]
        confidence = 0.0
        if confidences is not None and index < len(confidences):
            confidence = confidences[index]
        bbox = None
        if bboxes is not None and index < len(bboxes):
            bbox = tuple(bboxes[index])
        entries.append(
            DetectionEntry(
                label=label,
                confidence=confidence,
                track_id=track_id,
                bbox=bbox,
            )
        )

    return entries


def _sorted_counts(counter: Counter[str] | dict[str, int]) -> list[dict[str, int | str]]:
    return [
        {"label": label, "count": count}
        for label, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _bbox_iou(
    first_box: tuple[float, float, float, float] | None,
    second_box: tuple[float, float, float, float] | None,
) -> float:
    if first_box is None or second_box is None:
        return 0.0

    x1 = max(first_box[0], second_box[0])
    y1 = max(first_box[1], second_box[1])
    x2 = min(first_box[2], second_box[2])
    y2 = min(first_box[3], second_box[3])

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)
    intersection_area = intersection_width * intersection_height
    if intersection_area <= 0:
        return 0.0

    first_area = max(0.0, first_box[2] - first_box[0]) * max(0.0, first_box[3] - first_box[1])
    second_area = max(0.0, second_box[2] - second_box[0]) * max(0.0, second_box[3] - second_box[1])
    union_area = first_area + second_area - intersection_area
    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def _bbox_center_distance_ratio(
    first_box: tuple[float, float, float, float] | None,
    second_box: tuple[float, float, float, float] | None,
    *,
    frame_diagonal: float,
) -> float:
    if first_box is None or second_box is None or frame_diagonal <= 0:
        return 1.0

    first_center_x = (first_box[0] + first_box[2]) / 2
    first_center_y = (first_box[1] + first_box[3]) / 2
    second_center_x = (second_box[0] + second_box[2]) / 2
    second_center_y = (second_box[1] + second_box[3]) / 2
    distance = math.dist((first_center_x, first_center_y), (second_center_x, second_center_y))
    return distance / frame_diagonal


def build_detection_summary(
    *,
    frame_detection_totals: Counter[str],
    peak_frame_counts: Counter[str],
    object_counts: Counter[str] | dict[str, int],
    count_basis: str,
) -> dict:
    if object_counts:
        normalized_counts = dict(object_counts)
        normalized_basis = count_basis
    else:
        normalized_counts = dict(peak_frame_counts)
        normalized_basis = "peak_frame_detections"

    return {
        "object_counts": _sorted_counts(normalized_counts),
        "count_basis": normalized_basis,
        "frame_detection_totals": _sorted_counts(frame_detection_totals),
        "peak_frame_counts": _sorted_counts(peak_frame_counts),
    }


class StableObjectCounter:
    def __init__(self, *, frame_width: int, frame_height: int):
        self.frame_detection_totals: Counter[str] = Counter()
        self.peak_frame_counts: Counter[str] = Counter()
        self._frame_diagonal = max(math.hypot(frame_width, frame_height), 1.0)
        self._canonical_tracks: dict[int, StableTrackState] = {}
        self._source_to_canonical: dict[tuple[str, int], int] = {}
        self._next_track_key = 1

    def ingest(self, detections: list[DetectionEntry], *, frame_index: int) -> Counter[str]:
        frame_counts = Counter(detection.label for detection in detections)
        self.frame_detection_totals.update(frame_counts)
        for label, count in frame_counts.items():
            self.peak_frame_counts[label] = max(self.peak_frame_counts[label], count)

        stable_current_counts: Counter[str] = Counter()
        for detection in detections:
            if detection.track_id is None or detection.bbox is None:
                continue

            canonical_id = self._resolve_track(detection, frame_index=frame_index)
            track_state = self._canonical_tracks[canonical_id]
            track_state.record(
                confidence=detection.confidence,
                frame_index=frame_index,
                bbox=detection.bbox,
                track_id=detection.track_id,
            )
            if track_state.is_countable:
                stable_current_counts[detection.label] += 1

        return stable_current_counts

    def summary(self) -> dict:
        counted_tracks: Counter[str] = Counter()
        for track_state in self._canonical_tracks.values():
            if track_state.is_countable:
                counted_tracks[track_state.label] += 1

        return build_detection_summary(
            frame_detection_totals=self.frame_detection_totals,
            peak_frame_counts=self.peak_frame_counts,
            object_counts=counted_tracks,
            count_basis="unique_tracked_objects",
        )

    def _resolve_track(self, detection: DetectionEntry, *, frame_index: int) -> int:
        assert detection.track_id is not None

        source_key = (detection.label, detection.track_id)
        if source_key in self._source_to_canonical:
            return self._source_to_canonical[source_key]

        matched_track_id = self._match_existing_track(detection, frame_index=frame_index)
        if matched_track_id is not None:
            self._source_to_canonical[source_key] = matched_track_id
            return matched_track_id

        canonical_id = self._next_track_key
        self._next_track_key += 1
        self._canonical_tracks[canonical_id] = StableTrackState(
            label=detection.label,
            first_frame=frame_index,
            last_frame=frame_index,
        )
        self._source_to_canonical[source_key] = canonical_id
        return canonical_id

    def _match_existing_track(
        self,
        detection: DetectionEntry,
        *,
        frame_index: int,
    ) -> int | None:
        best_track_id: int | None = None
        best_score = -1.0

        for track_id, track_state in self._canonical_tracks.items():
            if track_state.label != detection.label:
                continue

            frame_gap = frame_index - track_state.last_frame
            if frame_gap <= 0 or frame_gap > MAX_TRACK_MERGE_GAP:
                continue

            overlap = _bbox_iou(track_state.last_bbox, detection.bbox)
            distance_ratio = _bbox_center_distance_ratio(
                track_state.last_bbox,
                detection.bbox,
                frame_diagonal=self._frame_diagonal,
            )

            if overlap < TRACK_MERGE_IOU and distance_ratio > TRACK_MERGE_CENTER_RATIO:
                continue

            score = overlap + (1.0 - distance_ratio)
            if score > best_score:
                best_track_id = track_id
                best_score = score

        return best_track_id


class VideoProcessingError(RuntimeError):
    """Raised when a video source or stream cannot be processed."""


class VideoTracker:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)

    def _build_model(self) -> YOLO:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        return YOLO(str(self.model_path))

    def process_video(
        self,
        source_path: str | Path,
        output_path: str | Path,
        *,
        frame_stride: int = 1,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> dict:
        source_path = Path(source_path)
        output_path = Path(output_path)
        frame_stride = max(1, int(frame_stride))

        if not source_path.exists():
            raise FileNotFoundError(f"Video file not found: {source_path}")

        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise VideoProcessingError(f"Could not open video: {source_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        if width <= 0 or height <= 0:
            capture.release()
            raise VideoProcessingError("The uploaded video does not report a valid frame size.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_fps = max(fps / frame_stride, 1.0)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*BROWSER_VIDEO_CODEC),
            output_fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise VideoProcessingError(
                f"Could not create output video with codec {BROWSER_VIDEO_CODEC}: {output_path}"
            )

        model = self._build_model()
        processed_frames = 0
        written_frames = 0
        stable_counter = StableObjectCounter(frame_width=width, frame_height=height)

        def report_progress() -> None:
            if progress_callback is None:
                return

            progress = 0
            if total_frames > 0:
                progress = min(int(processed_frames * 100 / total_frames), 100)

            progress_callback(
                {
                    "processed_frames": processed_frames,
                    "written_frames": written_frames,
                    "total_frames": total_frames,
                    "progress": progress,
                }
            )

        try:
            report_progress()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                processed_frames += 1
                if frame_stride > 1 and (processed_frames - 1) % frame_stride != 0:
                    if processed_frames == 1 or processed_frames % 15 == 0:
                        report_progress()
                    continue

                results = model.track(
                    frame,
                    persist=True,
                    verbose=False,
                    conf=TRACK_CONFIDENCE,
                    iou=TRACK_NMS_IOU,
                )
                detection_result = results[0]
                detection_entries = extract_detection_entries(detection_result)
                stable_counter.ingest(detection_entries, frame_index=processed_frames)

                annotated_frame = detection_result.plot()
                writer.write(annotated_frame)
                written_frames += 1

                if processed_frames == 1 or processed_frames % 15 == 0:
                    report_progress()

            if written_frames == 0:
                raise VideoProcessingError("No frames were written to the output video.")

            processed_frames = total_frames or processed_frames
            report_progress()
            return stable_counter.summary()
        finally:
            capture.release()
            writer.release()


class WebcamTracker:
    def __init__(self, model_path: str | Path, camera_index: int = 0):
        self.model_path = Path(model_path)
        self.camera_index = camera_index
        self._lock = threading.Lock()
        self._capture: cv2.VideoCapture | None = None
        self._model: YOLO | None = None
        self._running = False
        self._source: int | str = camera_index
        self._frame_index = 0
        self._stable_counter: StableObjectCounter | None = None
        self._latest_stats = {
            "current_frame_counts": [],
            "tracked_object_counts": [],
            "count_basis": "current_frame_detections",
        }

    def _stop_unlocked(self) -> None:
        self._running = False
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._model = None
        self._frame_index = 0
        self._stable_counter = None
        self._latest_stats = {
            "current_frame_counts": [],
            "tracked_object_counts": [],
            "count_basis": "current_frame_detections",
        }

    def _normalize_source(self, camera_source: int | str | None) -> int | str:
        if camera_source is None:
            return self.camera_index
        if isinstance(camera_source, int):
            return camera_source

        source_text = str(camera_source).strip()
        if not source_text:
            return self.camera_index
        if source_text.isdigit():
            return int(source_text)
        return source_text

    def start(self, camera_source: int | str | None = None) -> None:
        with self._lock:
            normalized_source = self._normalize_source(camera_source)

            if self._running and normalized_source == self._source:
                return
            if self._running:
                self._stop_unlocked()

            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            capture = cv2.VideoCapture(normalized_source)
            if not capture.isOpened():
                capture.release()
                raise VideoProcessingError(
                    f"Could not open camera source: {normalized_source}."
                )

            self._capture = capture
            self._model = YOLO(str(self.model_path))
            self._running = True
            self._source = normalized_source
            self._frame_index = 0
            self._stable_counter = None
            self._latest_stats = {
                "current_frame_counts": [],
                "tracked_object_counts": [],
                "count_basis": "current_frame_detections",
            }

    def stop(self) -> None:
        with self._lock:
            self._stop_unlocked()

    @property
    def source(self) -> int | str:
        return self._source

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "source": str(self._source),
                "running": self._running,
                "current_frame_counts": list(self._latest_stats["current_frame_counts"]),
                "tracked_object_counts": list(self._latest_stats["tracked_object_counts"]),
                "count_basis": self._latest_stats["count_basis"],
            }

    def stream(self):
        try:
            while True:
                with self._lock:
                    if not self._running or self._capture is None or self._model is None:
                        break

                    ok, frame = self._capture.read()
                    if not ok:
                        raise VideoProcessingError("Unable to read a frame from the webcam.")

                    if self._stable_counter is None:
                        frame_height, frame_width = frame.shape[:2]
                        self._stable_counter = StableObjectCounter(
                            frame_width=frame_width,
                            frame_height=frame_height,
                        )

                    self._frame_index += 1
                    results = self._model.track(
                        frame,
                        persist=True,
                        verbose=False,
                        conf=TRACK_CONFIDENCE,
                        iou=TRACK_NMS_IOU,
                    )
                    detection_result = results[0]
                    detection_entries = extract_detection_entries(detection_result)
                    raw_current_counts = Counter(entry.label for entry in detection_entries)
                    self._stable_counter.ingest(detection_entries, frame_index=self._frame_index)
                    live_summary = self._stable_counter.summary()
                    self._latest_stats = {
                        "current_frame_counts": _sorted_counts(raw_current_counts),
                        "tracked_object_counts": live_summary["object_counts"],
                        "count_basis": live_summary["count_basis"],
                    }

                annotated_frame = detection_result.plot()
                success, buffer = cv2.imencode(".jpg", annotated_frame)
                if not success:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
        finally:
            self.stop()
