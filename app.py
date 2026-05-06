from __future__ import annotations

print("STEP 1 - app.py started")

import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from flask import Flask, Response, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from detector import VIDEO_EXTENSIONS, VideoProcessingError, VideoTracker, WebcamTracker

print("STEP 2 - imports completed")
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
WORKSPACE_MODEL_PATH = BASE_DIR / "models" / "yolov8n.pt"
LEGACY_MODEL_PATH = Path(r"D:\Anas coding\computer vision object detction\yolov8n.pt")


def resolve_default_model_path() -> Path:
    configured_model = os.environ.get("YOLO_MODEL_PATH")
    if configured_model:
        return Path(configured_model)
    if WORKSPACE_MODEL_PATH.exists():
        return WORKSPACE_MODEL_PATH
    if LEGACY_MODEL_PATH.exists():
        return LEGACY_MODEL_PATH
    return WORKSPACE_MODEL_PATH


DEFAULT_MODEL_PATH = resolve_default_model_path()
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
MAX_UPLOAD_BYTES = 512 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

print("STEP 3 - creating webcam tracker")
webcam_tracker = WebcamTracker(DEFAULT_MODEL_PATH, camera_index=CAMERA_INDEX)
print("STEP 4 - webcam tracker created")

JOBS_LOCK = threading.Lock()
JOBS: dict[str, dict] = {}

VIDEO_SPEED_PROFILES = {
    "fast": {
        "label": "Fast",
        "description": "Processes every third frame for the quickest result.",
        "frame_stride": 3,
    },
    "balanced": {
        "label": "Balanced",
        "description": "Processes every other frame for a good speed and quality balance.",
        "frame_stride": 2,
    },
    "accurate": {
        "label": "Accurate",
        "description": "Processes every frame for the best tracking quality.",
        "frame_stride": 1,
    },
}


def ensure_storage() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def json_error(message: str, status_code: int = 400):
    return jsonify({"error": message}), status_code


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_payload(job_id: str) -> dict | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return None
        return dict(job)


def update_job(job_id: str, **changes) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job.update(changes)


def run_video_job(
    *,
    job_id: str,
    upload_path: Path,
    output_path: Path,
    download_name: str,
    speed_mode: str,
) -> None:
    profile = VIDEO_SPEED_PROFILES[speed_mode]

    def on_progress(progress: dict) -> None:
        total_frames = progress.get("total_frames") or 0
        processed_frames = progress.get("processed_frames") or 0
        message = f"Processed {processed_frames}"
        if total_frames > 0:
            message += f" of {total_frames} frames"
        update_job(
            job_id,
            status="processing",
            progress=progress.get("progress", 0),
            processed_frames=processed_frames,
            written_frames=progress.get("written_frames", 0),
            total_frames=total_frames,
            message=message,
            updated_at=utc_now(),
        )

    try:
        tracker = VideoTracker(DEFAULT_MODEL_PATH)
        update_job(
            job_id,
            status="processing",
            progress=0,
            message="Preparing video tracking.",
            updated_at=utc_now(),
        )
        summary = tracker.process_video(
            upload_path,
            output_path,
            frame_stride=profile["frame_stride"],
            progress_callback=on_progress,
        )
        update_job(
            job_id,
            status="completed",
            progress=100,
            message="Video processed successfully.",
            output_url=f"/outputs/{output_path.name}",
            download_name=download_name,
            object_counts=summary["object_counts"],
            count_basis=summary["count_basis"],
            frame_detection_totals=summary["frame_detection_totals"],
            peak_frame_counts=summary["peak_frame_counts"],
            updated_at=utc_now(),
            completed_at=utc_now(),
        )
    except (FileNotFoundError, VideoProcessingError) as exc:
        upload_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        update_job(
            job_id,
            status="failed",
            error=str(exc),
            message=str(exc),
            updated_at=utc_now(),
            completed_at=utc_now(),
        )
    except Exception as exc:  # pragma: no cover - defensive fallback for runtime inference errors
        upload_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        update_job(
            job_id,
            status="failed",
            error=f"Unexpected processing error: {exc}",
            message=f"Unexpected processing error: {exc}",
            updated_at=utc_now(),
            completed_at=utc_now(),
        )


@app.get("/")
def index():
    return render_template(
        "index.html",
        model_path=str(DEFAULT_MODEL_PATH),
        model_exists=DEFAULT_MODEL_PATH.exists(),
        camera_index=CAMERA_INDEX,
        default_camera_source=str(CAMERA_INDEX),
        supported_formats=", ".join(sorted(ext.lstrip(".") for ext in VIDEO_EXTENSIONS)),
        speed_profiles=VIDEO_SPEED_PROFILES,
        default_speed_mode="balanced",
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_path": str(DEFAULT_MODEL_PATH),
            "model_exists": DEFAULT_MODEL_PATH.exists(),
            "camera_index": CAMERA_INDEX,
            "camera_source": str(webcam_tracker.source),
        }
    )


@app.post("/api/process-video")
def process_video():
    ensure_storage()

    if "video" not in request.files:
        return json_error("Choose a video file first.")

    uploaded_file = request.files["video"]
    if not uploaded_file.filename:
        return json_error("The selected file has no name.")

    safe_name = secure_filename(uploaded_file.filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        return json_error(
            f"Unsupported video type. Use one of: {', '.join(sorted(VIDEO_EXTENSIONS))}."
        )

    speed_mode = request.form.get("speed_mode", "balanced")
    if speed_mode not in VIDEO_SPEED_PROFILES:
        return json_error("Unsupported speed option selected.")

    source_name = f"{uuid4().hex}{suffix}"
    upload_path = UPLOADS_DIR / source_name
    output_path = OUTPUTS_DIR / f"{Path(source_name).stem}_tracked.mp4"
    download_name = f"{Path(safe_name).stem}_tracked.mp4"
    uploaded_file.save(upload_path)
    job_id = uuid4().hex

    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Queued for processing.",
            "source_name": uploaded_file.filename,
            "download_name": download_name,
            "speed_mode": speed_mode,
            "error": None,
            "output_url": None,
            "processed_frames": 0,
            "written_frames": 0,
            "total_frames": 0,
            "object_counts": [],
            "count_basis": "unique_tracked_objects",
            "frame_detection_totals": [],
            "peak_frame_counts": [],
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "completed_at": None,
        }

    worker = threading.Thread(
        target=run_video_job,
        kwargs={
            "job_id": job_id,
            "upload_path": upload_path,
            "output_path": output_path,
            "download_name": download_name,
            "speed_mode": speed_mode,
        },
        daemon=True,
    )
    worker.start()

    return (
        jsonify(
            {
                "job_id": job_id,
                "status": "queued",
                "message": "Video accepted for processing.",
                "status_url": url_for("job_status", job_id=job_id),
            }
        ),
        202,
    )


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    job = job_payload(job_id)
    if job is None:
        return json_error("Job not found.", 404)
    return jsonify(job)


@app.post("/api/webcam/start")
def start_webcam():
    payload = request.get_json(silent=True) or {}
    camera_source = payload.get("camera_source", str(CAMERA_INDEX))

    try:
        webcam_tracker.start(camera_source=camera_source)
    except (FileNotFoundError, VideoProcessingError) as exc:
        return json_error(str(exc), 500)
    except Exception as exc:  # pragma: no cover - defensive fallback for runtime inference errors
        return json_error(f"Unable to start webcam tracking: {exc}", 500)

    return jsonify(
        {
            "message": "Webcam started.",
            "stream_url": url_for("webcam_stream"),
            "camera_source": str(webcam_tracker.source),
        }
    )


@app.post("/api/webcam/stop")
def stop_webcam():
    webcam_tracker.stop()
    return jsonify({"message": "Webcam stopped."})


@app.get("/api/webcam/stats")
def webcam_stats():
    return jsonify(webcam_tracker.get_stats())


@app.get("/webcam_stream")
def webcam_stream():
    try:
        webcam_tracker.start()
    except (FileNotFoundError, VideoProcessingError) as exc:
        return json_error(str(exc), 500)

    return Response(
        webcam_tracker.stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/outputs/<path:filename>")
def serve_output(filename: str):
    return send_from_directory(OUTPUTS_DIR, filename)


if __name__ == "__main__":
    ensure_storage()
    from waitress import serve

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    threads = max(int(os.environ.get("WAITRESS_THREADS", "4")), 1)

    print(f"STEP 5 - starting server on {host}:{port}")

    serve(app, host=host, port=port, threads=threads)