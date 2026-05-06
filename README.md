# YOLO Vision Desk

This project turns your local YOLOv8 tracking script into a browser app with two modes:

- upload a video and receive an annotated output file
- start live webcam tracking from a page in your browser

Uploads now run as background jobs, so the browser does not sit on one long request while every frame is processed.

## What this app can deploy well

The upload workflow deploys cleanly as a normal web app.

The current live camera workflow is server-side. That means:

- `0`, `1`, `2` camera sources use cameras attached to the machine running Python
- phone cameras work when the phone exposes an IP stream and you paste that stream URL into the app
- a cloud host does not automatically get access to a visitor's browser webcam

If you later want true browser-to-server webcam capture for remote users, we should build a separate WebRTC or browser-frame-upload flow.

## Files

- `app.py` runs the Flask server
- `detector.py` contains the video and webcam tracking logic
- `templates/index.html` is the browser UI
- `static/` contains the page styling and client-side behavior

## Python setup

Your old project virtual environment at `D:\Anas coding\computer vision object detction\.venv` is broken because it points at a missing Python 3.8 install. A clean Python 3.10 environment is the safest reset on this machine.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the app

The app prefers a local workspace model at:

`models\yolov8n.pt`

If that file is missing, it falls back to:

`D:\Anas coding\computer vision object detction\yolov8n.pt`

If you want to use a different model file, set `YOLO_MODEL_PATH` before starting Flask.

```powershell
$env:YOLO_MODEL_PATH='C:\path\to\your-model.pt'
python app.py
```

Then open:

`http://127.0.0.1:5000`

## Deploy with Docker

This project now includes a `Dockerfile`, which is the easiest production path for platforms like Render, Railway, or any VPS that supports Docker.

### 1. Make sure the model file is available

The container expects the YOLO weights at:

`models/yolov8n.pt`

If you do not want to keep the model inside the repo, deploy with an environment variable that points to a model file already present on the host:

`YOLO_MODEL_PATH=/absolute/path/to/yolov8n.pt`

### 2. Build locally

```powershell
docker build -t yolo-vision-desk .
```

### 3. Run locally in a container

```powershell
docker run --rm -p 5000:5000 yolo-vision-desk
```

Then open:

`http://127.0.0.1:5000`

### 4. Deploy to a cloud host

The most practical hosted setup is:

1. Push this project to GitHub
2. Keep `Dockerfile` in the project root
3. Create a web service on a Docker-friendly host
4. Set `YOLO_MODEL_PATH` if your model lives outside `models/yolov8n.pt`
5. Deploy

Useful official docs:

- Docker Python container guide: https://docs.docker.com/guides/python/containerize/
- Render web services: https://render.com/docs/web-services/
- Render Docker deploys: https://render.com/docs/docker
- Railway Dockerfile deploys: https://docs.railway.com/deploy/dockerfiles

## Production notes

- The app now respects the host's `PORT` environment variable, which most platforms require
- Uploaded files and output videos are written to the local filesystem inside `uploads/` and `outputs/`
- On many cloud hosts, that filesystem is ephemeral, so files can disappear after restart or redeploy
- For a longer-lived production version, store uploads and processed videos in object storage such as S3-compatible storage
- Job state is currently stored in memory, so this app is best deployed as a single instance unless we move jobs into a database or queue

## Example production environment variables

```text
PORT=5000
YOLO_MODEL_PATH=/app/models/yolov8n.pt
CAMERA_INDEX=0
YOLO_TRACK_CONFIDENCE=0.45
YOLO_MIN_TRACK_FRAMES=5
YOLO_MIN_TRACK_AVG_CONFIDENCE=0.5
WAITRESS_THREADS=4
```

## Notes

- Uploaded videos are stored in `uploads/`
- Processed outputs are written to `outputs/`
- The upload screen includes `Fast`, `Balanced`, and `Accurate` processing modes
- Live webcam tracking uses camera index `0` by default
- The live camera input also accepts a network camera URL, which lets you use a phone camera if the phone exposes an IP stream
- If your webcam is on another index, set `CAMERA_INDEX` before running the app

```powershell
$env:CAMERA_INDEX='1'
python app.py
```
