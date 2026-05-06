const uploadForm = document.getElementById("upload-form");
const uploadButton = document.getElementById("upload-button");
const uploadStatus = document.getElementById("upload-status");
const progressShell = document.getElementById("progress-shell");
const progressBar = document.getElementById("progress-bar");
const progressMeta = document.getElementById("progress-meta");
const resultShell = document.getElementById("result-shell");
const resultVideo = document.getElementById("result-video");
const downloadLink = document.getElementById("download-link");
const speedMode = document.getElementById("speed-mode");
const speedHelp = document.getElementById("speed-help");
const uploadSummaryShell = document.getElementById("upload-summary-shell");
const uploadSummaryBasis = document.getElementById("upload-summary-basis");
const uploadSummaryEmpty = document.getElementById("upload-summary-empty");
const uploadSummaryList = document.getElementById("upload-summary-list");

const startWebcamButton = document.getElementById("start-webcam");
const stopWebcamButton = document.getElementById("stop-webcam");
const webcamStatus = document.getElementById("webcam-status");
const webcamStream = document.getElementById("webcam-stream");
const streamPlaceholder = document.getElementById("stream-placeholder");
const cameraSourceInput = document.getElementById("camera-source");
const liveSummaryShell = document.getElementById("live-summary-shell");
const liveSummaryBasis = document.getElementById("live-summary-basis");
const liveCurrentList = document.getElementById("live-current-list");
const liveTotalList = document.getElementById("live-total-list");

let currentJobId = null;
let jobPollHandle = null;
let webcamStatsHandle = null;

function setUploadState(message, isWorking) {
    uploadStatus.textContent = message;
    uploadButton.disabled = isWorking;
    speedMode.disabled = isWorking;
}

function showResultVideo(url, downloadName) {
    resultVideo.src = url;
    downloadLink.href = url;
    downloadLink.download = downloadName;
    resultShell.classList.remove("hidden");
}

function countBasisLabel(countBasis) {
    if (countBasis === "unique_tracked_objects") {
        return "Counts are based on unique tracked objects across the clip.";
    }
    if (countBasis === "peak_frame_detections") {
        return "Counts are based on the busiest processed frame.";
    }
    if (countBasis === "current_frame_detections") {
        return "Counts are based on detections in the current frame.";
    }
    return "";
}

function renderCountList(container, items) {
    container.innerHTML = "";

    if (!items || items.length === 0) {
        const emptyRow = document.createElement("div");
        emptyRow.className = "summary-row";
        emptyRow.innerHTML = "<span class=\"summary-label\">None</span><strong class=\"summary-count\">0</strong>";
        container.appendChild(emptyRow);
        return;
    }

    items.forEach((item) => {
        const row = document.createElement("div");
        row.className = "summary-row";
        row.innerHTML = `<span class="summary-label">${item.label}</span><strong class="summary-count">${item.count}</strong>`;
        container.appendChild(row);
    });
}

function renderUploadSummary(payload) {
    const counts = payload.object_counts || [];
    uploadSummaryShell.classList.remove("hidden");
    uploadSummaryBasis.textContent = countBasisLabel(payload.count_basis);
    uploadSummaryEmpty.classList.toggle("hidden", counts.length > 0);
    renderCountList(uploadSummaryList, counts);
}

function resetUploadSummary() {
    uploadSummaryShell.classList.add("hidden");
    uploadSummaryBasis.textContent = "";
    uploadSummaryEmpty.classList.add("hidden");
    uploadSummaryList.innerHTML = "";
}

function setProgress(progress, details = "") {
    progressShell.classList.remove("hidden");
    progressShell.setAttribute("aria-hidden", "false");
    progressBar.style.width = `${progress}%`;
    progressMeta.textContent = details || `${progress}%`;
}

function resetProgress() {
    progressBar.style.width = "0%";
    progressMeta.textContent = "0%";
    progressShell.classList.add("hidden");
    progressShell.setAttribute("aria-hidden", "true");
}

function stopPolling() {
    if (jobPollHandle) {
        clearTimeout(jobPollHandle);
        jobPollHandle = null;
    }
}

function stopWebcamStatsPolling() {
    if (webcamStatsHandle) {
        clearTimeout(webcamStatsHandle);
        webcamStatsHandle = null;
    }
}

function updateSpeedHelp() {
    const profile = window.appConfig.speedProfiles[speedMode.value];
    speedHelp.textContent = profile ? profile.description : "";
}

async function pollJob(jobId) {
    try {
        const response = await fetch(`/api/jobs/${jobId}`);
        const payload = await response.json();

        if (!response.ok) {
            throw new Error(payload.error || "Could not read processing status.");
        }

        const progress = Number(payload.progress || 0);
        const processed = Number(payload.processed_frames || 0);
        const total = Number(payload.total_frames || 0);
        const detail = total > 0 ? `${progress}% | ${processed}/${total} frames` : `${progress}%`;

        if (payload.status === "queued" || payload.status === "processing") {
            setUploadState(payload.message || "Processing video...", true);
            setProgress(progress, detail);
            jobPollHandle = setTimeout(() => pollJob(jobId), 1000);
            return;
        }

        if (payload.status === "completed") {
            stopPolling();
            currentJobId = null;
            setUploadState(payload.message || "Video processed successfully.", false);
            setProgress(100, "100% | Ready");
            showResultVideo(payload.output_url, payload.download_name);
            renderUploadSummary(payload);
            return;
        }

        stopPolling();
        currentJobId = null;
        setUploadState(payload.error || payload.message || "Video processing failed.", false);
    } catch (error) {
        stopPolling();
        currentJobId = null;
        setUploadState(error.message, false);
    }
}

async function handleUpload(event) {
    event.preventDefault();

    stopPolling();
    currentJobId = null;

    const formData = new FormData(uploadForm);
    if (!formData.get("video") || !formData.get("video").name) {
        setUploadState("Choose a video before running tracking.", false);
        return;
    }

    setUploadState("Uploading video and creating a processing job...", true);
    setProgress(2, "Starting");
    resultShell.classList.add("hidden");
    resultVideo.removeAttribute("src");
    resultVideo.load();
    resetUploadSummary();

    try {
        const response = await fetch("/api/process-video", {
            method: "POST",
            body: formData,
        });
        const payload = await response.json();

        if (!response.ok) {
            throw new Error(payload.error || "Video processing failed.");
        }

        currentJobId = payload.job_id;
        setUploadState(payload.message || "Video accepted for processing.", true);
        setProgress(5, "Queued");
        pollJob(currentJobId);
    } catch (error) {
        resetProgress();
        setUploadState(error.message, false);
    }
}

async function pollWebcamStats() {
    try {
        const response = await fetch("/api/webcam/stats");
        const payload = await response.json();

        if (!response.ok) {
            throw new Error(payload.error || "Could not read webcam stats.");
        }

        if (!payload.running) {
            liveSummaryShell.classList.add("hidden");
            stopWebcamStatsPolling();
            return;
        }

        liveSummaryShell.classList.remove("hidden");
        liveSummaryBasis.textContent = countBasisLabel(payload.count_basis);
        renderCountList(liveCurrentList, payload.current_frame_counts || []);
        renderCountList(liveTotalList, payload.tracked_object_counts || []);
        webcamStatsHandle = setTimeout(pollWebcamStats, 1000);
    } catch (error) {
        liveSummaryShell.classList.add("hidden");
        stopWebcamStatsPolling();
    }
}

async function startWebcam() {
    if (!window.appConfig.modelExists) {
        webcamStatus.textContent = "The configured model path is missing, so webcam tracking cannot start.";
        return;
    }

    const cameraSource = cameraSourceInput.value.trim() || "0";
    webcamStatus.textContent = "Starting webcam tracking...";
    startWebcamButton.disabled = true;
    cameraSourceInput.disabled = true;

    try {
        const response = await fetch("/api/webcam/start", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ camera_source: cameraSource }),
        });
        const payload = await response.json();

        if (!response.ok) {
            throw new Error(payload.error || "Unable to start webcam tracking.");
        }

        webcamStream.src = `${payload.stream_url}?t=${Date.now()}`;
        webcamStream.classList.remove("hidden");
        streamPlaceholder.classList.add("hidden");
        webcamStatus.textContent = `Live tracking is running from source ${payload.camera_source}.`;
        stopWebcamButton.disabled = false;
        stopWebcamStatsPolling();
        pollWebcamStats();
    } catch (error) {
        webcamStatus.textContent = error.message;
        startWebcamButton.disabled = false;
        cameraSourceInput.disabled = false;
    }
}

async function stopWebcam() {
    stopWebcamButton.disabled = true;

    try {
        await fetch("/api/webcam/stop", { method: "POST" });
    } catch (error) {
        console.error(error);
    }

    webcamStream.removeAttribute("src");
    webcamStream.classList.add("hidden");
    streamPlaceholder.classList.remove("hidden");
    webcamStatus.textContent = "Webcam tracking stopped.";
    startWebcamButton.disabled = false;
    cameraSourceInput.disabled = false;
    liveSummaryShell.classList.add("hidden");
    liveSummaryBasis.textContent = "";
    liveCurrentList.innerHTML = "";
    liveTotalList.innerHTML = "";
    stopWebcamStatsPolling();
}

uploadForm.addEventListener("submit", handleUpload);
startWebcamButton.addEventListener("click", startWebcam);
stopWebcamButton.addEventListener("click", stopWebcam);
speedMode.addEventListener("change", updateSpeedHelp);
updateSpeedHelp();

window.addEventListener("beforeunload", () => {
    stopPolling();
    stopWebcamStatsPolling();
    navigator.sendBeacon("/api/webcam/stop");
});
