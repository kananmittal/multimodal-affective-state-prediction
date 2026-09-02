"""
Local Music4D-style group-emotion dashboard.

Reproduces the interface of Music4D-demo-GER-dash-emo-telemetry.py — a live
composited multi-source video wall with per-source emotion overlays, plus the
/api/emotions telemetry endpoint the robot team pulls from — but sourced from
the bundled clips in Video/ rather than UDP camera feeds on ports 9000-9002,
and running on whatever accelerator is present (CUDA / MPS / CPU).

    python3 run_local_dashboard.py
    python3 run_local_dashboard.py --port 5055 --fps 12

Endpoints:
    /                -> dashboard page
    /video_feed      -> MJPEG stream of the composited wall
    /api/emotions    -> JSON telemetry (entities, dominant emotion, colour)
"""

import argparse
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify

from run_local import DEFAULT_SOURCES, infer, load_model, pick_device

HERE = Path(__file__).resolve().parent

# Palette carried over from the original dashboard (BGR for cv2).
EMO_COLORS_BGR = {
    "joy":      (0, 220, 255),
    "anger":    (40, 40, 255),
    "surprise": (255, 100, 200),
    "sadness":  (255, 140, 80),
    "fear":     (0, 140, 255),
    "disgust":  (80, 200, 120),
    "boredom":  (150, 150, 150),
    "neutral":  (200, 200, 200),
    "default":  (255, 255, 255),
}
VALID_EMOTIONS = ["joy", "anger", "fear", "disgust", "surprise", "sadness",
                  "boredom", "neutral"]

PANEL_W, PANEL_H = 640, 360
HEADER_H = 96

app = Flask(__name__)


class SharedState:
    def __init__(self, names):
        self.lock = threading.Lock()
        self.frames = {n: None for n in names}
        self.emotions = {n: "WAITING" for n in names}
        self.raw = {n: "" for n in names}
        self.latency = {n: None for n in names}
        self.output_jpeg = None
        self.running = True
        self.cycles = 0


state = None


def parse_emotions(text):
    """Pull known emotion words out of a free-form model reply."""
    low = text.lower()
    found = [e for e in VALID_EMOTIONS if e in low]
    return found[:2]


def capture_worker(name, path, fps):
    """Loop a clip into shared state at a steady frame rate."""
    cap = cv2.VideoCapture(str(path))
    delay = 1.0 / max(1, fps)
    while state.running:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        with state.lock:
            state.frames[name] = frame
        time.sleep(delay)
    cap.release()


def analysis_worker(model, processor, device, names):
    """Round-robin one inference at a time so the wall updates every ~13s
    rather than stalling for a full three-source sweep."""
    print("Analysis thread started", flush=True)
    i = 0
    while state.running:
        name = names[i % len(names)]
        i += 1
        with state.lock:
            frame = state.frames[name]
            frame = None if frame is None else frame.copy()
        if frame is None:
            time.sleep(0.2)
            continue
        try:
            t0 = time.time()
            text = infer(model, processor, device, frame)
            dt = time.time() - t0
            emos = parse_emotions(text)
            with state.lock:
                state.raw[name] = text
                state.emotions[name] = ", ".join(emos) if emos else "unparsed"
                state.latency[name] = round(dt, 1)
                state.cycles += 1
            print(f"[{name:9s}] {dt:5.1f}s -> {text}", flush=True)
        except Exception as e:  # keep the wall alive on a bad frame
            print(f"inference error on {name}: {e}", flush=True)
            time.sleep(0.5)
    print("Analysis thread stopped", flush=True)


def dominant_emotion(emotions):
    words = []
    for v in emotions.values():
        if v not in ("WAITING", "unparsed"):
            words += [w.strip() for w in v.split(",") if w.strip()]
    return Counter(words).most_common(1)[0][0] if words else "neutral"


def draw_panel(frame, name, emotion, latency):
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    if frame is not None:
        h, w = frame.shape[:2]
        scale = min(PANEL_W / w, PANEL_H / h)
        rw, rh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (rw, rh))
        y0, x0 = (PANEL_H - rh) // 2, (PANEL_W - rw) // 2
        panel[y0:y0 + rh, x0:x0 + rw] = resized
    else:
        cv2.putText(panel, "NO SIGNAL", (PANEL_W // 2 - 90, PANEL_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 60), 2, cv2.LINE_AA)

    key = emotion.split(",")[0].strip()
    col = EMO_COLORS_BGR.get(key, EMO_COLORS_BGR["default"])

    cv2.rectangle(panel, (0, 0), (PANEL_W, 30), (0, 0, 0), -1)
    cv2.putText(panel, name, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    if latency is not None:
        tag = f"{latency}s"
        cv2.putText(panel, tag, (PANEL_W - 70, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (120, 120, 120), 1, cv2.LINE_AA)

    cv2.rectangle(panel, (0, PANEL_H - 46), (PANEL_W, PANEL_H), (0, 0, 0), -1)
    cv2.putText(panel, emotion.upper()[:34], (12, PANEL_H - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    cv2.rectangle(panel, (0, PANEL_H - 46), (PANEL_W, PANEL_H - 43), col, -1)
    cv2.rectangle(panel, (0, 0), (PANEL_W - 1, PANEL_H - 1), (40, 40, 40), 1)
    return panel


def compose_worker(names, fps):
    delay = 1.0 / max(1, fps)
    while state.running:
        with state.lock:
            frames = {n: (None if state.frames[n] is None else state.frames[n].copy())
                      for n in names}
            emotions = state.emotions.copy()
            latency = state.latency.copy()
            cycles = state.cycles
        panels = [draw_panel(frames[n], n, emotions[n], latency[n]) for n in names]
        wall = np.hstack(panels)

        dom = dominant_emotion(emotions)
        col = EMO_COLORS_BGR.get(dom, EMO_COLORS_BGR["default"])
        header = np.zeros((HEADER_H, wall.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, "MUSIC4D  /  GROUP EMOTION MONITOR", (18, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2, cv2.LINE_AA)
        cv2.putText(header, f"DOMINANT: {dom.upper()}", (18, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2, cv2.LINE_AA)
        cv2.putText(header, f"inferences: {cycles}", (wall.shape[1] - 220, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.circle(header, (wall.shape[1] - 40, 64), 10, col, -1)

        canvas = np.vstack([header, wall])
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with state.lock:
                state.output_jpeg = buf.tobytes()
        time.sleep(delay)


@app.route("/")
def index():
    return """
<html><head><title>Music4D Monitor</title>
<style>
  body{background:#0a0a0a;margin:0;padding:0;height:100vh;width:100vw;
       display:flex;flex-direction:column;justify-content:center;align-items:center;
       overflow:hidden;font-family:-apple-system,system-ui,sans-serif;color:#888}
  img{max-width:100%;max-height:92vh;object-fit:contain}
  a{color:#666;font-size:12px;text-decoration:none;padding:8px}
</style></head>
<body>
  <img src="/video_feed">
  <div><a href="/api/emotions">/api/emotions</a></div>
</body></html>
"""


def generate():
    while True:
        with state.lock:
            frame = state.output_jpeg
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.05)


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/emotions")
def api_emotions():
    with state.lock:
        cur = state.emotions.copy()
        raw = state.raw.copy()
        lat = state.latency.copy()
    dom = dominant_emotion(cur)
    b, g, r = EMO_COLORS_BGR.get(dom, EMO_COLORS_BGR["default"])
    return jsonify({
        "timestamp": time.time(),
        "entities": cur,
        "raw_model_output": raw,
        "latency_seconds": lat,
        "dominant_emotion": dom,
        "dominant_color_rgb": [r, g, b],
        "dominant_color_hex": f"#{r:02x}{g:02x}{b:02x}",
    })


def main():
    global state
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5055,
                    help="macOS holds 5000 for AirPlay; default 5055")
    ap.add_argument("--fps", type=int, default=12, help="capture/compose rate")
    ap.add_argument("--max-pixels", type=int, default=768)
    args = ap.parse_args()

    sources = {}
    for name, rel in DEFAULT_SOURCES.items():
        p = HERE / rel
        if p.exists():
            sources[name] = p
        else:
            print(f"skipping {name}: {p} missing")
    if not sources:
        raise SystemExit("no video sources found")

    names = list(sources)
    state = SharedState(names)

    device, dtype = pick_device()
    model, processor = load_model(device, dtype, args.max_pixels)

    threads = [threading.Thread(target=capture_worker, args=(n, p, args.fps),
                                daemon=True) for n, p in sources.items()]
    threads.append(threading.Thread(target=compose_worker, args=(names, args.fps),
                                    daemon=True))
    threads.append(threading.Thread(target=analysis_worker,
                                    args=(model, processor, device, names),
                                    daemon=True))
    for t in threads:
        t.start()

    print(f"\n  DASHBOARD : http://127.0.0.1:{args.port}")
    print(f"  TELEMETRY : http://127.0.0.1:{args.port}/api/emotions\n", flush=True)
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
    finally:
        state.running = False


if __name__ == "__main__":
    main()
