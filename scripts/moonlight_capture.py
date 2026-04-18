#!/usr/bin/env python3
"""
moonlight_capture.py
Launches Moonlight (streaming from a Sunshine host) and serves the
captured frames as a local MJPEG HTTP stream on port 5001.

Usage:
  python3 moonlight_capture.py --host 10.35.43.175 --app "Desktop"

The workzone app then reads from: http://localhost:5001/stream
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

MOONLIGHT_SCRIPT = Path.home() / "moonlight-launch.sh"

_frame_lock = threading.Lock()
_current_frame: bytes = b""
_capture_running = False


# ── MJPEG HTTP handler ─────────────────────────────────────────────────────────
class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/stream", "/video_feed", "/"):
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=--frame"
            )
            self.end_headers()
            try:
                while _capture_running:
                    with _frame_lock:
                        frame = _current_frame
                    if frame:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(1 / 30)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass


# ── x11grab → JPEG frame extractor ────────────────────────────────────────────
def _capture_loop(display: str, resolution: str, fps: int):
    global _capture_running, _current_frame

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-f", "x11grab",
        "-video_size", resolution,
        "-framerate", str(fps),
        "-i", display,
        "-vf", f"fps={fps}",
        "-vcodec", "mjpeg",
        "-q:v", "5",
        "-f", "image2pipe",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    JPEG_START = b"\xff\xd8"
    JPEG_END   = b"\xff\xd9"
    buf = b""

    while _capture_running:
        chunk = proc.stdout.read(8192)
        if not chunk:
            break
        buf += chunk
        while True:
            s = buf.find(JPEG_START)
            if s == -1:
                break
            e = buf.find(JPEG_END, s + 2)
            if e == -1:
                break
            with _frame_lock:
                _current_frame = buf[s : e + 2]
            buf = buf[e + 2 :]

    proc.terminate()
    print("[MoonlightCapture] Capture loop stopped.")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    global _capture_running

    parser = argparse.ArgumentParser(description="Moonlight → MJPEG bridge")
    parser.add_argument("--host",       default="10.35.43.175")
    parser.add_argument("--app",        default="Desktop")
    parser.add_argument("--port",       type=int, default=5001)
    parser.add_argument("--resolution", default="1280x720")
    parser.add_argument("--fps",        type=int, default=30)
    parser.add_argument("--display",    default=":1")
    args = parser.parse_args()

    if not MOONLIGHT_SCRIPT.exists():
        print(f"[MoonlightCapture] ERROR: {MOONLIGHT_SCRIPT} not found.")
        sys.exit(1)

    env = os.environ.copy()
    env["__EGL_VENDOR_LIBRARY_FILENAMES"] = (
        "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
    )
    env["DISPLAY"] = args.display

    print(f"[MoonlightCapture] Launching Moonlight → {args.host}  app='{args.app}'")
    moonlight_proc = subprocess.Popen(
        [str(MOONLIGHT_SCRIPT), "stream", args.host, args.app],
        env=env,
    )

    print("[MoonlightCapture] Waiting for Moonlight to connect (5s)...")
    time.sleep(5)

    if moonlight_proc.poll() is not None:
        print("[MoonlightCapture] ERROR: Moonlight exited early. Check pairing and host IP.")
        sys.exit(1)

    _capture_running = True
    cap_thread = threading.Thread(
        target=_capture_loop,
        args=(args.display, args.resolution, args.fps),
        daemon=True,
    )
    cap_thread.start()
    print(f"[MoonlightCapture] Capturing {args.resolution}@{args.fps}fps from {args.display}")

    server = HTTPServer(("0.0.0.0", args.port), MJPEGHandler)
    print(f"[MoonlightCapture] MJPEG stream ready → http://localhost:{args.port}/stream")

    def _shutdown(sig, frame):
        global _capture_running
        print("\n[MoonlightCapture] Shutting down...")
        _capture_running = False
        moonlight_proc.terminate()
        server.server_close()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.serve_forever()


if __name__ == "__main__":
    main()
