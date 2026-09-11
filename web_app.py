import argparse
import io
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import cv2
import numpy as np

from main import DomainExpansionApp


# Global app instance and frame buffer
current_frame_bytes = None
frame_lock = threading.Lock()
app_instance = None


class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global current_frame_bytes, app_instance

        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Domain Expansion AR - Live Stream</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: #0d0a14;
            color: #f0edf6;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 24px 16px;
            min-height: 100vh;
        }
        h1 {
            font-size: 28px;
            font-weight: 800;
            background: linear-gradient(135deg, #bd34fe, #41d1ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
            letter-spacing: 1px;
        }
        .subtitle {
            font-size: 14px;
            color: #8b80a4;
            margin-bottom: 20px;
        }
        .stream-container {
            position: relative;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 12px 36px rgba(189, 52, 254, 0.25), 0 0 0 1px rgba(255, 255, 255, 0.1);
            max-width: 640px;
            width: 100%;
        }
        .stream-container img {
            display: block;
            width: 100%;
            height: auto;
        }
        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 24px;
            justify-content: center;
            max-width: 640px;
        }
        button {
            background: #1e172e;
            color: #fff;
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        button:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.4);
        }
        .btn-expand {
            background: linear-gradient(135deg, #7b1fa2, #ba68c8);
            border: none;
            box-shadow: 0 4px 20px rgba(186, 104, 200, 0.4);
        }
        .btn-expand:hover {
            background: linear-gradient(135deg, #8e24aa, #ce93d8);
            box-shadow: 0 6px 24px rgba(186, 104, 200, 0.6);
        }
        .btn-sukuna {
            background: linear-gradient(135deg, #b71c1c, #e53935);
            border: none;
            box-shadow: 0 4px 20px rgba(229, 57, 53, 0.4);
        }
        .btn-sukuna:hover {
            background: linear-gradient(135deg, #c62828, #ef5350);
            box-shadow: 0 6px 24px rgba(229, 57, 53, 0.6);
        }
        .btn-reset {
            background: #2a2238;
        }
        .status-badge {
            margin-top: 16px;
            font-size: 13px;
            color: #10b981;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(0.9); opacity: 0.8; }
            50% { transform: scale(1.3); opacity: 1; }
            100% { transform: scale(0.9); opacity: 0.8; }
        }
    </style>
</head>
<body>
    <h1>領域展開 • DOMAIN EXPANSION</h1>
    <div class="subtitle">Real-Time OpenCV + MediaPipe AR Stream</div>

    <div class="stream-container">
        <img src="/video_feed" alt="Domain Expansion Stream">
    </div>

    <div class="controls">
        <button class="btn-expand" onclick="sendAction('/action?cmd=expand')">⚡ 領域展開 (Expand)</button>
        <button onclick="sendAction('/action?cmd=infinite_void')">🌌 無量空処 (Infinite Void)</button>
        <button class="btn-sukuna" onclick="sendAction('/action?cmd=malevolent_shrine')">🩸 伏魔御廚子 (Sukuna)</button>
        <button class="btn-reset" onclick="sendAction('/action?cmd=reset')">🔄 Collapse / Reset</button>
    </div>

    <div class="status-badge">
        <div class="status-dot"></div> Live WebSocket / MJPEG Stream Active
    </div>

    <script>
        function sendAction(url) {
            fetch(url).catch(console.error);
        }
    </script>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))

        elif self.path == "/video_feed":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()

            try:
                while True:
                    with frame_lock:
                        frame_data = current_frame_bytes

                    if frame_data is not None:
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame_data)))
                        self.end_headers()
                        self.wfile.write(frame_data)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.033)  # ~30 FPS
            except Exception:
                pass

        elif self.path.startswith("/action"):
            cmd = self.path.split("cmd=")[-1] if "cmd=" in self.path else ""
            if app_instance:
                if cmd == "expand":
                    app_instance.trigger_domain()
                elif cmd == "reset":
                    app_instance.reset_domain()
                elif cmd == "infinite_void":
                    app_instance.set_theme("infinite_void")
                elif cmd == "malevolent_shrine":
                    app_instance.set_theme("malevolent_shrine")

            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_error(404)


def video_processing_thread(app: DomainExpansionApp):
    global current_frame_bytes
    print("Video processing thread started.")
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

    while app.cap.isOpened():
        ret, raw_frame = app.cap.read()
        if not ret or raw_frame is None:
            if app.is_demo:
                continue
            else:
                break

        output_frame = app.process_frame(raw_frame)
        ret, buffer = cv2.imencode(".jpg", output_frame, encode_param)
        if ret:
            with frame_lock:
                current_frame_bytes = buffer.tobytes()

        time.sleep(0.015)


def main():
    global app_instance
    parser = argparse.ArgumentParser(description="Live Web Streamer for Domain Expansion AR")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--demo", action="store_true", help="Force demo mode")
    parser.add_argument("--theme", type=str, default="infinite_void", help="Theme")
    args = parser.parse_args()

    app_instance = DomainExpansionApp(
        camera_idx=args.camera,
        demo_mode=args.demo,
        theme=args.theme,
        headless=True,
    )

    t = threading.Thread(target=video_processing_thread, args=(app_instance,), daemon=True)
    t.start()

    server_address = ("0.0.0.0", args.port)
    httpd = HTTPServer(server_address, StreamingHandler)
    print("\n=======================================================")
    print(f"  DOMAIN EXPANSION AR - LIVE WEB STREAM ACTIVE!")
    print(f"  URL: http://localhost:{args.port}/")
    print(f"  (In VS Code Remote-SSH, open the Ports tab to view)")
    print("=======================================================\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Stopping web server...")
    finally:
        httpd.server_close()
        app_instance.cleanup()


if __name__ == "__main__":
    main()
