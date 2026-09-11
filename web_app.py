import argparse
import io
import os
import sys
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import urllib.parse
import threading
import cv2
import numpy as np

from main import DomainExpansionApp
from utils.demo_feed import SyntheticDemoCamera


class CursedARWebServer:
    def __init__(self, port: int = 8080):
        self.port = port
        self.app = DomainExpansionApp(
            demo_mode=True,
            theme="malevolent_shrine",
            headless=True,
        )
        self.demo_cam = SyntheticDemoCamera(width=640, height=480)
        self.lock = threading.Lock()
        self.last_action = None

    def process_image(self, input_bgr: np.ndarray, action: str = "", theme: str = "") -> np.ndarray:
        with self.lock:
            if theme and theme != self.app.theme_name:
                self.app.set_theme(theme)

            if action == "expand":
                self.app.trigger_domain()
            elif action == "reset":
                self.app.reset_domain()

            return self.app.process_frame(input_bgr)

    def process_demo_frame(self, action: str = "", theme: str = "") -> np.ndarray:
        with self.lock:
            if theme and theme != self.app.theme_name:
                self.app.set_theme(theme)

            if action == "expand":
                self.app.trigger_domain()
            elif action == "reset":
                self.app.reset_domain()

            ret, demo_raw = self.demo_cam.read()
            if not ret or demo_raw is None:
                self.demo_cam = SyntheticDemoCamera(width=640, height=480)
                _, demo_raw = self.demo_cam.read()

            return self.app.process_frame(demo_raw)


SERVER_INSTANCE = None


class ARStreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress noisy request logs to keep terminal readable
        return

    def do_GET(self):
        global SERVER_INSTANCE
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DomainVision - JJK AR Filter</title>
    <style>
        :root {
            --bg: #09060f;
            --card-bg: rgba(22, 14, 38, 0.85);
            --primary: #9d4edd;
            --primary-glow: #c77dff;
            --accent-blue: #00f0ff;
            --accent-red: #ff0055;
            --border: rgba(199, 125, 255, 0.25);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: var(--bg);
            color: #f1edfa;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans CJK JP", sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 24px 16px;
            min-height: 100vh;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        h1 {
            font-size: 32px;
            font-weight: 900;
            background: linear-gradient(135deg, #e0aaff, #c77dff, #00f0ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 2px;
            text-shadow: 0 0 30px rgba(199, 125, 255, 0.4);
        }
        .subtitle {
            font-size: 14px;
            color: #a497be;
            margin-top: 4px;
        }
        .main-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 16px 40px rgba(0,0,0,0.6), 0 0 30px rgba(157, 78, 221, 0.2);
            backdrop-filter: blur(12px);
            max-width: 680px;
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .source-tabs {
            display: flex;
            background: rgba(10, 6, 18, 0.8);
            border-radius: 10px;
            padding: 4px;
            margin-bottom: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            width: 100%;
            max-width: 420px;
        }
        .tab-btn {
            flex: 1;
            padding: 10px 16px;
            border: none;
            background: transparent;
            color: #a497be;
            font-size: 13px;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .tab-btn.active {
            background: linear-gradient(135deg, #7b2cbf, #9d4edd);
            color: #fff;
            box-shadow: 0 4px 12px rgba(157, 78, 221, 0.4);
        }
        .viewport {
            position: relative;
            width: 640px;
            height: 480px;
            max-width: 100%;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.15);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        #output-canvas {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }
        #hidden-video {
            display: none;
        }
        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 18px;
            justify-content: center;
            width: 100%;
        }
        .btn {
            padding: 12px 18px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            color: #fff;
            background: #1d142d;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4);
        }
        .btn-expand {
            background: linear-gradient(135deg, #7209b7, #b5179e);
            border: none;
            box-shadow: 0 4px 16px rgba(181, 23, 158, 0.5);
        }
        .btn-expand:hover {
            background: linear-gradient(135deg, #9d4edd, #f72585);
            box-shadow: 0 6px 22px rgba(247, 37, 133, 0.6);
        }
        .btn-sukuna {
            background: linear-gradient(135deg, #990000, #d90429);
            border: 1px solid rgba(255, 60, 60, 0.4);
            box-shadow: 0 4px 16px rgba(217, 4, 41, 0.4);
        }
        .btn-sukuna:hover {
            background: linear-gradient(135deg, #b00, #ef233c);
            box-shadow: 0 6px 20px rgba(255, 40, 60, 0.6);
        }
        .btn-theme.active {
            outline: 2px solid #fff;
            outline-offset: 2px;
            box-shadow: 0 0 16px rgba(255, 255, 255, 0.6);
        }
        .btn-reset {
            background: #2b1f41;
            border-color: rgba(255, 255, 255, 0.2);
        }
        .status-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            width: 100%;
            margin-top: 14px;
            font-size: 12px;
            color: #9d8db8;
        }
        .status-badge {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 8px #10b981;
        }
        .camera-msg {
            position: absolute;
            color: #fff;
            text-align: center;
            padding: 16px;
            background: rgba(15, 10, 25, 0.9);
            border-radius: 8px;
            border: 1px solid var(--border);
            display: none;
            max-width: 80%;
        }
        .mudra-section {
            margin-top: 20px;
            max-width: 680px;
            width: 100%;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        .mudra-card {
            background: rgba(20, 14, 32, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 14px;
            transition: all 0.25s ease;
        }
        .mudra-card.active-card {
            border-color: #c77dff;
            background: rgba(35, 20, 55, 0.95);
            box-shadow: 0 8px 24px rgba(157, 78, 221, 0.25);
        }
        .mudra-title {
            font-size: 14px;
            font-weight: 800;
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 6px;
        }
        .mudra-desc {
            font-size: 12px;
            color: #b7a9ce;
            line-height: 1.45;
        }
        .mudra-tip {
            font-size: 11px;
            color: #00f0ff;
            margin-top: 6px;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>領域展開 • DOMAIN EXPANSION</h1>
        <div class="subtitle">DomainVision 2.0 — Canonical Finger Tracking & Layered AR Filter</div>
    </div>

    <div class="main-card">
        <div class="source-tabs">
            <button class="tab-btn active" id="tab-webcam" onclick="setMode('webcam')">📷 My Local Webcam</button>
            <button class="tab-btn" id="tab-demo" onclick="setMode('demo')">🤖 Animated Demo Feed</button>
        </div>

        <div class="viewport">
            <img id="output-canvas" src="" alt="Domain Expansion Feed">
            <div id="camera-notice" class="camera-msg">
                <strong>Requesting camera permission...</strong><br>
                <span style="font-size: 12px; color: #bbb;">Please allow camera access in your browser to use your own webcam over SSH.</span>
            </div>
            <video id="hidden-video" playsinline autoplay muted></video>
        </div>

        <div class="controls">
            <button class="btn btn-expand" onclick="triggerAction('expand')">⚡ 領域展開 (Expand)</button>
            <button class="btn btn-theme btn-sukuna active" id="btn-sukuna" onclick="setTheme('malevolent_shrine')">🩸 伏魔御廚子 (Sukuna)</button>
            <button class="btn btn-theme" id="btn-void" onclick="setTheme('infinite_void')">🌌 無量空処 (Infinite Void)</button>
            <button class="btn btn-reset" onclick="triggerAction('reset')">🔄 Reset Domain</button>
            <button class="btn" onclick="saveSnapshot()">📸 Snapshot</button>
        </div>

        <div class="status-row">
            <div class="status-badge">
                <div class="dot" id="stream-dot"></div>
                <span id="status-text">Connecting to Python pipeline...</span>
            </div>
            <div id="fps-counter">FPS: --</div>
        </div>
    </div>

    <div class="mudra-section">
        <div class="mudra-card active-card" id="card-sukuna">
            <div class="mudra-title" style="color: #ff4d6d;">🩸 閻魔天印 (Sukuna)</div>
            <div class="mudra-desc">
                Bring both palms together at chest level. Keep <strong>thumbs upright</strong>, touch <strong>index fingertips</strong>, and curl ring & pinky fingers inward.
            </div>
            <div class="mudra-tip">⏱️ Hold for 12 frames to charge and trigger Malevolent Shrine!</div>
        </div>
        <div class="mudra-card" id="card-void">
            <div class="mudra-title" style="color: #c77dff;">🌌 帝釈天印 (Gojo)</div>
            <div class="mudra-desc">
                Single hand raised to eye level. <strong>Cross your middle finger over your index finger</strong> with ring & pinky fingers curled down by thumb.
            </div>
            <div class="mudra-tip">⏱️ Hold for 12 frames to trigger Infinite Void!</div>
        </div>
    </div>

    <canvas id="offscreen-canvas" width="640" height="480" style="display: none;"></canvas>

    <script>
        let mode = 'webcam'; // 'webcam' or 'demo'
        let currentTheme = 'malevolent_shrine';
        let pendingAction = '';
        let isProcessing = false;
        let videoStream = null;

        const video = document.getElementById('hidden-video');
        const outputImg = document.getElementById('output-canvas');
        const offCanvas = document.getElementById('offscreen-canvas');
        const offCtx = offCanvas.getContext('2d');
        const cameraNotice = document.getElementById('camera-notice');
        const statusText = document.getElementById('status-text');
        const fpsCounter = document.getElementById('fps-counter');

        let frameCount = 0;
        let lastFpsTime = performance.now();

        async function initCamera() {
            cameraNotice.style.display = 'block';
            try {
                videoStream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }
                });
                video.srcObject = videoStream;
                await video.play();
                cameraNotice.style.display = 'none';
                statusText.innerText = "Live Webcam active. Bring palms together in Enma-ten mudra!";
            } catch (err) {
                console.warn("Could not access local webcam:", err);
                cameraNotice.innerHTML = "<strong>Local webcam not accessible or denied.</strong><br>Switching automatically to Synthetic Demo Feed.";
                setTimeout(() => {
                    cameraNotice.style.display = 'none';
                    setMode('demo');
                }, 2000);
            }
        }

        function setMode(newMode) {
            mode = newMode;
            document.getElementById('tab-webcam').classList.toggle('active', mode === 'webcam');
            document.getElementById('tab-demo').classList.toggle('active', mode === 'demo');

            if (mode === 'webcam') {
                if (!videoStream) initCamera();
                statusText.innerText = "Local Webcam active.";
            } else {
                cameraNotice.style.display = 'none';
                statusText.innerText = "Synthetic Demo Feed active (Looping canonical Enma-ten mudra).";
            }
        }

        function triggerAction(action) {
            pendingAction = action;
        }

        function setTheme(theme) {
            currentTheme = theme;
            document.getElementById('btn-sukuna').classList.toggle('active', theme === 'malevolent_shrine');
            document.getElementById('btn-void').classList.toggle('active', theme === 'infinite_void');
            document.getElementById('card-sukuna').classList.toggle('active-card', theme === 'malevolent_shrine');
            document.getElementById('card-void').classList.toggle('active-card', theme === 'infinite_void');
        }

        function saveSnapshot() {
            if (outputImg.src) {
                const a = document.createElement('a');
                a.href = outputImg.src;
                a.download = `domain_expansion_${Date.now()}.jpg`;
                a.click();
            }
        }

        async function processLoop() {
            if (!isProcessing) {
                isProcessing = true;
                const act = pendingAction;
                pendingAction = '';

                try {
                    let response;
                    if (mode === 'webcam' && video.readyState >= 2) {
                        // Capture frame from user local webcam
                        offCtx.drawImage(video, 0, 0, 640, 480);
                        const blob = await new Promise(resolve => offCanvas.toBlob(resolve, 'image/jpeg', 0.82));
                        
                        response = await fetch(`/api/process_frame?theme=${currentTheme}&action=${act}`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'image/jpeg' },
                            body: blob
                        });
                    } else {
                        // Demo mode
                        response = await fetch(`/api/demo_frame?theme=${currentTheme}&action=${act}`);
                    }

                    if (response && response.ok) {
                        const imgBlob = await response.blob();
                        const oldUrl = outputImg.src;
                        outputImg.src = URL.createObjectURL(imgBlob);
                        if (oldUrl.startsWith('blob:')) URL.revokeObjectURL(oldUrl);

                        frameCount++;
                        const now = performance.now();
                        if (now - lastFpsTime >= 1000) {
                            const fps = ((frameCount * 1000) / (now - lastFpsTime)).toFixed(1);
                            fpsCounter.innerText = `FPS: ${fps}`;
                            frameCount = 0;
                            lastFpsTime = now;
                        }
                    }
                } catch (err) {
                    // Ignore frame drop
                } finally {
                    isProcessing = false;
                }
            }
            requestAnimationFrame(processLoop);
        }

        // Start webcam on load
        initCamera();
        requestAnimationFrame(processLoop);
    </script>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))

        elif parsed.path == "/api/demo_frame":
            # Generate synthetic demo frame on server
            action = params.get("action", [""])[0]
            theme = params.get("theme", [""])[0]
            out_frame = SERVER_INSTANCE.process_demo_frame(action=action, theme=theme)
            ret, buf = cv2.imencode(".jpg", out_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

            if ret:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(buf)))
                self.end_headers()
                self.wfile.write(buf.tobytes())
            else:
                self.send_error(500)

        else:
            self.send_error(404)

    def do_POST(self):
        global SERVER_INSTANCE
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/api/process_frame":
            content_len = int(self.headers.get("Content-Length", 0))
            if content_len <= 0:
                self.send_error(400)
                return

            raw_bytes = self.rfile.read(content_len)
            nparr = np.frombuffer(raw_bytes, np.uint8)
            input_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if input_bgr is None:
                self.send_error(400)
                return

            action = params.get("action", [""])[0]
            theme = params.get("theme", [""])[0]
            out_frame = SERVER_INSTANCE.process_image(input_bgr, action=action, theme=theme)

            ret, buf = cv2.imencode(".jpg", out_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(buf)))
                self.end_headers()
                self.wfile.write(buf.tobytes())
            else:
                self.send_error(500)
        else:
            self.send_error(404)


def main():
    global SERVER_INSTANCE
    parser = argparse.ArgumentParser(description="DomainVision Web Streamer")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    args = parser.parse_args()

    SERVER_INSTANCE = CursedARWebServer(port=args.port)
    server_address = ("0.0.0.0", args.port)
    httpd = ThreadingHTTPServer(server_address, ARStreamHandler)

    print("\n=======================================================")
    print(f"  DOMAINVISION AR WEB SERVER STARTED!")
    print(f"  URL: http://localhost:{args.port}/")
    print(f"  - Use 'My Local Webcam' tab to use your laptop webcam!")
    print(f"  - Use 'Animated Demo Feed' tab to watch Gojo!")
    print(f"  - Press Ctrl+C in terminal to stop.")
    print("=======================================================\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web server...")
    finally:
        httpd.server_close()
        if hasattr(SERVER_INSTANCE.app, "cleanup"):
            SERVER_INSTANCE.app.cleanup()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
