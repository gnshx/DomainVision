import argparse
import io
import os
import sys
import time
from typing import Optional
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import urllib.parse
import threading
import cv2
import numpy as np

from main import DomainExpansionApp
from utils.demo_feed import SyntheticDemoCamera


class CursedARWebServer:
    def __init__(self, port: int = 8080, width: int = 640, height: int = 360):
        self.port = port
        self.width = width
        self.height = height
        self.app = DomainExpansionApp(
            demo_mode=True,
            theme="malevolent_shrine",
            width=self.width,
            height=self.height,
            headless=True,
        )
        self.demo_cam = SyntheticDemoCamera(width=self.width, height=self.height)
        self.lock = threading.Lock()
        self.last_action = None

    def process_image(self, input_bgr: np.ndarray, action: str = "") -> np.ndarray:
        with self.lock:
            if action == "expand":
                self.app.trigger_domain()
            elif action == "reset":
                self.app.reset_domain()

            if input_bgr.shape[1] != self.width or input_bgr.shape[0] != self.height:
                input_bgr = cv2.resize(input_bgr, (self.width, self.height))

            return self.app.process_frame(input_bgr)

    def process_demo_frame(self, user_cam_bgr: Optional[np.ndarray] = None, action: str = "") -> np.ndarray:
        with self.lock:
            if action == "expand":
                self.app.trigger_domain()
            elif action == "reset":
                self.app.reset_domain()

            if user_cam_bgr is not None:
                if user_cam_bgr.shape[1] != self.width or user_cam_bgr.shape[0] != self.height:
                    user_cam_bgr = cv2.resize(user_cam_bgr, (self.width, self.height))
                user_tracking = self.app.tracker.process(user_cam_bgr)
                self.app._cached_webcam_hands = user_tracking.get("hands", [])

            ret, demo_raw = self.demo_cam.read()
            if not ret or demo_raw is None:
                self.demo_cam = SyntheticDemoCamera(width=self.width, height=self.height)
                _, demo_raw = self.demo_cam.read()

            self.demo_cam.set_state(self.app.state)
            return self.app.process_frame(demo_raw)


SERVER_INSTANCE = None


class ARStreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress noisy HTTP logs
        return

    def _send_telemetry_headers(self):
        global SERVER_INSTANCE
        app = SERVER_INSTANCE.app
        rec = app.gesture_recognizer

        fps_val = f"{app.profiler.fps:.1f}"
        theme_val = app.theme_name
        state_val = app.state
        sign_val = rec.detected_sign_name or ""
        match_val = str(rec.last_match_pct)
        sukuna_match = str(getattr(rec, "last_sukuna_pct", 0))
        gojo_match = str(getattr(rec, "last_gojo_pct", 0))

        self.send_header("X-FPS", fps_val)
        self.send_header("X-Theme", theme_val)
        self.send_header("X-State", state_val)
        self.send_header("X-Sign", urllib.parse.quote(sign_val))
        self.send_header("X-Match", match_val)
        self.send_header("X-Sukuna-Match", sukuna_match)
        self.send_header("X-Gojo-Match", gojo_match)
        self.send_header("Access-Control-Expose-Headers", "X-FPS, X-Theme, X-State, X-Sign, X-Match, X-Sukuna-Match, X-Gojo-Match")

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
    <title>DomainVision - JJK AR Filter (領域展開)</title>
    <style>
        :root {
            --bg: #07040d;
            --card-bg: rgba(18, 12, 30, 0.90);
            --primary: #9d4edd;
            --primary-glow: #c77dff;
            --accent-cyan: #00f0ff;
            --accent-red: #ff0055;
            --border: rgba(199, 125, 255, 0.28);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: var(--bg);
            color: #f1edfa;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans CJK JP", sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px 16px;
            min-height: 100vh;
        }
        .header {
            text-align: center;
            margin-bottom: 16px;
        }
        h1 {
            font-size: 28px;
            font-weight: 900;
            background: linear-gradient(135deg, #e0aaff, #c77dff, #00f0ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 2px;
            text-shadow: 0 0 30px rgba(199, 125, 255, 0.4);
        }
        .subtitle {
            font-size: 13px;
            color: #b3a4cb;
            margin-top: 4px;
        }
        .main-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 16px 40px rgba(0,0,0,0.7), 0 0 30px rgba(157, 78, 221, 0.18);
            backdrop-filter: blur(14px);
            max-width: 820px;
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .source-tabs {
            display: flex;
            background: rgba(10, 6, 18, 0.85);
            border-radius: 10px;
            padding: 4px;
            margin-bottom: 14px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            width: 100%;
            max-width: 440px;
        }
        .tab-btn {
            flex: 1;
            padding: 9px 14px;
            border: none;
            background: transparent;
            color: #a497be;
            font-size: 13px;
            font-weight: 700;
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
            width: 100%;
            max-width: 780px;
            aspect-ratio: 16 / 9;
            background: #020105;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(199, 125, 255, 0.35);
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.8);
        }
        #output-canvas {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }
        #hidden-video {
            display: none;
        }
        /* Prominent Floating Telemetry Overlay (Never overshadowed) */
        .hud-overlay {
            position: absolute;
            top: 10px;
            left: 10px;
            right: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            pointer-events: none;
            z-index: 60;
        }
        .hud-pill {
            background: rgba(12, 8, 22, 0.92);
            border: 1px solid rgba(0, 240, 255, 0.6);
            border-radius: 20px;
            padding: 5px 12px;
            font-size: 11px;
            font-weight: 800;
            color: #00f0ff;
            letter-spacing: 0.5px;
            backdrop-filter: blur(8px);
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.7);
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .hud-pill.hud-fps {
            border-color: rgba(57, 255, 20, 0.7);
            color: #39ff14;
        }
        .hud-pill.hud-domain {
            border-color: rgba(255, 77, 109, 0.7);
            color: #ff758f;
        }
        .pulse-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #39ff14;
            box-shadow: 0 0 8px #39ff14;
            display: inline-block;
        }

        /* Picture-in-Picture Webcam Box at Bottom (During Animated Demo Feed) */
        .pip-container {
            position: absolute;
            bottom: 12px;
            right: 12px;
            width: 180px;
            height: 115px;
            border-radius: 10px;
            overflow: hidden;
            border: 2px solid #00f0ff;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.85), 0 0 15px rgba(0, 240, 255, 0.35);
            background: #000;
            z-index: 50;
            display: none;
            flex-direction: column;
        }
        .pip-header {
            background: rgba(14, 10, 24, 0.94);
            padding: 4px 8px;
            font-size: 10px;
            font-weight: 800;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 6px;
            letter-spacing: 0.5px;
            border-bottom: 1px solid rgba(0, 240, 255, 0.4);
        }
        .pip-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #ff0055;
            box-shadow: 0 0 6px #ff0055;
        }
        #pip-video {
            width: 100%;
            height: calc(100% - 20px);
            object-fit: cover;
            transform: scaleX(-1);
        }

        .camera-msg {
            position: absolute;
            color: #fff;
            text-align: center;
            padding: 16px;
            background: rgba(15, 10, 25, 0.92);
            border-radius: 8px;
            border: 1px solid var(--border);
            display: none;
            max-width: 80%;
            z-index: 30;
        }
        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 16px;
            justify-content: center;
            width: 100%;
        }
        .btn {
            padding: 10px 18px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            font-size: 13px;
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
        .btn-reset {
            background: #2b1f41;
            border-color: rgba(255, 255, 255, 0.2);
        }
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            width: 100%;
            margin-top: 14px;
            font-size: 12px;
            color: #b7a9ce;
            padding: 6px 8px;
            background: rgba(10, 6, 18, 0.5);
            border-radius: 8px;
        }
        .status-badge {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .auto-tag {
            background: rgba(0, 240, 255, 0.15);
            border: 1px solid rgba(0, 240, 255, 0.5);
            color: #00f0ff;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
        }
        .mudra-section {
            margin-top: 16px;
            max-width: 820px;
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
            position: relative;
        }
        .mudra-card.active-card {
            border-color: #00f0ff;
            background: rgba(30, 20, 48, 0.95);
            box-shadow: 0 8px 24px rgba(0, 240, 255, 0.25);
        }
        .mudra-title {
            font-size: 14px;
            font-weight: 800;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 6px;
        }
        .mudra-desc {
            font-size: 12px;
            color: #b7a9ce;
            line-height: 1.45;
        }
        .match-badge {
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 10px;
            font-weight: 800;
            background: rgba(255, 255, 255, 0.1);
            color: #ddd;
        }
        .match-badge.matched {
            background: #10b981;
            color: #fff;
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.6);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>領域展開 • DOMAIN EXPANSION</h1>
        <div class="subtitle">Automatic Finger-Symbol Recognition & High-FPS AR Filter</div>
    </div>

    <div class="main-card">
        <div class="source-tabs">
            <button class="tab-btn active" id="tab-webcam" onclick="setMode('webcam')">📷 My Local Webcam</button>
            <button class="tab-btn" id="tab-demo" onclick="setMode('demo')">🤖 Animated Demo Feed</button>
        </div>

        <div class="viewport">
            <!-- Prominent Telemetry Overlay (Always 100% visible, never cut off) -->
            <div class="hud-overlay">
                <div class="hud-pill hud-fps">
                    <span class="pulse-dot"></span>
                    <span id="telemetry-fps">FPS: 30.0</span>
                </div>
                <div class="hud-pill hud-domain">
                    <span id="telemetry-domain">DOMAIN: MALEVOLENT SHRINE</span>
                </div>
            </div>

            <canvas id="output-canvas" width="640" height="360"></canvas>

            <!-- Picture-in-Picture Live Webcam Box at Bottom (During Animated Demo Feed) -->
            <div id="pip-container" class="pip-container">
                <div class="pip-header">
                    <span class="pip-dot"></span>
                    <span>Your Webcam (Live)</span>
                </div>
                <video id="pip-video" playsinline autoplay muted></video>
            </div>

            <div id="camera-notice" class="camera-msg">
                <strong>Requesting camera permission...</strong><br>
                <span style="font-size: 12px; color: #bbb;">Please allow webcam access to track your finger signs!</span>
            </div>
            <video id="hidden-video" playsinline autoplay muted></video>
        </div>

        <div class="controls">
            <button class="btn btn-expand" onclick="triggerAction('expand')">⚡ Force Expand (領域展開)</button>
            <button class="btn btn-reset" onclick="triggerAction('reset')">🔄 Collapse / Reset</button>
            <button class="btn" onclick="saveSnapshot()">📸 Snapshot</button>
        </div>

        <div class="status-bar">
            <div class="status-badge">
                <span class="auto-tag">⚡ AUTO-THEME</span>
                <span id="status-text">Form Sukuna or Gojo hand sign with fingers</span>
            </div>
            <div id="fps-stat" style="font-weight: 700; color: #39ff14;">PIPELINE: ACTIVE</div>
        </div>
    </div>

    <!-- Dual Mudra Finger Signs Section (Both evaluated automatically) -->
    <div class="mudra-section">
        <div class="mudra-card active-card" id="card-sukuna">
            <div class="mudra-title" style="color: #ff4d6d;">
                <span>🩸 閻魔天印 (Sukuna)</span>
                <span class="match-badge" id="badge-sukuna">0%</span>
            </div>
            <div class="mudra-desc">
                Clasp both palms together in front of chest. Keep <strong>thumbs upright</strong>, touch <strong>index fingertips</strong>, curl ring & pinky fingers inward.
            </div>
        </div>

        <div class="mudra-card" id="card-void">
            <div class="mudra-title" style="color: #c77dff;">
                <span>🌌 帝釈天印 (Gojo)</span>
                <span class="match-badge" id="badge-void">0%</span>
            </div>
            <div class="mudra-desc">
                Single hand raised vertically. <strong>Cross middle finger over index finger</strong> with ring and pinky fingers folded tight into palm.
            </div>
        </div>
    </div>

    <canvas id="offscreen-canvas" width="640" height="360" style="display: none;"></canvas>

    <script>
        let mode = 'webcam'; // 'webcam' or 'demo'
        let pendingAction = '';
        let isProcessing = false;
        let videoStream = null;

        const video = document.getElementById('hidden-video');
        const pipVideo = document.getElementById('pip-video');
        const pipContainer = document.getElementById('pip-container');
        const outputCanvas = document.getElementById('output-canvas');
        const outCtx = outputCanvas.getContext('2d');
        const offCanvas = document.getElementById('offscreen-canvas');
        const offCtx = offCanvas.getContext('2d');
        const cameraNotice = document.getElementById('camera-notice');
        const statusText = document.getElementById('status-text');
        const telemetryFps = document.getElementById('telemetry-fps');
        const telemetryDomain = document.getElementById('telemetry-domain');
        const cardSukuna = document.getElementById('card-sukuna');
        const cardVoid = document.getElementById('card-void');
        const badgeSukuna = document.getElementById('badge-sukuna');
        const badgeVoid = document.getElementById('badge-void');

        let frameCount = 0;
        let lastFpsTime = performance.now();

        function drawInitialPlaceholder() {
            outCtx.fillStyle = "#0c0816";
            outCtx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
            outCtx.font = "bold 15px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
            outCtx.fillStyle = "#c77dff";
            outCtx.textAlign = "center";
            outCtx.fillText("領域展開 • INITIALIZING DOMAIN STREAM", outputCanvas.width / 2, outputCanvas.height / 2 - 8);
            outCtx.font = "12px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
            outCtx.fillStyle = "#00f0ff";
            outCtx.fillText("Connecting camera & neural vision pipeline...", outputCanvas.width / 2, outputCanvas.height / 2 + 16);
        }
        drawInitialPlaceholder();

        async function initCamera() {
            cameraNotice.style.display = 'block';
            try {
                videoStream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 640 }, height: { ideal: 360 }, facingMode: "user" }
                });
                video.srcObject = videoStream;
                pipVideo.srcObject = videoStream;
                await video.play();
                await pipVideo.play();
                cameraNotice.style.display = 'none';
                statusText.innerText = "Webcam active. Form Sukuna or Gojo hand sign!";
            } catch (err) {
                console.warn("Webcam not directly accessible:", err);
                cameraNotice.innerHTML = "<strong>Local webcam not detected or permission denied.</strong><br>Switching to Animated Demo Feed.";
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
                pipContainer.style.display = 'none';
                if (!videoStream) initCamera();
                statusText.innerText = "Local Webcam active. Make hand sign to switch domain.";
            } else {
                cameraNotice.style.display = 'none';
                // Show live webcam PiP display in bottom right corner during animated demo feed
                pipContainer.style.display = 'flex';
                statusText.innerText = "Animated Demo Feed active (Showing your webcam at bottom).";
            }
        }

        function triggerAction(action) {
            pendingAction = action;
        }

        function saveSnapshot() {
            try {
                const a = document.createElement('a');
                a.href = outputCanvas.toDataURL('image/jpeg', 0.92);
                a.download = `domain_snapshot_${Date.now()}.jpg`;
                a.click();
            } catch (err) {
                console.warn("Snapshot error:", err);
            }
        }

        async function processLoop() {
            if (!isProcessing) {
                isProcessing = true;
                const act = pendingAction;
                pendingAction = '';
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 2500);

                try {
                    let response;
                    if (mode === 'webcam' && video.readyState >= 2) {
                        offCtx.drawImage(video, 0, 0, 640, 360);
                        const blob = await new Promise(resolve => offCanvas.toBlob(resolve, 'image/jpeg', 0.75));

                        response = await fetch(`/api/process_frame?action=${act}`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'image/jpeg' },
                            body: blob,
                            signal: controller.signal
                        });
                    } else if (mode === 'demo' && videoStream && (pipVideo.readyState >= 2 || video.readyState >= 2)) {
                        // User is viewing animated demo feed with live webcam active at bottom!
                        // Send webcam frame so user's real fingers are tracked to trigger expansion!
                        const srcVid = pipVideo.readyState >= 2 ? pipVideo : video;
                        offCtx.drawImage(srcVid, 0, 0, 640, 360);
                        const blob = await new Promise(resolve => offCanvas.toBlob(resolve, 'image/jpeg', 0.75));

                        response = await fetch(`/api/demo_frame?action=${act}`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'image/jpeg' },
                            body: blob,
                            signal: controller.signal
                        });
                    } else {
                        // Demo mode without active camera
                        response = await fetch(`/api/demo_frame?action=${act}`, {
                            signal: controller.signal
                        });
                    }

                    clearTimeout(timeoutId);

                    if (response && response.ok) {
                        // Read telemetry headers from server
                        const serverFps = response.headers.get("X-FPS");
                        const serverTheme = response.headers.get("X-Theme") || "malevolent_shrine";
                        const serverSign = decodeURIComponent(response.headers.get("X-Sign") || "");
                        const serverMatch = parseInt(response.headers.get("X-Match") || "0");
                        const sukunaMatch = parseInt(response.headers.get("X-Sukuna-Match") || "0");
                        const gojoMatch = parseInt(response.headers.get("X-Gojo-Match") || "0");

                        // Update floating HUD telemetry
                        frameCount++;
                        const now = performance.now();
                        if (now - lastFpsTime >= 800) {
                            const clientFps = ((frameCount * 1000) / (now - lastFpsTime)).toFixed(1);
                            telemetryFps.innerText = `FPS: ${serverFps || clientFps}`;
                            frameCount = 0;
                            lastFpsTime = now;
                        }

                        // Real-time mudra percentage badges
                        badgeSukuna.innerText = `${sukunaMatch}%`;
                        badgeSukuna.className = `match-badge ${sukunaMatch >= 78 ? 'matched' : ''}`;
                        badgeVoid.innerText = `${gojoMatch}%`;
                        badgeVoid.className = `match-badge ${gojoMatch >= 78 ? 'matched' : ''}`;

                        // Auto-Theme & Mudra Card updates based on finger symbols
                        if (serverTheme === 'malevolent_shrine') {
                            telemetryDomain.innerText = "DOMAIN: MALEVOLENT SHRINE (Sukuna)";
                            cardSukuna.classList.add('active-card');
                            cardVoid.classList.remove('active-card');
                        } else {
                            telemetryDomain.innerText = "DOMAIN: INFINITE VOID (Gojo)";
                            cardVoid.classList.add('active-card');
                            cardSukuna.classList.remove('active-card');
                        }

                        if (serverSign) {
                            statusText.innerText = `${serverSign} (${serverMatch}%)`;
                        }

                        const imgBlob = await response.blob();
                        if (imgBlob && imgBlob.size > 0) {
                            try {
                                const bmp = await createImageBitmap(imgBlob);
                                outCtx.drawImage(bmp, 0, 0, outputCanvas.width, outputCanvas.height);
                                bmp.close();
                            } catch (decodeErr) {
                                // Frame decode drop
                            }
                        }
                    }
                } catch (err) {
                    clearTimeout(timeoutId);
                    // Network or timeout drop, continue loop smoothly
                } finally {
                    isProcessing = false;
                }
            }
            requestAnimationFrame(processLoop);
        }

        initCamera();
        requestAnimationFrame(processLoop);
    </script>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))

        elif parsed.path == "/api/demo_frame":
            action = params.get("action", [""])[0]
            try:
                out_frame = SERVER_INSTANCE.process_demo_frame(action=action)
                ret, buf = cv2.imencode(".jpg", out_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

                if ret:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(buf)))
                    self._send_telemetry_headers()
                    self.end_headers()
                    self.wfile.write(buf.tobytes())
                else:
                    self.send_error(500)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_error(500, str(e))

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
            try:
                out_frame = SERVER_INSTANCE.process_image(input_bgr, action=action)
                ret, buf = cv2.imencode(".jpg", out_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(buf)))
                    self._send_telemetry_headers()
                    self.end_headers()
                    self.wfile.write(buf.tobytes())
                else:
                    self.send_error(500)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_error(500, str(e))

        elif parsed.path == "/api/demo_frame":
            content_len = int(self.headers.get("Content-Length", 0))
            user_cam_bgr = None
            if content_len > 0:
                raw_bytes = self.rfile.read(content_len)
                nparr = np.frombuffer(raw_bytes, np.uint8)
                user_cam_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            action = params.get("action", [""])[0]
            try:
                out_frame = SERVER_INSTANCE.process_demo_frame(user_cam_bgr=user_cam_bgr, action=action)
                ret, buf = cv2.imencode(".jpg", out_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(buf)))
                    self._send_telemetry_headers()
                    self.end_headers()
                    self.wfile.write(buf.tobytes())
                else:
                    self.send_error(500)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_error(500, str(e))
        else:
            self.send_error(404)


def main():
    global SERVER_INSTANCE
    parser = argparse.ArgumentParser(description="DomainVision Web Streamer")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--width", type=int, default=640, help="Processing width (default: 640)")
    parser.add_argument("--height", type=int, default=360, help="Processing height (default: 360)")
    args = parser.parse_args()

    SERVER_INSTANCE = CursedARWebServer(port=args.port, width=args.width, height=args.height)
    server_address = ("0.0.0.0", args.port)
    ThreadingHTTPServer.allow_reuse_address = True
    httpd = ThreadingHTTPServer(server_address, ARStreamHandler)

    print("\n=======================================================")
    print(f"  DOMAINVISION AR WEB SERVER STARTED!")
    print(f"  URL: http://localhost:{args.port}/")
    print("  - Auto Domain Switching by Finger Sign (Sukuna / Gojo)")
    print("  - Picture-in-Picture live webcam at bottom during demo feed")
    print("  - Press Ctrl+C in terminal to stop.")
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
