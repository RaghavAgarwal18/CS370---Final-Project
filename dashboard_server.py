from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO

from tracking_stats import TrackingStats

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
stats = TrackingStats(tracker_name="dashboard")


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "dashboard.html")


@app.get("/state")
def state():
    return jsonify(stats.snapshot())


@app.post("/update")
def update():
    payload = request.get_json(silent=True) or {}
    snapshot = stats.apply_snapshot(payload)
    socketio.emit("posture_update", snapshot)
    return jsonify({"ok": True, "state": snapshot})


@app.post("/reset")
def reset():
    stats.reset()
    snapshot = stats.snapshot()
    socketio.emit("posture_update", snapshot)
    return jsonify({"ok": True, "state": snapshot})


@socketio.on("connect")
def connect():
    socketio.emit("posture_update", stats.snapshot())


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
