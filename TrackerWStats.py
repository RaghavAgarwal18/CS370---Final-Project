from __future__ import annotations
import os
import time
from dataclasses import dataclass
import cv2
import numpy as np
import math
from tracking_stats import TrackingStats
from camera_utils import open_camera


BAD_POSTURE_DELAY = float(os.getenv("BAD_POSTURE_DELAY", "5.0"))
PULSE_DURATION = float(os.getenv("PULSE_DURATION", "1.0"))
POST_PULSE_COOLDOWN = float(os.getenv("POST_PULSE_COOLDOWN", "5.0"))
CALIBRATION_MIN_SAMPLES = int(os.getenv("CALIBRATION_MIN_SAMPLES", "12"))
CALIBRATION_MIN_SECONDS = float(os.getenv("CALIBRATION_MIN_SECONDS", "3.0"))

try:
    import RPi.GPIO as GPIO  # type: ignore
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    RELAY_PIN_1 = int(os.getenv("RELAY_PIN_1", "27"))
    RELAY_PIN_2 = int(os.getenv("RELAY_PIN_2", "17"))
    GPIO.setup(RELAY_PIN_1, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(RELAY_PIN_2, GPIO.OUT, initial=GPIO.HIGH)

    def relays_on() -> None:
        GPIO.output(RELAY_PIN_1, GPIO.LOW)
        GPIO.output(RELAY_PIN_2, GPIO.LOW)

    def relays_off() -> None:
        GPIO.output(RELAY_PIN_1, GPIO.HIGH)
        GPIO.output(RELAY_PIN_2, GPIO.HIGH)
except Exception:
    # Safe mock for non-Pi environments
    def relays_on() -> None:
        # keep silent in mock to avoid terminal clutter
        pass

    def relays_off() -> None:
        # keep silent in mock to avoid terminal clutter
        pass

@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int


@dataclass
class FrameAnalysis:
    status: str
    score: int
    issues: list[str]
    face_detected: bool
    calibrating: bool
    body_offset: tuple[float, float]
    face_box: Box | None
    body_box: Box | None
    baseline: tuple[float, float] | None


@dataclass
class TrackerRuntimeState:
    calibration_start: float
    calibration_samples: list[tuple[float, float]]
    baseline: tuple[float, float] | None
    last_publish: float
    last_frame_time: float = 0.0
    frame_index: int = 0
    process_every_n_frames: int = 5
    detect_scale: float = 0.4
    cached_face: Box | None = None
    cached_body: Box | None = None
    missed_detection_frames: int = 0
    max_cached_face_frames: int = 8
    # Shock state
    bad_posture_since: float | None = None
    pulse_active: bool = False
    pulse_start_time: float = 0.0
    cooldown_until: float = 0.0
    # Smoothing and eye-angle baseline
    smooth_face_y: float | None = None
    smooth_face_h: float | None = None
    smooth_eye_angle: float = 0.0
    baseline_eye_angle: float | None = None
    ema_alpha: float = 0.35
    eye_tilt_threshold_deg: float = 9.0
    # Face-based posture baseline (from CheckPosture logic)
    baseline_face_y: float | None = None
    baseline_face_h: float | None = None
    y_drop_threshold_px: float = 18.0
    forward_scale_ratio: float = 1.15
    slouch_threshold_frames: int = 15
    slouch_buffer: int = 0


def load_cascade(filename: str) -> cv2.CascadeClassifier:
    # Try cv2.data path first (works on most systems)
    if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
        cascade_path = cv2.data.haarcascades + filename
        cascade = cv2.CascadeClassifier(cascade_path)
        if not cascade.empty():
            return cascade
    
    # Fallback paths for Pi/Linux systems
    fallback_paths = [
        f"/usr/share/opencv4/cascades/{filename}",
        f"/usr/local/share/opencv4/cascades/{filename}",
        f"/home/pi/.local/lib/python*/site-packages/cv2/data/{filename}",
    ]
    
    for path in fallback_paths:
        import glob
        matches = glob.glob(path)
        if matches:
            cascade = cv2.CascadeClassifier(matches[0])
            if not cascade.empty():
                return cascade
    
    raise RuntimeError(f"Failed to load OpenCV cascade: {filename}. Tried cv2.data and common fallback paths.")


def find_eye_angle(gray: np.ndarray, face: Box, eye_cascade: cv2.CascadeClassifier) -> float | None:
    try:
        x, y, w, h = face.x, face.y, face.w, face.h
        roi_gray = gray[y:y + int(h / 1.8), x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(25, 25))
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            e1, e2 = eyes[0], eyes[1]
            p1 = (e1[0] + e1[2] / 2, e1[1] + e1[3] / 2)
            p2 = (e2[0] + e2[2] / 2, e2[1] + e2[3] / 2)
            dy = p2[1] - p1[1]
            dx = p2[0] - p1[0]
            angle = math.degrees(math.atan2(dy, dx))
            return angle
    except Exception:
        return None
    return None


def pick_largest(rectangles: np.ndarray) -> Box | None:
    if rectangles is None or len(rectangles) == 0:
        return None
    x, y, w, h = max(rectangles, key=lambda rect: rect[2] * rect[3])
    return Box(int(x), int(y), int(w), int(h))

def center(box: Box) -> tuple[float, float]:
    return box.x + box.w / 2.0, box.y + box.h / 2.0


def draw_box(frame: np.ndarray, box: Box, color: tuple[int, int, int], label: str) -> None:
    cv2.rectangle(frame, (box.x, box.y), (box.x + box.w, box.y + box.h), color, 2)
    cv2.putText(
        frame,
        label,
        (box.x, max(20, box.y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def score_posture(face: Box, body: Box, baseline: tuple[float, float] | None) -> tuple[int, list[str], tuple[float, float]]:
    face_x, face_y = center(face)
    body_x, _ = center(body)
    body_top_y = body.y

    raw_x_offset = face_x - body_x
    raw_y_offset = face_y - body_top_y

    if baseline is None:
        return 0, [], (raw_x_offset, raw_y_offset)

    base_x, base_y = baseline
    deviation_x = abs(raw_x_offset - base_x)
    deviation_y = abs(raw_y_offset - base_y)

    x_score = max(0.0, 1.0 - deviation_x / max(1.0, body.w * 0.35))
    y_score = max(0.0, 1.0 - deviation_y / max(1.0, body.h * 0.30))
    score = int(round(100.0 * (0.65 * x_score + 0.35 * y_score)))

    issues: list[str] = []
    if deviation_x > body.w * 0.25:
        issues.append("head shifted")
    if deviation_y > body.h * 0.20:
        issues.append("leaning forward")

    return score, issues, (raw_x_offset, raw_y_offset)


def analyze_frame(
    face: Box | None,
    body: Box | None,
    baseline: tuple[float, float] | None,
    calibration_start: float,
    calibration_samples: list[tuple[float, float]],
    now: float,
) -> tuple[FrameAnalysis, tuple[float, float] | None, list[tuple[float, float]]]:
    if face is None and body is None:
        return (
            FrameAnalysis(
                status="no_detection",
                score=0,
                issues=[],
                face_detected=False,
                calibrating=baseline is None,
                body_offset=(0.0, 0.0),
                face_box=None,
                body_box=None,
                baseline=baseline,
            ),
            baseline,
            calibration_samples,
        )

    if face is None:
        return (
            FrameAnalysis(
                status="body_only",
                score=0,
                issues=[],
                face_detected=False,
                calibrating=baseline is None,
                body_offset=(0.0, 0.0),
                face_box=None,
                body_box=body,
                baseline=baseline,
            ),
            baseline,
            calibration_samples,
        )

    if body is None:
        return (
            FrameAnalysis(
                status="face_only",
                score=0,
                issues=[],
                face_detected=True,
                calibrating=baseline is None,
                body_offset=(0.0, 0.0),
                face_box=face,
                body_box=None,
                baseline=baseline,
            ),
            baseline,
            calibration_samples,
        )

    score, issues, body_offset = score_posture(face, body, baseline)
    calibrating = baseline is None
    status = "calibrating"

    if baseline is None:
        calibration_samples = [*calibration_samples, body_offset]
        elapsed = now - calibration_start
        if len(calibration_samples) >= CALIBRATION_MIN_SAMPLES and elapsed >= CALIBRATION_MIN_SECONDS:
            offsets = np.array(calibration_samples, dtype=np.float32)
            baseline = (float(np.median(offsets[:, 0])), float(np.median(offsets[:, 1])))
            calibrating = False
            status = "good"
            score = 100
            issues = []
    else:
        status = "good" if score >= 80 else "bad"

    return (
        FrameAnalysis(
            status=status,
            score=score,
            issues=issues,
            face_detected=True,
            calibrating=calibrating,
            body_offset=body_offset,
            face_box=face,
            body_box=body,
            baseline=baseline,
        ),
        baseline,
        calibration_samples,
    )


def render_frame(frame: np.ndarray, analysis: FrameAnalysis) -> None:
    # Minimal HUD: show boxes and a concise status line
    if analysis.face_box is not None:
        draw_box(frame, analysis.face_box, (0, 200, 200), "Face")

    if analysis.body_box is not None:
        draw_box(frame, analysis.body_box, (200, 160, 0), "Upper body")

    color = (0, 200, 0) if analysis.status == "good" else (0, 0, 200)
    cv2.putText(frame, f"{analysis.status.upper()}  SCORE:{analysis.score}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def show_status_frame(window_name: str, message: str) -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, message, (35, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, "Press Q to quit", (35, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
    cv2.imshow(window_name, frame)


def scale_box(box: Box, scale: float) -> Box:
    return Box(
        int(box.x * scale),
        int(box.y * scale),
        int(box.w * scale),
        int(box.h * scale),
    )


def update_cached_detections(
    gray: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    upper_body_cascade: cv2.CascadeClassifier,
    state: TrackerRuntimeState,
) -> tuple[Box | None, Box | None]:
    if (state.frame_index % state.process_every_n_frames) != 0:
        return state.cached_face, state.cached_body

    small_gray = cv2.resize(
        gray,
        None,
        fx=state.detect_scale,
        fy=state.detect_scale,
        interpolation=cv2.INTER_LINEAR,
    )
    faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    bodies = upper_body_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))

    detected_face = pick_largest(faces)
    detected_body = pick_largest(bodies)
    scale = 1.0 / state.detect_scale

    if detected_face is not None:
        state.cached_face = scale_box(detected_face, scale)
        state.missed_detection_frames = 0
    else:
        state.missed_detection_frames += 1
        if state.missed_detection_frames > state.max_cached_face_frames:
            state.cached_face = None

    if detected_body is not None:
        state.cached_body = scale_box(detected_body, scale)

    return state.cached_face, state.cached_body


def build_stats_extra(analysis: FrameAnalysis) -> dict[str, object]:
    extra: dict[str, object] = {}
    if analysis.face_box is not None:
        extra["face_box"] = analysis.face_box.__dict__
    else:
        extra["face_box"] = None

    if analysis.body_box is not None:
        extra["body_box"] = analysis.body_box.__dict__
    else:
        extra["body_box"] = None

    if analysis.baseline is not None:
        extra["baseline"] = {"x": analysis.baseline[0], "y": analysis.baseline[1]}
    else:
        extra["baseline"] = None

    return extra


def maybe_publish_stats(
    stats: TrackingStats,
    analysis: FrameAnalysis,
    now: float,
    state: TrackerRuntimeState,
) -> None:
    if state.frame_index % 2 != 0:
        return

    stats.set_state(
        status=analysis.status,
        score=analysis.score,
        issues=analysis.issues,
        face_detected=analysis.face_detected,
        calibrating=analysis.calibrating,
        pulse_active=state.pulse_active,
        extra=build_stats_extra(analysis),
    )

    if now - state.last_publish >= 0.5:
        stats.publish()
        state.last_publish = now


def update_shock_state(stats: TrackingStats, analysis: FrameAnalysis, state: TrackerRuntimeState, now: float) -> None:
    # Enter bad posture timer
    if analysis.status == "bad":
        if not state.pulse_active and now > state.cooldown_until:
            if state.bad_posture_since is None:
                state.bad_posture_since = now
            elif now - state.bad_posture_since >= BAD_POSTURE_DELAY:
                # Trigger pulse
                relays_on()
                state.pulse_active = True
                state.pulse_start_time = now
                state.cooldown_until = now + POST_PULSE_COOLDOWN
                try:
                    stats.record_shock(analysis.issues)
                except Exception:
                    pass
    else:
        state.bad_posture_since = None

    # Turn off pulse after duration
    if state.pulse_active and (now - state.pulse_start_time) >= PULSE_DURATION:
        relays_off()
        state.pulse_active = False


def process_tracker_frame(
    cap: cv2.VideoCapture,
    window_name: str,
    face_cascade: cv2.CascadeClassifier,
    upper_body_cascade: cv2.CascadeClassifier,
    eye_cascade: cv2.CascadeClassifier,
    stats: TrackingStats,
    state: TrackerRuntimeState,
) -> bool:
    ret, frame = cap.read()
    if not ret:
        show_status_frame(window_name, "Waiting for camera frame...")
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    now = time.time()
    fps = 0.0
    if state.last_frame_time > 0.0:
        fps = 1.0 / max(0.0001, now - state.last_frame_time)
    state.last_frame_time = now

    frame = cv2.flip(frame, 1)
    preview = frame.copy()
    cv2.putText(preview, "Camera connected", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(preview, f"FPS: {fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    state.frame_index += 1
    face, body = update_cached_detections(gray, face_cascade, upper_body_cascade, state)

    # Compute eye angle and smooth face metrics for tilt detection
    eye_angle = None
    if face is not None:
        eye_angle = find_eye_angle(gray, face, eye_cascade)
        _, face_y = center(face)
        face_h = float(face.h)
        # EMA smoothing
        if state.smooth_face_y is None:
            state.smooth_face_y = face_y
        else:
            state.smooth_face_y = state.ema_alpha * face_y + (1.0 - state.ema_alpha) * state.smooth_face_y

        if state.smooth_face_h is None:
            state.smooth_face_h = face_h
        else:
            state.smooth_face_h = state.ema_alpha * face_h + (1.0 - state.ema_alpha) * state.smooth_face_h

        if eye_angle is not None:
            state.smooth_eye_angle = state.ema_alpha * eye_angle + (1.0 - state.ema_alpha) * state.smooth_eye_angle
        # Calibrate eye baseline while tracker is calibrating for body baseline
        is_calibrating = state.baseline is None
        if is_calibrating or state.baseline_eye_angle is None:
            if eye_angle is not None:
                if state.baseline_eye_angle is None:
                    state.baseline_eye_angle = eye_angle
                else:
                    state.baseline_eye_angle = state.ema_alpha * eye_angle + (1.0 - state.ema_alpha) * state.baseline_eye_angle

    now = time.time()
    analysis, state.baseline, state.calibration_samples = analyze_frame(
        face,
        body,
        state.baseline,
        state.calibration_start,
        state.calibration_samples,
        now,
    )

    # If significant head tilt detected compared to calibrated baseline, mark issue
    if face is not None and eye_angle is not None and state.baseline_eye_angle is not None:
        angle_delta = abs(state.smooth_eye_angle - state.baseline_eye_angle)
        if angle_delta > state.eye_tilt_threshold_deg:
            if "Keep head level" not in analysis.issues:
                analysis.issues.append("Keep head level")
            # Downgrade status to bad if not already
            if analysis.status != "bad":
                analysis.status = "bad"
                analysis.score = min(analysis.score, 75)

    # Face-based posture checks similar to CheckPosture logic
    # Update baseline face metrics while calibrating
    is_calibrating = state.baseline is None
    if face is not None:
        # update baseline face y/h during calibration
        face_center_y = center(face)[1]
        face_h = float(face.h)
        if is_calibrating or state.baseline_face_y is None or state.baseline_face_h is None:
            # EMA update
            if state.baseline_face_y is None:
                state.baseline_face_y = face_center_y
            else:
                state.baseline_face_y = state.ema_alpha * face_center_y + (1.0 - state.ema_alpha) * state.baseline_face_y

            if state.baseline_face_h is None:
                state.baseline_face_h = face_h
            else:
                state.baseline_face_h = state.ema_alpha * face_h + (1.0 - state.ema_alpha) * state.baseline_face_h
        else:
            # Evaluate posture deviations
            y_drop = face_center_y - state.baseline_face_y
            forward_ratio = face_h / max(1.0, state.baseline_face_h)
            face_slouch = False
            if y_drop > state.y_drop_threshold_px + 5:
                face_slouch = True
                if "You're slouching down - sit taller" not in analysis.issues:
                    analysis.issues.append("You're slouching down - sit taller")
            if forward_ratio > state.forward_scale_ratio + 0.05:
                face_slouch = True
                if "Head too forward - pull neck back" not in analysis.issues:
                    analysis.issues.append("Head too forward - pull neck back")
            # eye tilt already added above
            if face_slouch:
                state.slouch_buffer += 1
            else:
                state.slouch_buffer = 0

            if state.slouch_buffer >= state.slouch_threshold_frames:
                if analysis.status != "bad":
                    analysis.status = "bad"
                    analysis.score = min(analysis.score, 70)

    render_frame(preview, analysis)

    # Show countdown warning if bad posture is being counted down
    if analysis.status == "bad" and state.bad_posture_since is not None and not state.pulse_active:
        time_in_bad = now - state.bad_posture_since
        time_until_pulse = max(0.0, BAD_POSTURE_DELAY - time_in_bad)
        warning_text = f"⚡ PULSE IN {time_until_pulse:.1f}s"
        cv2.putText(preview, warning_text, (10, preview.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    elif state.pulse_active:
        time_left = max(0.0, PULSE_DURATION - (now - state.pulse_start_time))
        cv2.putText(preview, f"⚡ PULSING... {time_left:.1f}s", (10, preview.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    elif now < state.cooldown_until:
        cooldown_left = max(0.0, state.cooldown_until - now)
        cv2.putText(preview, f"⏳ Cooldown: {cooldown_left:.1f}s", (10, preview.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    # Update shock/pulse state (may activate relays)
    update_shock_state(stats, analysis, state, now)

    maybe_publish_stats(stats, analysis, now, state)

    # If calibrating, show a clear overlay with progress and hint to recalibrate
    if analysis.calibrating:
        elapsed = now - state.calibration_start
        cv2.putText(preview, f"CALIBRATING {elapsed:.1f}s/{CALIBRATION_MIN_SECONDS:.1f}s", (10, preview.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
        cv2.putText(preview, f"Samples: {len(state.calibration_samples)}", (10, preview.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(preview, "Press R to recalibrate", (10, preview.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    cv2.imshow(window_name, preview)
    # accept 'q' or 'Q' to quit, 'r' or 'R' to recalibrate
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        return False
    if key == ord('r') or key == ord('R'):
        state.calibration_start = time.time()
        state.calibration_samples = []
        state.baseline = None
        # also reset face/eye baseline smoothing so calibration starts fresh
        state.baseline_eye_angle = None
        state.baseline_face_y = None
        state.baseline_face_h = None
        return True
    return True


def run_tracker() -> None:
    face_cascade = load_cascade("haarcascade_frontalface_default.xml")
    upper_body_cascade = load_cascade("haarcascade_upperbody.xml")
    eye_cascade = load_cascade("haarcascade_eye.xml")

    cap, camera_label = open_camera()
    print("PostureGuard: camera started")

    window_name = "Posture Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass

    splash = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(splash, "Starting camera...", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(splash, "Press Q to quit", (50, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
    cv2.imshow(window_name, splash)
    cv2.waitKey(1)

    stats = TrackingStats(
        server_url=os.getenv("POSTURE_STATS_SERVER", "http://127.0.0.1:5000"),
        tracker_name=os.getenv("TRACKER_NAME", "tracker_w_stats"),
    )
    state = TrackerRuntimeState(
        calibration_start=time.time(),
        calibration_samples=[],
        baseline=None,
        last_publish=time.time(),
    )

    while cap.isOpened():
        if not process_tracker_frame(
            cap,
            window_name,
            face_cascade,
            upper_body_cascade,
            eye_cascade,
            stats,
            state,
        ):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_tracker()