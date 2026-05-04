import time
import math

import cv2
from plyer import notification

# ===== GPIO SETUP (with fallback mocking) =====
try:
    import RPi.GPIO as GPIO
except ModuleNotFoundError:
    # Mock GPIO for non-Pi systems (testing/development)
    class MockGPIO:
        BCM = "BCM"
        OUT = "OUT"
        HIGH = 1
        LOW = 0
        
        @staticmethod
        def setwarnings(flag):
            """Stub for GPIO.setwarnings on non-Pi systems."""
            pass
        
        @staticmethod
        def setmode(mode):
            """Stub for GPIO.setmode on non-Pi systems."""
            pass
        
        @staticmethod
        def setup(pin, mode, initial=None):
            """Stub for GPIO.setup on non-Pi systems."""
            pass
        
        @staticmethod
        def output(pin, level):
            """Stub for GPIO.output on non-Pi systems."""
            pass
        
        @staticmethod
        def cleanup():
            """Stub for GPIO.cleanup on non-Pi systems."""
            pass
    
    GPIO = MockGPIO()

RELAY_PIN_1 = 27
RELAY_PIN_2 = 17

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN_1, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(RELAY_PIN_2, GPIO.OUT, initial=GPIO.HIGH)

def relays_on():
    """Activate both relays."""
    GPIO.output(RELAY_PIN_1, GPIO.LOW)
    GPIO.output(RELAY_PIN_2, GPIO.LOW)

def relays_off():
    """Deactivate both relays."""
    GPIO.output(RELAY_PIN_1, GPIO.HIGH)
    GPIO.output(RELAY_PIN_2, GPIO.HIGH)


class PostureTracker:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise RuntimeError("OpenCV Haar cascades could not be loaded.")

        # Baseline calibration values (captured while posture is assumed good).
        self.calibration_seconds = 3.0
        self.calibration_start = None
        self.baseline_face_y = None
        self.baseline_face_h = None
        self.baseline_eye_angle = 0.0

        # Sensitive slouch thresholds.
        self.y_drop_threshold_px = 18
        self.forward_scale_ratio = 1.15
        self.eye_tilt_threshold_deg = 9.0

        # Smoothing to reduce jitter.
        self.smooth_face_y = None
        self.smooth_face_h = None
        self.smooth_eye_angle = 0.0
        self.ema_alpha = 0.35

        # Low-end performance controls.
        self.detect_scale = 0.5
        self.process_every_n_frames = 3
        self.frame_index = 0
        self.cached_face = None
        self.cached_eye_angle = None
        self.missed_face_frames = 0
        self.max_cached_face_frames = 8

        # Posture and relay state
        self.is_slouching = False
        self.bad_posture_start_time = None
        self.pulse_active = False
        self.pulse_start_time = 0
        self.cooldown_until = 0
        
        # Timing constants (seconds)
        self.bad_posture_delay = 3.0  # wait 3 seconds of bad posture before triggering
        self.pulse_duration = 1.0  # relay stays on for 1 second
        self.post_pulse_cooldown = 2.0  # wait 2 seconds after pulse before checking again

    def _ema(self, previous, current):
        if previous is None:
            return current
        return self.ema_alpha * current + (1.0 - self.ema_alpha) * previous

    def _find_face(self, gray):
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(90, 90),
        )
        if len(faces) == 0:
            return None

        # Keep largest face (closest/primary user).
        return max(faces, key=lambda r: r[2] * r[3])

    def _find_eye_angle(self, gray, face):
        x, y, w, h = face
        roi_gray = gray[y : y + h, x : x + w]
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(18, 18),
        )

        if len(eyes) < 2:
            return None

        # Use two largest eyes.
        eyes_sorted = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        centers = []
        for ex, ey, ew, eh in eyes_sorted:
            centers.append((x + ex + ew / 2.0, y + ey + eh / 2.0))

        (x1, y1), (x2, y2) = sorted(centers, key=lambda p: p[0])
        return math.degrees(math.atan2(y2 - y1, x2 - x1))

    def _update_calibration(self, face_y, face_h, eye_angle):
        self.baseline_face_y = self._ema(self.baseline_face_y, face_y)
        self.baseline_face_h = self._ema(self.baseline_face_h, face_h)
        if eye_angle is not None:
            self.baseline_eye_angle = self._ema(self.baseline_eye_angle, eye_angle)

    def _check_posture(self, face_y, face_h, eye_angle):
        feedback = []
        slouch_score = 0

        y_drop = face_y - self.baseline_face_y
        if y_drop > self.y_drop_threshold_px:
            feedback.append("You're slouching down - sit taller")
            slouch_score += 1

        forward_ratio = face_h / max(self.baseline_face_h, 1.0)
        if forward_ratio > self.forward_scale_ratio:
            feedback.append("Head too forward - pull neck back")
            slouch_score += 1

        if eye_angle is not None:
            angle_delta = abs(eye_angle - self.baseline_eye_angle)
            if angle_delta > self.eye_tilt_threshold_deg:
                feedback.append("Keep head level")
                slouch_score += 1

        return feedback, slouch_score > 0

    def _notify_slouch(self):
        """Handle pulse activation and timing logic."""
        now = time.time()
        
        # End pulse if duration exceeded
        if self.pulse_active and (now - self.pulse_start_time >= self.pulse_duration):
            relays_off()
            self.pulse_active = False
            self.cooldown_until = now + self.post_pulse_cooldown
            self.bad_posture_start_time = None
        
        # Start pulse if bad posture delay reached and not in cooldown
        if self.is_slouching and not self.pulse_active and now > self.cooldown_until:
            if self.bad_posture_start_time is None:
                self.bad_posture_start_time = now
            
            time_in_bad = now - self.bad_posture_start_time
            if time_in_bad >= self.bad_posture_delay:
                relays_on()
                self.pulse_active = True
                self.pulse_start_time = now
                self.bad_posture_start_time = None
        
        # Reset timer if posture corrected
        if not self.is_slouching:
            self.bad_posture_start_time = None
            if self.pulse_active:
                relays_off()
                self.pulse_active = False
                self.cooldown_until = now + self.post_pulse_cooldown

    def _draw_hud(self, frame, face, feedback, is_calibrating, now):
        if face is not None:
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 220, 90), 2)

        y_pos = 30
        
        # Calibration timer
        if is_calibrating:
            calib_elapsed = now - self.calibration_start
            calib_remaining = max(0, self.calibration_seconds - calib_elapsed)
            cv2.putText(
                frame,
                f"Calibrating... {calib_remaining:.1f}s",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            y_pos += 35

        # Posture feedback messages
        for i, msg in enumerate(feedback):
            cv2.putText(
                frame,
                msg,
                (10, y_pos + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
            )
        
        # Pulse/cooldown timers
        if self.pulse_active:
            # Show pulse duration remaining
            time_left = max(0, self.pulse_duration - (now - self.pulse_start_time))
            cv2.putText(
                frame,
                f"PULSE: {time_left:.1f}s",
                (10, frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3,
            )
        elif now < self.cooldown_until:
            # Show cooldown remaining
            cooldown_left = max(0, self.cooldown_until - now)
            cv2.putText(
                frame,
                f"Cooldown: {cooldown_left:.1f}s",
                (10, frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
            )
        elif self.is_slouching and self.bad_posture_start_time is not None:
            # Show bad posture countdown
            time_in_bad = now - self.bad_posture_start_time
            time_until_pulse = max(0, self.bad_posture_delay - time_in_bad)
            cv2.putText(
                frame,
                f"Bad posture: {time_until_pulse:.1f}s until pulse",
                (10, frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
            )
    
    def track(self):
        """Main tracking loop."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        self.calibration_start = time.time()
        print("Posture tracker started. Calibrating for 3 seconds...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frame_index += 1
            run_detector = (self.frame_index % self.process_every_n_frames) == 0

            face = None
            eye_angle = None

            if run_detector:
                small_gray = cv2.resize(
                    gray,
                    None,
                    fx=self.detect_scale,
                    fy=self.detect_scale,
                    interpolation=cv2.INTER_LINEAR,
                )
                detected_face = self._find_face(small_gray)
                if detected_face is not None:
                    x, y, w, h = detected_face
                    scale = 1.0 / self.detect_scale
                    face = (
                        int(x * scale),
                        int(y * scale),
                        int(w * scale),
                        int(h * scale),
                    )
                    eye_angle = self._find_eye_angle(gray, face)
                    self.cached_face = face
                    if eye_angle is not None:
                        self.cached_eye_angle = eye_angle
                    self.missed_face_frames = 0
                else:
                    self.missed_face_frames += 1
                    if self.missed_face_frames > self.max_cached_face_frames:
                        self.cached_face = None
                        self.cached_eye_angle = None
                    face = self.cached_face
                    eye_angle = self.cached_eye_angle
            else:
                face = self.cached_face
                eye_angle = self.cached_eye_angle

            feedback = []
            is_calibrating = (time.time() - self.calibration_start) < self.calibration_seconds
            is_slouching_now = False
            now = time.time()

            if face is None:
                feedback.append("Face not found - stay centered")
                self.is_slouching = False
            else:
                x, y, w, h = face
                face_y = y + h / 2.0
                face_h = float(h)

                self.smooth_face_y = self._ema(self.smooth_face_y, face_y)
                self.smooth_face_h = self._ema(self.smooth_face_h, face_h)
                if eye_angle is not None:
                    self.smooth_eye_angle = self._ema(self.smooth_eye_angle, eye_angle)

                if is_calibrating or self.baseline_face_y is None or self.baseline_face_h is None:
                    self._update_calibration(self.smooth_face_y, self.smooth_face_h, self.smooth_eye_angle)
                else:
                    feedback, is_slouching_now = self._check_posture(
                        self.smooth_face_y,
                        self.smooth_face_h,
                        self.smooth_eye_angle if eye_angle is not None else None,
                    )

                    if is_slouching_now and not self.is_slouching:
                        pass  # Update state; pulse logic runs next
                    
                    self.is_slouching = is_slouching_now
                    self._notify_slouch()  # Handle pulse timing

            self._draw_hud(frame, face, feedback, is_calibrating, now)
            
            cv2.imshow("Posture Tracker", frame)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        relays_off()
        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Posture tracker stopped.")

if __name__ == "__main__":
    tracker = PostureTracker()
    tracker.track()
