import time
import math
import cv2

try:
    import RPi.GPIO as GPIO
except ModuleNotFoundError:
    class MockGPIO:
        BCM = "BCM"
        OUT = "OUT"
        HIGH = 1
        LOW = 0
        @staticmethod
        def setwarnings(flag): pass
        @staticmethod
        def setmode(mode): pass
        @staticmethod
        def setup(pin, mode, initial=None): pass
        @staticmethod
        def output(pin, level): pass
        @staticmethod
        def cleanup(): pass
    GPIO = MockGPIO()

RELAY_PIN_1 = 27
RELAY_PIN_2 = 17

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN_1, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(RELAY_PIN_2, GPIO.OUT, initial=GPIO.HIGH)

def relays_on():
    GPIO.output(RELAY_PIN_1, GPIO.LOW)
    GPIO.output(RELAY_PIN_2, GPIO.LOW)

def relays_off():
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

        # Calibration
        self.calibration_seconds  = 4.0
        self.calibration_start    = None
        self.baseline_face_y      = None
        self.baseline_face_h      = None
        self.baseline_eye_angle   = 0.0

        # Thresholds — normalized where possible
        # Face drop: fraction of face height (so distance-independent)
        self.y_drop_ratio         = 0.35   # head drops >35% of its own height
        # Forward lean: face grows more than this fraction
        self.forward_scale_ratio  = 1.12
        # Eye tilt
        self.eye_tilt_thresh      = 8.0    # degrees
        # Require 2 bad signals to trigger (reduces false positives)
        self.slouch_score_thresh  = 2

        # Smoothing
        self.smooth_face_y        = None
        self.smooth_face_h        = None
        self.smooth_eye_angle     = 0.0
        self.ema_alpha            = 0.30   # lower = smoother, less reactive

        # Slow drift for face size baseline
        # Allows baseline to gently follow if person moves closer/further
        self.drift_alpha          = 0.002  # very slow drift

        # Frame skipping
        self.detect_scale         = 0.5
        self.process_every_n      = 3
        self.frame_index          = 0
        self.cached_face          = None
        self.cached_eye_angle     = None
        self.missed_face_frames   = 0
        self.max_missed_frames    = 8

        # Good posture confirmation — must be good for N seconds before resetting
        self.good_posture_since   = None
        self.good_confirm_secs    = 1.5

        # Relay/pulse state
        self.is_slouching         = False
        self.bad_posture_start    = None
        self.pulse_active         = False
        self.pulse_start_time     = 0
        self.cooldown_until       = 0

        # Timing
        self.bad_posture_delay    = 3.0
        self.pulse_duration       = 1.0
        self.post_pulse_cooldown  = 5.0

    def _ema(self, previous, current, alpha=None):
        if alpha is None:
            alpha = self.ema_alpha
        if previous is None:
            return current
        return alpha * current + (1.0 - alpha) * previous

    def _find_face(self, gray):
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(80, 80),
        )
        if len(faces) == 0:
            return None
        return max(faces, key=lambda r: r[2] * r[3])

    def _find_eye_angle(self, gray, face):
        x, y, w, h = face
        roi = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(
            roi, scaleFactor=1.1, minNeighbors=10, minSize=(18, 18)
        )
        if len(eyes) < 2:
            return None
        eyes_sorted = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
        centers = [(x + ex + ew/2.0, y + ey + eh/2.0)
                   for ex, ey, ew, eh in eyes_sorted]
        (x1, y1), (x2, y2) = sorted(centers, key=lambda p: p[0])
        return math.degrees(math.atan2(y2-y1, x2-x1))

    def _update_calibration(self, face_y, face_h, eye_angle):
        self.baseline_face_y = self._ema(self.baseline_face_y, face_y)
        self.baseline_face_h = self._ema(self.baseline_face_h, face_h)
        if eye_angle is not None:
            self.baseline_eye_angle = self._ema(self.baseline_eye_angle, eye_angle)

    def _check_posture(self, face_y, face_h, eye_angle):
        feedback    = []
        slouch_score = 0

        # Check 1: face drop normalized by face height
        # Using face height makes it distance-independent
        y_drop_px    = face_y - self.baseline_face_y
        y_drop_ratio = y_drop_px / max(self.baseline_face_h, 1.0)
        if y_drop_ratio > self.y_drop_ratio:
            feedback.append("Sit taller")
            slouch_score += 1

        # Check 2: forward lean via face size growth
        # Apply slow drift so baseline follows gradual distance changes
        self.baseline_face_h = self._ema(
            self.baseline_face_h, face_h, alpha=self.drift_alpha
        )
        forward_ratio = face_h / max(self.baseline_face_h, 1.0)
        if forward_ratio > self.forward_scale_ratio:
            feedback.append("Pull head back")
            slouch_score += 1

        # Check 3: eye tilt
        if eye_angle is not None:
            angle_delta = abs(eye_angle - self.baseline_eye_angle)
            if angle_delta > self.eye_tilt_thresh:
                feedback.append("Level your head")
                slouch_score += 1

        is_bad = slouch_score >= self.slouch_score_thresh
        return feedback, is_bad

    def _handle_relay(self, is_bad_now):
        now = time.time()

        # Turn off pulse after duration
        if self.pulse_active and (now - self.pulse_start_time >= self.pulse_duration):
            relays_off()
            self.pulse_active    = False
            self.cooldown_until  = now + self.post_pulse_cooldown
            self.bad_posture_start = None

        if is_bad_now:
            # Reset good posture confirmation
            self.good_posture_since = None

            if self.bad_posture_start is None:
                self.bad_posture_start = now

            time_bad    = now - self.bad_posture_start
            in_cooldown = now < self.cooldown_until

            if (time_bad >= self.bad_posture_delay
                    and not self.pulse_active
                    and not in_cooldown):
                relays_on()
                self.pulse_active      = True
                self.pulse_start_time  = now
                self.bad_posture_start = None

        else:
            # Require sustained good posture before resetting
            if self.good_posture_since is None:
                self.good_posture_since = now

            good_for = now - self.good_posture_since
            if good_for >= self.good_confirm_secs:
                self.bad_posture_start = None
                if self.pulse_active:
                    relays_off()
                    self.pulse_active   = False
                    self.cooldown_until = now + self.post_pulse_cooldown

        self.is_slouching = is_bad_now

    def _draw_hud(self, frame, face, feedback, is_calibrating, now):
        if face is not None:
            x, y, w, h = face
            col = (0, 0, 255) if self.is_slouching else (90, 220, 90)
            cv2.rectangle(frame, (x, y), (x+w, y+h), col, 2)

        y_pos = 30

        if is_calibrating:
            elapsed   = now - self.calibration_start
            remaining = max(0, self.calibration_seconds - elapsed)
            cv2.putText(frame, f"Sit upright! Calibrating: {remaining:.1f}s",
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_pos += 35
        else:
            if self.is_slouching:
                cv2.putText(frame, "BAD POSTURE", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Good posture", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            y_pos += 35

        for msg in feedback:
            cv2.putText(frame, msg, (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_pos += 28

        # Status line at bottom
        h_frame = frame.shape[0]
        if self.pulse_active:
            t_left = max(0, self.pulse_duration - (now - self.pulse_start_time))
            cv2.putText(frame, f"PULSE ACTIVE: {t_left:.1f}s",
                (10, h_frame - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2)
        elif now < self.cooldown_until:
            t_left = max(0, self.cooldown_until - now)
            cv2.putText(frame, f"Cooldown: {t_left:.1f}s",
                (10, h_frame - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 165, 255), 2)
        elif self.is_slouching and self.bad_posture_start is not None:
            t_bad  = now - self.bad_posture_start
            t_left = max(0, self.bad_posture_delay - t_bad)
            cv2.putText(frame, f"Pulse in: {t_left:.1f}s",
                (10, h_frame - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 165, 255), 2)

    def track(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        self.calibration_start = time.time()
        print("Sit upright for calibration...")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.frame_index += 1
                run_detector = (self.frame_index % self.process_every_n) == 0

                face      = None
                eye_angle = None

                if run_detector:
                    small = cv2.resize(gray, None,
                        fx=self.detect_scale, fy=self.detect_scale,
                        interpolation=cv2.INTER_LINEAR)
                    det = self._find_face(small)
                    if det is not None:
                        x, y, w, h = det
                        s = 1.0 / self.detect_scale
                        face = (int(x*s), int(y*s), int(w*s), int(h*s))
                        eye_angle = self._find_eye_angle(gray, face)
                        self.cached_face      = face
                        self.cached_eye_angle = eye_angle
                        self.missed_face_frames = 0
                    else:
                        self.missed_face_frames += 1
                        if self.missed_face_frames > self.max_missed_frames:
                            self.cached_face      = None
                            self.cached_eye_angle = None
                        face      = self.cached_face
                        eye_angle = self.cached_eye_angle
                else:
                    face      = self.cached_face
                    eye_angle = self.cached_eye_angle

                now            = time.time()
                is_calibrating = (now - self.calibration_start) < self.calibration_seconds
                feedback       = []
                is_bad         = False

                if face is None:
                    feedback.append("Face not found - stay centered")
                    self._handle_relay(False)
                else:
                    x, y, w, h = face
                    face_y = y + h / 2.0
                    face_h = float(h)

                    self.smooth_face_y = self._ema(self.smooth_face_y, face_y)
                    self.smooth_face_h = self._ema(self.smooth_face_h, face_h)
                    if eye_angle is not None:
                        self.smooth_eye_angle = self._ema(
                            self.smooth_eye_angle, eye_angle)

                    if is_calibrating or self.baseline_face_y is None:
                        self._update_calibration(
                            self.smooth_face_y,
                            self.smooth_face_h,
                            self.smooth_eye_angle
                        )
                        self._handle_relay(False)
                    else:
                        feedback, is_bad = self._check_posture(
                            self.smooth_face_y,
                            self.smooth_face_h,
                            self.smooth_eye_angle if eye_angle is not None else None
                        )
                        self._handle_relay(is_bad)

                self._draw_hud(frame, face, feedback, is_calibrating, now)
                cv2.imshow("Posture Tracker (q to quit)", frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        finally:
            print("Shutting down...")
            relays_off()
            GPIO.cleanup()
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = PostureTracker()
    tracker.track()