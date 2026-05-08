import cv2
import time
import math
import mediapipe as mp
import RPi.GPIO as GPIO

# ===== GPIO SETUP =====
RELAY_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.LOW)  # make sure TENS is off at start

# ===== MEDIAPIPE SETUP =====
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===== WEBCAM SETUP =====
cap = cv2.VideoCapture(0)  # 0 = USB webcam plugged into Pi

# ===== SETTINGS (tweak these to your liking) =====
BAD_POSTURE_THRESHOLD  = 40   # pixel offset before considered bad posture
BAD_POSTURE_DELAY      = 3.0  # seconds of bad posture before triggering TENS
SHOCK_DURATION         = 1.0  # seconds the TENS stays on per trigger
COOLDOWN_DURATION      = 5.0  # seconds to wait before triggering again

# ===== STATE VARIABLES =====
prev_time         = time.time()
bad_posture_start = None   # when bad posture was first detected
last_shock_time   = 0      # when the last shock was triggered
shocking          = False  # is the TENS currently on?
shock_start_time  = 0      # when the current shock started

print("Posture detection started. Press Q to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to read from webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    posture = "No person detected"
    color   = (255, 255, 255)
    status_text = ""

    # ===== HANDLE SHOCK DURATION (turn off after SHOCK_DURATION seconds) =====
    if shocking and (time.time() - shock_start_time >= SHOCK_DURATION):
        GPIO.output(RELAY_PIN, GPIO.LOW)
        shocking = False
        last_shock_time = time.time()
        print("TENS off.")

    # ===== POSE DETECTION =====
    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        def get_point(idx):
            return int(lm[idx].x * w), int(lm[idx].y * h)

        left_shoulder  = get_point(11)
        right_shoulder = get_point(12)
        left_ear       = get_point(7)
        right_ear      = get_point(8)

        shoulder_mid = (
            (left_shoulder[0] + right_shoulder[0]) // 2,
            (left_shoulder[1] + right_shoulder[1]) // 2
        )
        ear_mid = (
            (left_ear[0] + right_ear[0]) // 2,
            (left_ear[1] + right_ear[1]) // 2
        )

        # draw key points
        cv2.circle(frame, shoulder_mid, 8, (255, 0, 0), -1)
        cv2.circle(frame, ear_mid,      8, (0, 255, 0), -1)

        forward_offset = ear_mid[0] - shoulder_mid[0]

        # ===== POSTURE LOGIC =====
        if abs(forward_offset) < BAD_POSTURE_THRESHOLD:
            posture = "Good posture"
            color   = (0, 255, 0)
            bad_posture_start = None  # reset timer

            # turn TENS off if posture corrected mid-shock
            if shocking:
                GPIO.output(RELAY_PIN, GPIO.LOW)
                shocking = False
                print("Posture corrected - TENS off.")

        else:
            posture = "Bad posture (leaning forward)"
            color   = (0, 0, 255)

            if bad_posture_start is None:
                bad_posture_start = time.time()

            time_in_bad_posture = time.time() - bad_posture_start
            time_since_last_shock = time.time() - last_shock_time

            # show countdown to shock
            countdown = max(0, BAD_POSTURE_DELAY - time_in_bad_posture)
            if countdown > 0:
                status_text = f"Shock in {countdown:.1f}s"
            elif shocking:
                status_text = "TENS ON!"
            else:
                status_text = f"Cooldown: {max(0, COOLDOWN_DURATION - time_since_last_shock):.1f}s"

            # trigger TENS if delay passed and not in cooldown and not already shocking
            if (time_in_bad_posture >= BAD_POSTURE_DELAY
                    and not shocking
                    and time_since_last_shock >= COOLDOWN_DURATION):
                GPIO.output(RELAY_PIN, GPIO.HIGH)
                shocking = True
                shock_start_time = time.time()
                print("Bad posture detected - TENS on!")

        # draw line and offset
        cv2.line(frame, shoulder_mid, ear_mid, color, 3)
        cv2.putText(frame, f"Offset: {forward_offset}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ===== FPS =====
    now      = time.time()
    fps      = 1 / (now - prev_time) if now != prev_time else 0
    prev_time = now

    # ===== ON-SCREEN TEXT =====
    cv2.putText(frame, posture,      (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,   color,          2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if status_text:
        cv2.putText(frame, status_text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    cv2.imshow("Posture Detection (press Q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===== CLEANUP =====
print("Shutting down...")
GPIO.output(RELAY_PIN, GPIO.LOW)  # make absolutely sure TENS is off
cap.release()
cv2.destroyAllWindows()
pose.close()
GPIO.cleanup()
print("Done.")
