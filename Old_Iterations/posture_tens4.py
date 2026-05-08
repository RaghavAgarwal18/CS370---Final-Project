# Install on Pi:
#   pip install mediapipe-rpi4 opencv-python RPi.GPIO

import cv2, time, math
import mediapipe as mp
import RPi.GPIO as GPIO

# ===== GPIO SETUP =====
RELAY_PIN         = 17
SHOCK_DURATION    = 0.5    # seconds relay stays ON
BAD_POSTURE_DELAY = 3.0    # seconds of bad posture before shock
SHOCK_COOLDOWN    = 10.0   # seconds before next shock

GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT, initial=GPIO.HIGH)
# HIGH = relay OFF (active-low relay module)

def trigger_shock():
    GPIO.output(RELAY_PIN, GPIO.LOW)   # relay ON
    time.sleep(SHOCK_DURATION)
    GPIO.output(RELAY_PIN, GPIO.HIGH)  # relay OFF

# ===== MEDIAPIPE SETUP =====
# mediapipe-rpi4 uses the old mp.solutions API — this is intentional
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,           # MUST be 0 on Pi — lightest model
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===== CAMERA =====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ===== POSTURE THRESHOLDS (tune these for your setup) =====
NECK_ANGLE_THRESH    = 15    # degrees — forward head lean
SPINE_ANGLE_THRESH   = 10    # degrees — slouching
SHOULDER_TILT_THRESH = 0.15  # fraction of shoulder width — lateral lean
VISIBILITY_THRESH    = 0.5   # mediapipe landmark confidence minimum

# ===== STATE =====
prev              = time.time()
bad_posture_since = None
shock_cooldown_ts = 0
shocking          = False
shock_start_time  = 0

# ===== HELPERS =====
def get_point(lm, idx, w, h):
    return int(lm[idx].x * w), int(lm[idx].y * h)

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def angle_from_vertical(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return abs(math.degrees(math.atan2(dx, dy)))

def visibility_ok(lm, *indices):
    return all(lm[i].visibility > VISIBILITY_THRESH for i in indices)

def handle_relay(is_bad_posture):
    """Manages relay state with delay, duration, and cooldown."""
    global shocking, shock_start_time, shock_cooldown_ts, bad_posture_since

    now = time.time()

    # Turn off shock after SHOCK_DURATION
    if shocking and (now - shock_start_time >= SHOCK_DURATION):
        GPIO.output(RELAY_PIN, GPIO.HIGH)  # OFF
        shocking = False
        shock_cooldown_ts = now + SHOCK_COOLDOWN

    if is_bad_posture:
        if bad_posture_since is None:
            bad_posture_since = now

        time_bad    = now - bad_posture_since
        in_cooldown = now < shock_cooldown_ts

        if time_bad >= BAD_POSTURE_DELAY and not shocking and not in_cooldown:
            GPIO.output(RELAY_PIN, GPIO.LOW)  # ON
            shocking = True
            shock_start_time = now
            bad_posture_since = now  # reset so needs another 3s
    else:
        # Good posture — reset and cut shock early if active
        bad_posture_since = None
        if shocking:
            GPIO.output(RELAY_PIN, GPIO.HIGH)  # OFF
            shocking = False
            shock_cooldown_ts = now + SHOCK_COOLDOWN

def get_status_text(is_bad):
    """Returns the shock countdown or status string for the HUD."""
    now = time.time()
    if not is_bad or bad_posture_since is None:
        return ""
    elapsed   = now - bad_posture_since
    remaining = max(0, BAD_POSTURE_DELAY - elapsed)
    if remaining > 0:
        return f"Shock in: {remaining:.1f}s"
    elif shocking:
        return "SHOCK ACTIVE"
    else:
        cooldown_left = max(0, shock_cooldown_ts - now)
        return f"Cooldown: {cooldown_left:.1f}s"

# ===== MAIN LOOP =====
print("Starting posture detection. Press Q to quit.")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        posture = "No person detected"
        color   = (255, 255, 255)
        is_bad  = False

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            # Draw full body skeleton
            mp_draw.draw_landmarks(
                frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # Landmark indices:
            # 7=left ear, 8=right ear
            # 11=left shoulder, 12=right shoulder
            # 23=left hip, 24=right hip
            if visibility_ok(lm, 7, 8, 11, 12, 23, 24):
                l_ear      = get_point(lm, 7,  w, h)
                r_ear      = get_point(lm, 8,  w, h)
                l_shoulder = get_point(lm, 11, w, h)
                r_shoulder = get_point(lm, 12, w, h)
                l_hip      = get_point(lm, 23, w, h)
                r_hip      = get_point(lm, 24, w, h)

                ear_mid      = midpoint(l_ear,      r_ear)
                shoulder_mid = midpoint(l_shoulder, r_shoulder)
                hip_mid      = midpoint(l_hip,      r_hip)

                # Check 1: forward head lean
                neck_angle   = angle_from_vertical(shoulder_mid, ear_mid)
                head_forward = neck_angle > NECK_ANGLE_THRESH

                # Check 2: spine slouch
                spine_angle = angle_from_vertical(hip_mid, shoulder_mid)
                slouching   = spine_angle > SPINE_ANGLE_THRESH

                # Check 3: lateral shoulder tilt
                shoulder_w       = distance(l_shoulder, r_shoulder)
                shoulder_tilt    = abs(l_shoulder[1] - r_shoulder[1])
                leaning_sideways = (
                    (shoulder_tilt / shoulder_w) > SHOULDER_TILT_THRESH
                    if shoulder_w > 0 else False
                )

                # Verdict
                issues = []
                if head_forward:     issues.append("head forward")
                if slouching:        issues.append("slouching")
                if leaning_sideways: issues.append("leaning sideways")

                is_bad = len(issues) > 0

                if not is_bad:
                    posture = "Good posture"
                    color   = (0, 255, 0)
                else:
                    posture = "Bad: " + ", ".join(issues)
                    color   = (0, 0, 255)

                # Relay control
                handle_relay(is_bad)

                # Draw posture lines
                cv2.line(frame, hip_mid,      shoulder_mid, color, 2)
                cv2.line(frame, shoulder_mid, ear_mid,      color, 2)
                cv2.circle(frame, ear_mid,      7, (0,   255, 0),   -1)
                cv2.circle(frame, shoulder_mid, 7, (255, 0,   0),   -1)
                cv2.circle(frame, hip_mid,      7, (0,   0,   255), -1)

                # Debug angles
                cv2.putText(frame,
                    f"Neck: {neck_angle:.1f}  Spine: {spine_angle:.1f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Shock status
                status = get_status_text(is_bad)
                if status:
                    cv2.putText(frame, status,
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 100, 255), 2)

            else:
                posture           = "Landmarks not visible"
                color             = (255, 255, 0)
                bad_posture_since = None
                if shocking:
                    GPIO.output(RELAY_PIN, GPIO.HIGH)
                    shocking = False

        else:
            # No person — make sure relay is off
            bad_posture_since = None
            if shocking:
                GPIO.output(RELAY_PIN, GPIO.HIGH)
                shocking = False

        # FPS
        now = time.time()
        fps  = 1 / (now - prev) if now != prev else 0
        prev = now

        cv2.putText(frame, posture,           (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,   color,          2)
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Posture Monitor (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Shutting down...")
    GPIO.output(RELAY_PIN, GPIO.HIGH)  # ensure relay OFF
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    pose.close()