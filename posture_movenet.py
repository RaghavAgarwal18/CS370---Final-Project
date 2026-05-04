import cv2, time, math
import numpy as np
import RPi.GPIO as GPIO

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

# ===== GPIO SETUP =====
RELAY_PIN         = 17
SHOCK_DURATION    = 0.5    # seconds relay stays ON
BAD_POSTURE_DELAY = 3.0    # seconds of bad posture before shock
SHOCK_COOLDOWN    = 10.0   # seconds before next shock

GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT, initial=GPIO.HIGH)
# HIGH = relay OFF (active-low relay module)

def trigger_relay_on():
    GPIO.output(RELAY_PIN, GPIO.LOW)

def trigger_relay_off():
    GPIO.output(RELAY_PIN, GPIO.HIGH)

# ===== MOVENET SETUP =====
MODEL_PATH  = "movenet_lightning.tflite"
INPUT_SIZE  = 192
CONF_THRESH = 0.3

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===== KEYPOINT INDICES =====
KP_LEFT_EAR       = 3
KP_RIGHT_EAR      = 4
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP       = 11
KP_RIGHT_HIP      = 12

# ===== CAMERA =====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ===== POSTURE THRESHOLDS =====
NECK_ANGLE_THRESH    = 15
SPINE_ANGLE_THRESH   = 10
SHOULDER_TILT_THRESH = 0.15

# ===== STATE =====
prev              = time.time()
bad_posture_since = None
shock_cooldown_ts = 0
shocking          = False
shock_start_time  = 0

# ===== HELPERS =====
def run_inference(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = np.expand_dims(img, axis=0).astype(np.uint8)
    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()
    kps = interpreter.get_tensor(output_details[0]["index"])
    return kps[0][0]  # shape [17, 3]

def kp_to_pixel(kp, w, h):
    return int(kp[1] * w), int(kp[0] * h)

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def angle_from_vertical(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return abs(math.degrees(math.atan2(dx, dy)))

def conf_ok(keypoints, *indices):
    return all(keypoints[i][2] > CONF_THRESH for i in indices)

def draw_skeleton(frame, keypoints, w, h):
    connections = [
        (KP_LEFT_EAR,      KP_LEFT_SHOULDER),
        (KP_RIGHT_EAR,     KP_RIGHT_SHOULDER),
        (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER),
        (KP_LEFT_SHOULDER, KP_LEFT_HIP),
        (KP_RIGHT_SHOULDER,KP_RIGHT_HIP),
        (KP_LEFT_HIP,      KP_RIGHT_HIP),
    ]
    for a, b in connections:
        if conf_ok(keypoints, a, b):
            p1 = kp_to_pixel(keypoints[a], w, h)
            p2 = kp_to_pixel(keypoints[b], w, h)
            cv2.line(frame, p1, p2, (100, 100, 255), 2)
    for i in range(13):
        if keypoints[i][2] > CONF_THRESH:
            p = kp_to_pixel(keypoints[i], w, h)
            cv2.circle(frame, p, 4, (255, 200, 0), -1)

def handle_relay(is_bad):
    global shocking, shock_start_time, shock_cooldown_ts, bad_posture_since
    now = time.time()

    # Turn off shock after duration
    if shocking and (now - shock_start_time >= SHOCK_DURATION):
        trigger_relay_off()
        shocking = False
        shock_cooldown_ts = now + SHOCK_COOLDOWN

    if is_bad:
        if bad_posture_since is None:
            bad_posture_since = now
        time_bad    = now - bad_posture_since
        in_cooldown = now < shock_cooldown_ts
        if time_bad >= BAD_POSTURE_DELAY and not shocking and not in_cooldown:
            trigger_relay_on()
            shocking = True
            shock_start_time = now
            bad_posture_since = now  # reset timer
    else:
        bad_posture_since = None
        if shocking:
            trigger_relay_off()
            shocking = False
            shock_cooldown_ts = now + SHOCK_COOLDOWN

def get_status_text(is_bad):
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

        keypoints = run_inference(frame)
        draw_skeleton(frame, keypoints, w, h)

        posture = "No person detected"
        color   = (255, 255, 255)
        is_bad  = False

        needed = [
            KP_LEFT_EAR, KP_RIGHT_EAR,
            KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
            KP_LEFT_HIP, KP_RIGHT_HIP
        ]

        if conf_ok(keypoints, *needed):
            l_ear      = kp_to_pixel(keypoints[KP_LEFT_EAR],       w, h)
            r_ear      = kp_to_pixel(keypoints[KP_RIGHT_EAR],      w, h)
            l_shoulder = kp_to_pixel(keypoints[KP_LEFT_SHOULDER],  w, h)
            r_shoulder = kp_to_pixel(keypoints[KP_RIGHT_SHOULDER], w, h)
            l_hip      = kp_to_pixel(keypoints[KP_LEFT_HIP],       w, h)
            r_hip      = kp_to_pixel(keypoints[KP_RIGHT_HIP],      w, h)

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
                trigger_relay_off()
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
    trigger_relay_off()
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()