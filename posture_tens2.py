# Install: pip install tflite-runtime opencv-python numpy RPi.GPIO
# Download model:
#   wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite
# Or MoveNet Thunder (better accuracy, slower):
#   wget https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite -O movenet_thunder.tflite

import cv2, time, math
import numpy as np
import RPi.GPIO as GPIO

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

# ===== GPIO SETUP =====
SHOCK_PIN         = 17
SHOCK_DURATION    = 0.5
BAD_POSTURE_DELAY = 3.0
SHOCK_COOLDOWN    = 10.0

GPIO.setmode(GPIO.BCM)
GPIO.setup(SHOCK_PIN, GPIO.OUT, initial=GPIO.HIGH)

def trigger_shock():
    GPIO.output(SHOCK_PIN, GPIO.LOW)
    time.sleep(SHOCK_DURATION)
    GPIO.output(SHOCK_PIN, GPIO.HIGH)

# ===== MOVENET KEYPOINT INDICES =====
KP_NOSE           = 0
KP_LEFT_EYE       = 1
KP_RIGHT_EYE      = 2
KP_LEFT_EAR       = 3
KP_RIGHT_EAR      = 4
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW     = 7
KP_RIGHT_ELBOW    = 8
KP_LEFT_WRIST     = 9
KP_RIGHT_WRIST    = 10
KP_LEFT_HIP       = 11
KP_RIGHT_HIP      = 12

# ===== LOAD MODEL =====
MODEL_PATH  = "movenet_thunder.tflite"
INPUT_SIZE  = 256   # Thunder=256, Lightning=192
CONF_THRESH = 0.3

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===== CAMERA =====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ===== STATE =====
prev              = time.time()
bad_posture_since = None
shock_cooldown_ts = 0

# ===== POSTURE THRESHOLDS =====
NECK_ANGLE_THRESH    = 15
SPINE_ANGLE_THRESH   = 10
SHOULDER_TILT_THRESH = 0.15

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

# ===== MAIN LOOP =====
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

            neck_angle   = angle_from_vertical(shoulder_mid, ear_mid)
            head_forward = neck_angle > NECK_ANGLE_THRESH

            spine_angle = angle_from_vertical(hip_mid, shoulder_mid)
            slouching   = spine_angle > SPINE_ANGLE_THRESH

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

            now = time.time()

            if not issues:
                posture           = "Good posture"
                color             = (0, 255, 0)
                bad_posture_since = None
            else:
                posture = "Bad: " + ", ".join(issues)
                color   = (0, 0, 255)

                if bad_posture_since is None:
                    bad_posture_since = now
                elif (
                    now - bad_posture_since >= BAD_POSTURE_DELAY
                    and now > shock_cooldown_ts
                ):
                    trigger_shock()
                    shock_cooldown_ts = now + SHOCK_COOLDOWN
                    bad_posture_since = now

            cv2.line(frame, hip_mid,      shoulder_mid, color, 2)
            cv2.line(frame, shoulder_mid, ear_mid,      color, 2)
            cv2.circle(frame, ear_mid,      7, (0,   255, 0),   -1)
            cv2.circle(frame, shoulder_mid, 7, (255, 0,   0),   -1)
            cv2.circle(frame, hip_mid,      7, (0,   0,   255), -1)

            cv2.putText(frame, f"Neck: {neck_angle:.1f}  Spine: {spine_angle:.1f}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            if bad_posture_since and issues:
                elapsed   = now - bad_posture_since
                remaining = max(0, BAD_POSTURE_DELAY - elapsed)
                cv2.putText(frame, f"Shock in: {remaining:.1f}s",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

        else:
            posture           = "Landmarks not visible"
            color             = (255, 255, 0)
            bad_posture_since = None

        now = time.time()
        fps  = 1 / (now - prev) if now != prev else 0
        prev = now

        cv2.putText(frame, posture,           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,   color,          2)
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Posture Monitor (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.output(SHOCK_PIN, GPIO.HIGH)
    GPIO.cleanup()