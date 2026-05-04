import cv2, time, math
import numpy as np
import RPi.GPIO as GPIO

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

# ===== GPIO SETUP =====
RELAY_PIN         = 27
SHOCK_DURATION    = 0.5
BAD_POSTURE_DELAY = 3.0
SHOCK_COOLDOWN    = 10.0

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT, initial=GPIO.HIGH)

def trigger_relay_on():
    GPIO.output(RELAY_PIN, GPIO.LOW)

def trigger_relay_off():
    GPIO.output(RELAY_PIN, GPIO.HIGH)

# ===== MOVENET SETUP =====
MODEL_PATH  = "movenet_lightning.tflite"
INPUT_SIZE  = 192
CONF_THRESH = 0.15

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===== KEYPOINT INDICES =====
KP_LEFT_EAR       = 3
KP_RIGHT_EAR      = 4
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6

# ===== CAMERA =====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ===== THRESHOLDS (strict) =====
NECK_Y_DEVIATION  = 12   # pixels — any head height change from baseline
OFFSET_DEVIATION  = 12   # pixels — any horizontal head shift
TILT_DEVIATION    = 0.06 # shoulder tilt

# ===== CALIBRATION =====
CALIBRATION_TIME = 4.0

# ===== STATE =====
prev              = time.time()
bad_posture_since = None
shock_cooldown_ts = 0
shocking          = False
shock_start_time  = 0

calibrating          = True
calibration_start    = None
baseline_neck_y      = None
baseline_offset      = None
baseline_tilt        = None
calib_neck_samples   = []
calib_offset_samples = []
calib_tilt_samples   = []

# EMA smoothing to reduce jitter
smooth_neck_y  = None
smooth_offset  = None
smooth_tilt    = None
EMA_ALPHA      = 0.4

# ===== HELPERS =====
def run_inference(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = np.expand_dims(img, axis=0).astype(np.uint8)
    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()
    kps = interpreter.get_tensor(output_details[0]["index"])
    return kps[0][0]

def kp_to_pixel(kp, w, h):
    return int(kp[1] * w), int(kp[0] * h)

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def conf_ok(keypoints, *indices):
    return all(keypoints[i][2] > CONF_THRESH for i in indices)

def ema(prev, curr, alpha=EMA_ALPHA):
    if prev is None:
        return curr
    return alpha * curr + (1 - alpha) * prev

def draw_upper_body(frame, keypoints, w, h, color):
    shoulders_ok = conf_ok(keypoints, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
    ears_ok      = conf_ok(keypoints, KP_LEFT_EAR, KP_RIGHT_EAR)
    if shoulders_ok:
        l_s = kp_to_pixel(keypoints[KP_LEFT_SHOULDER],  w, h)
        r_s = kp_to_pixel(keypoints[KP_RIGHT_SHOULDER], w, h)
        cv2.line(frame, l_s, r_s, (100, 100, 255), 2)
        cv2.circle(frame, l_s, 6, (255, 0, 0), -1)
        cv2.circle(frame, r_s, 6, (255, 0, 0), -1)
        if ears_ok:
            l_e = kp_to_pixel(keypoints[KP_LEFT_EAR],  w, h)
            r_e = kp_to_pixel(keypoints[KP_RIGHT_EAR], w, h)
            s_mid = midpoint(l_s, r_s)
            e_mid = midpoint(l_e, r_e)
            cv2.line(frame, l_e, l_s,   color, 2)
            cv2.line(frame, r_e, r_s,   color, 2)
            cv2.line(frame, s_mid, e_mid, color, 3)
            cv2.circle(frame, l_e,   6, (0, 255, 0), -1)
            cv2.circle(frame, r_e,   6, (0, 255, 0), -1)
            cv2.circle(frame, e_mid, 8, (0, 255, 0), -1)
            cv2.circle(frame, s_mid, 8, (255, 0, 0), -1)

def handle_relay(is_bad):
    global shocking, shock_start_time, shock_cooldown_ts, bad_posture_since
    now = time.time()

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
            bad_posture_since = now
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
print("Sit upright for 4 seconds to calibrate...")
calibration_start = time.time()

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        keypoints = run_inference(frame)

        posture = "No person detected"
        color   = (255, 255, 255)
        is_bad  = False

        shoulders_ok = conf_ok(keypoints, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
        ears_ok      = conf_ok(keypoints, KP_LEFT_EAR,      KP_RIGHT_EAR)

        if shoulders_ok and ears_ok:
            l_shoulder   = kp_to_pixel(keypoints[KP_LEFT_SHOULDER],  w, h)
            r_shoulder   = kp_to_pixel(keypoints[KP_RIGHT_SHOULDER], w, h)
            l_ear        = kp_to_pixel(keypoints[KP_LEFT_EAR],       w, h)
            r_ear        = kp_to_pixel(keypoints[KP_RIGHT_EAR],      w, h)

            shoulder_mid = midpoint(l_shoulder, r_shoulder)
            ear_mid      = midpoint(l_ear,      r_ear)

            raw_neck_y  = shoulder_mid[1] - ear_mid[1]
            raw_offset  = ear_mid[0] - shoulder_mid[0]
            shoulder_w  = distance(l_shoulder, r_shoulder)
            raw_tilt    = (abs(l_shoulder[1] - r_shoulder[1]) / shoulder_w
                           if shoulder_w > 0 else 0)

            # Smooth all signals with EMA to reduce jitter
            smooth_neck_y = ema(smooth_neck_y, raw_neck_y)
            smooth_offset = ema(smooth_offset, raw_offset)
            smooth_tilt   = ema(smooth_tilt,   raw_tilt)

            # ===== CALIBRATION =====
            if calibrating:
                elapsed_calib   = time.time() - calibration_start
                remaining_calib = max(0, CALIBRATION_TIME - elapsed_calib)

                calib_neck_samples.append(smooth_neck_y)
                calib_offset_samples.append(smooth_offset)
                calib_tilt_samples.append(smooth_tilt)

                if elapsed_calib >= CALIBRATION_TIME:
                    baseline_neck_y = sum(calib_neck_samples)   / len(calib_neck_samples)
                    baseline_offset = sum(calib_offset_samples) / len(calib_offset_samples)
                    baseline_tilt   = sum(calib_tilt_samples)   / len(calib_tilt_samples)
                    calibrating     = False
                    print(f"Calibration done.")
                    print(f"  neck_y={baseline_neck_y:.1f}  "
                          f"offset={baseline_offset:.1f}  "
                          f"tilt={baseline_tilt:.3f}")

                draw_upper_body(frame, keypoints, w, h, (0, 255, 255))
                cv2.putText(frame,
                    f"Sit upright! {remaining_calib:.1f}s",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame,
                    f"neck_y={smooth_neck_y:.0f}  "
                    f"offset={smooth_offset:.0f}  "
                    f"tilt={smooth_tilt:.3f}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # ===== DETECTION =====
            else:
                issues       = []
                slouch_score = 0

                neck_y_diff  = smooth_neck_y - baseline_neck_y
                offset_diff  = smooth_offset - baseline_offset
                tilt_diff    = smooth_tilt   - baseline_tilt

                # Check 1: head height change in EITHER direction
                # up = leaning forward, down = slouching/relaxing back
                if abs(neck_y_diff) > NECK_Y_DEVIATION:
                    if neck_y_diff > 0:
                        issues.append("head up")
                    else:
                        issues.append("head down")
                    slouch_score += 1

                # Check 2: horizontal head shift in EITHER direction
                if abs(offset_diff) > OFFSET_DEVIATION:
                    if offset_diff < 0:
                        issues.append("leaning forward")
                    else:
                        issues.append("leaning back")
                    slouch_score += 1

                # Check 3: shoulder tilt
                if tilt_diff > TILT_DEVIATION:
                    issues.append("leaning sideways")
                    slouch_score += 1

                # Any single issue triggers
                is_bad = slouch_score >= 1

                if not is_bad:
                    posture = "Good posture"
                    color   = (0, 255, 0)
                else:
                    posture = "Bad: " + ", ".join(issues)
                    color   = (0, 0, 255)

                draw_upper_body(frame, keypoints, w, h, color)
                handle_relay(is_bad)

                cv2.putText(frame, posture, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                now = time.time()
                fps  = 1 / (now - prev) if now != prev else 0
                prev = now
                cv2.putText(frame, f"FPS: {fps:.0f}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.putText(frame,
                    f"neck:{smooth_neck_y:.0f}({neck_y_diff:+.0f})  "
                    f"off:{smooth_offset:.0f}({offset_diff:+.0f})  "
                    f"tilt:{tilt_diff:+.3f}",
                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                status = get_status_text(is_bad)
                if status:
                    cv2.putText(frame, status,
                        (10, 135), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 100, 255), 2)

        else:
            posture           = "Landmarks not visible"
            color             = (255, 255, 0)
            bad_posture_since = None
            if shocking:
                trigger_relay_off()
                shocking = False
            if not calibrating:
                cv2.putText(frame, posture, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Posture Monitor (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Shutting down...")
    trigger_relay_off()
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()