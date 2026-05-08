import cv2, time, math
import numpy as np
import RPi.GPIO as GPIO

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

# ===== GPIO =====
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

# ===== MOVENET =====
MODEL_PATH = "movenet_lightning.tflite"
INPUT_SIZE = 192
CONF_THRESH = 0.25

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

KP_LEFT_EAR       = 3
KP_RIGHT_EAR      = 4
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6

# ===== CAMERA =====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ===== SETTINGS =====
CALIBRATION_TIME  = 4.0
BAD_POSTURE_DELAY = 3.0
ALERT_DURATION    = 0.5
ALERT_COOLDOWN    = 10.0

# *** TIGHTENED THRESHOLD ***
# Was 0.25 — too lenient, head had to move a lot
# Now 0.10 — flags bad posture with much smaller forward lean
# Lower = stricter. Try 0.08 if still too lenient, 0.12 if too sensitive
NECK_FORWARD_THRESH = 0.06

# ===== STATE =====
prev               = time.time()
calibrating        = True
calibration_start  = time.time()
baseline_neck      = None
calib_samples      = []
smooth_neck        = None

# *** TIGHTENED EMA ***
# Was 0.35 — smoothed out too much, slow to react to posture changes
# Now 0.55 — reacts faster to changes in posture
EMA_ALPHA          = 0.55

bad_posture_since  = None
alert_active       = False
alert_start_time   = 0
alert_cooldown_until = 0

# ===== HELPERS =====
def run_inference(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = np.expand_dims(img, axis=0).astype(np.uint8)
    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0][0]

def kp_to_pixel(kp, w, h):
    return int(kp[1] * w), int(kp[0] * h)

def midpoint(p1, p2):
    return ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def conf_ok(keypoints, *indices):
    return all(keypoints[i][2] > CONF_THRESH for i in indices)

def ema(prev_val, curr_val):
    if prev_val is None:
        return float(curr_val)
    return EMA_ALPHA * curr_val + (1 - EMA_ALPHA) * prev_val

def handle_alert(is_bad):
    global alert_active, alert_start_time, alert_cooldown_until, bad_posture_since
    now = time.time()

    if alert_active and (now - alert_start_time >= ALERT_DURATION):
        relays_off()
        alert_active = False
        alert_cooldown_until = now + ALERT_COOLDOWN

    if is_bad:
        if bad_posture_since is None:
            bad_posture_since = now
        time_bad = now - bad_posture_since
        if (time_bad >= BAD_POSTURE_DELAY
                and not alert_active
                and now >= alert_cooldown_until):
            relays_on()
            alert_active     = True
            alert_start_time = now
            bad_posture_since = now
    else:
        bad_posture_since = None
        if alert_active:
            relays_off()
            alert_active = False
            alert_cooldown_until = time.time() + ALERT_COOLDOWN

def status_text(is_bad):
    now = time.time()
    if not is_bad or bad_posture_since is None:
        return ""
    remaining = max(0, BAD_POSTURE_DELAY - (now - bad_posture_since))
    if remaining > 0:
        return f"Alert in: {remaining:.1f}s"
    if alert_active:
        return "ALERT ACTIVE"
    cooldown = max(0, alert_cooldown_until - now)
    if cooldown > 0:
        return f"Cooldown: {cooldown:.1f}s"
    return ""

# ===== MAIN LOOP =====
print(f"Sit upright for {CALIBRATION_TIME:.0f} seconds to calibrate...")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        keypoints = run_inference(frame)

        color   = (255, 255, 255)
        posture = "No person detected"
        is_bad  = False

        shoulders_ok = conf_ok(keypoints, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
        ears_ok      = conf_ok(keypoints, KP_LEFT_EAR,      KP_RIGHT_EAR)

        if shoulders_ok and ears_ok:
            l_s = kp_to_pixel(keypoints[KP_LEFT_SHOULDER],  w, h)
            r_s = kp_to_pixel(keypoints[KP_RIGHT_SHOULDER], w, h)
            l_e = kp_to_pixel(keypoints[KP_LEFT_EAR],       w, h)
            r_e = kp_to_pixel(keypoints[KP_RIGHT_EAR],      w, h)

            s_mid = midpoint(l_s, r_s)
            e_mid = midpoint(l_e, r_e)

            shoulder_w = distance(l_s, r_s)

            if shoulder_w < 40:
                cv2.putText(frame, "Move closer",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                cv2.imshow("Posture Monitor (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            raw_neck   = (s_mid[1] - e_mid[1]) / shoulder_w
            smooth_neck = ema(smooth_neck, raw_neck)

            # Draw skeleton
            cv2.line(frame, l_s,   r_s,   (100, 100, 255), 2)
            cv2.line(frame, l_e,   l_s,   (200, 200, 200), 1)
            cv2.line(frame, r_e,   r_s,   (200, 200, 200), 1)
            cv2.line(frame, s_mid, e_mid, color, 3)
            cv2.circle(frame, l_s,   6, (255, 0,   0), -1)
            cv2.circle(frame, r_s,   6, (255, 0,   0), -1)
            cv2.circle(frame, l_e,   6, (0,   255, 0), -1)
            cv2.circle(frame, r_e,   6, (0,   255, 0), -1)
            cv2.circle(frame, s_mid, 8, (255, 0,   0), -1)
            cv2.circle(frame, e_mid, 8, (0,   255, 0), -1)

            # ===== CALIBRATION =====
            if calibrating:
                elapsed   = time.time() - calibration_start
                remaining = max(0, CALIBRATION_TIME - elapsed)
                calib_samples.append(smooth_neck)

                if elapsed >= CALIBRATION_TIME:
                    baseline_neck = float(np.median(calib_samples))
                    calibrating   = False
                    print(f"Calibration done. baseline_neck={baseline_neck:.3f}")

                cv2.putText(frame,
                    f"Sit upright! {remaining:.1f}s",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                cv2.putText(frame,
                    f"neck={smooth_neck:.3f}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

            # ===== DETECTION =====
            else:
                neck_diff = smooth_neck - baseline_neck
                is_bad = neck_diff < -NECK_FORWARD_THRESH

                if is_bad:
                    posture = f"Lean forward detected ({neck_diff:+.2f})"
                    color   = (0, 0, 255)
                else:
                    posture = "Good posture"
                    color   = (0, 255, 0)

                cv2.line(frame, s_mid, e_mid, color, 3)

                handle_alert(is_bad)

                cv2.putText(frame, posture,
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                now = time.time()
                fps  = 1 / (now - prev) if now != prev else 0
                prev = now

                cv2.putText(frame, f"FPS: {fps:.0f}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame,
                    f"neck={smooth_neck:.3f}  base={baseline_neck:.3f}  diff={neck_diff:+.3f}",
                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

                s = status_text(is_bad)
                if s:
                    cv2.putText(frame, s,
                        (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,100,255), 2)

        else:
            bad_posture_since = None
            handle_alert(False)
            if not calibrating:
                cv2.putText(frame, "Landmarks not visible",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.imshow("Posture Monitor (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Shutting down...")
    relays_off()
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
