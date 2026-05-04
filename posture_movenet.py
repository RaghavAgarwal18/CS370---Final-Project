import cv2
import time
import math
import numpy as np
import RPi.GPIO as GPIO

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite


# ============================================================
# GPIO SETUP
# ============================================================

RELAY_PIN_1 = 27
RELAY_PIN_2 = 17

ALERT_DURATION = 0.5
BAD_POSTURE_DELAY = 3.0
ALERT_COOLDOWN = 10.0

# Most relay modules are ACTIVE LOW:
# LOW  = relay ON
# HIGH = relay OFF
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

GPIO.setup(RELAY_PIN_1, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(RELAY_PIN_2, GPIO.OUT, initial=GPIO.HIGH)


def trigger_relays_on():
    GPIO.output(RELAY_PIN_1, GPIO.LOW)
    GPIO.output(RELAY_PIN_2, GPIO.LOW)


def trigger_relays_off():
    GPIO.output(RELAY_PIN_1, GPIO.HIGH)
    GPIO.output(RELAY_PIN_2, GPIO.HIGH)


# ============================================================
# MOVENET SETUP
# ============================================================

MODEL_PATH = "movenet_lightning.tflite"
INPUT_SIZE = 192

# Higher confidence = fewer random bad keypoints
CONF_THRESH = 0.30

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# ============================================================
# KEYPOINT INDICES FOR MOVENET
# ============================================================

KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6


# ============================================================
# CAMERA SETUP
# ============================================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ============================================================
# POSTURE THRESHOLDS
# These are normalized by shoulder width, not raw pixels.
# ============================================================

NECK_RATIO_DEVIATION = 0.10
OFFSET_RATIO_DEVIATION = 0.12
TILT_RATIO_DEVIATION = 0.08

SEVERE_NECK_DEVIATION = 0.18
SEVERE_OFFSET_DEVIATION = 0.20
SEVERE_TILT_DEVIATION = 0.14


# ============================================================
# CALIBRATION
# ============================================================

CALIBRATION_TIME = 4.0


# ============================================================
# STATE
# ============================================================

prev = time.time()

bad_posture_since = None
alert_cooldown_until = 0

alert_active = False
alert_start_time = 0

calibrating = True
calibration_start = time.time()

baseline_neck = None
baseline_offset = None
baseline_tilt = None

calib_neck_samples = []
calib_offset_samples = []
calib_tilt_samples = []

smooth_neck = None
smooth_offset = None
smooth_tilt = None

EMA_ALPHA = 0.35


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def run_inference(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    inp = np.expand_dims(img, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()

    keypoints = interpreter.get_tensor(output_details[0]["index"])

    return keypoints[0][0]


def kp_to_pixel(kp, w, h):
    x = int(kp[1] * w)
    y = int(kp[0] * h)
    return x, y


def midpoint(p1, p2):
    return (
        (p1[0] + p2[0]) // 2,
        (p1[1] + p2[1]) // 2
    )


def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def conf_ok(keypoints, *indices):
    return all(keypoints[i][2] > CONF_THRESH for i in indices)


def ema(prev_value, curr_value, alpha=EMA_ALPHA):
    if prev_value is None:
        return curr_value
    return alpha * curr_value + (1 - alpha) * prev_value


def classify_posture(neck_ratio, offset_ratio, tilt_ratio,
                     baseline_neck, baseline_offset, baseline_tilt):
    issues = []
    score = 0
    severe = False

    neck_diff = neck_ratio - baseline_neck
    offset_diff = offset_ratio - baseline_offset
    tilt_diff = tilt_ratio - baseline_tilt

    # Head/neck vertical change
    if abs(neck_diff) > NECK_RATIO_DEVIATION:
        score += 1

        if neck_diff < 0:
            issues.append("head dropped")
        else:
            issues.append("head raised")

    if abs(neck_diff) > SEVERE_NECK_DEVIATION:
        severe = True

    # Head shift left/right
    if abs(offset_diff) > OFFSET_RATIO_DEVIATION:
        score += 1

        if offset_diff < 0:
            issues.append("head left")
        else:
            issues.append("head right")

    if abs(offset_diff) > SEVERE_OFFSET_DEVIATION:
        severe = True

    # Shoulder tilt
    if tilt_diff > TILT_RATIO_DEVIATION:
        score += 1
        issues.append("shoulder tilt")

    if tilt_diff > SEVERE_TILT_DEVIATION:
        severe = True

    # Better than old version:
    # bad if 2+ mild problems OR 1 severe problem
    is_bad = score >= 2 or severe

    return is_bad, issues, neck_diff, offset_diff, tilt_diff


def draw_upper_body(frame, keypoints, w, h, color):
    shoulders_ok = conf_ok(
        keypoints,
        KP_LEFT_SHOULDER,
        KP_RIGHT_SHOULDER
    )

    ears_ok = conf_ok(
        keypoints,
        KP_LEFT_EAR,
        KP_RIGHT_EAR
    )

    if not shoulders_ok:
        return

    l_shoulder = kp_to_pixel(keypoints[KP_LEFT_SHOULDER], w, h)
    r_shoulder = kp_to_pixel(keypoints[KP_RIGHT_SHOULDER], w, h)

    cv2.line(frame, l_shoulder, r_shoulder, (100, 100, 255), 2)
    cv2.circle(frame, l_shoulder, 6, (255, 0, 0), -1)
    cv2.circle(frame, r_shoulder, 6, (255, 0, 0), -1)

    if ears_ok:
        l_ear = kp_to_pixel(keypoints[KP_LEFT_EAR], w, h)
        r_ear = kp_to_pixel(keypoints[KP_RIGHT_EAR], w, h)

        shoulder_mid = midpoint(l_shoulder, r_shoulder)
        ear_mid = midpoint(l_ear, r_ear)

        cv2.line(frame, l_ear, l_shoulder, color, 2)
        cv2.line(frame, r_ear, r_shoulder, color, 2)
        cv2.line(frame, shoulder_mid, ear_mid, color, 3)

        cv2.circle(frame, l_ear, 6, (0, 255, 0), -1)
        cv2.circle(frame, r_ear, 6, (0, 255, 0), -1)

        cv2.circle(frame, shoulder_mid, 8, (255, 0, 0), -1)
        cv2.circle(frame, ear_mid, 8, (0, 255, 0), -1)


def handle_alert(is_bad):
    global alert_active
    global alert_start_time
    global alert_cooldown_until
    global bad_posture_since

    now = time.time()

    # Turn alert off after duration
    if alert_active and now - alert_start_time >= ALERT_DURATION:
        trigger_relays_off()
        alert_active = False
        alert_cooldown_until = now + ALERT_COOLDOWN

    if is_bad:
        if bad_posture_since is None:
            bad_posture_since = now

        time_bad = now - bad_posture_since
        in_cooldown = now < alert_cooldown_until

        if time_bad >= BAD_POSTURE_DELAY and not alert_active and not in_cooldown:
            trigger_relays_on()
            alert_active = True
            alert_start_time = now

            # Reset timer so it does not instantly trigger again
            bad_posture_since = now

    else:
        bad_posture_since = None

        if alert_active:
            trigger_relays_off()
            alert_active = False
            alert_cooldown_until = now + ALERT_COOLDOWN


def get_status_text(is_bad):
    now = time.time()

    if not is_bad or bad_posture_since is None:
        return ""

    elapsed = now - bad_posture_since
    remaining = max(0, BAD_POSTURE_DELAY - elapsed)

    if remaining > 0:
        return f"Alert in: {remaining:.1f}s"

    if alert_active:
        return "ALERT ACTIVE"

    cooldown_left = max(0, alert_cooldown_until - now)

    if cooldown_left > 0:
        return f"Cooldown: {cooldown_left:.1f}s"

    return ""


# ============================================================
# MAIN LOOP
# ============================================================

print("Sit upright for 4 seconds to calibrate...")

try:
    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        keypoints = run_inference(frame)

        posture = "No person detected"
        color = (255, 255, 255)
        is_bad = False

        shoulders_ok = conf_ok(
            keypoints,
            KP_LEFT_SHOULDER,
            KP_RIGHT_SHOULDER
        )

        ears_ok = conf_ok(
            keypoints,
            KP_LEFT_EAR,
            KP_RIGHT_EAR
        )

        if shoulders_ok and ears_ok:
            l_shoulder = kp_to_pixel(keypoints[KP_LEFT_SHOULDER], w, h)
            r_shoulder = kp_to_pixel(keypoints[KP_RIGHT_SHOULDER], w, h)
            l_ear = kp_to_pixel(keypoints[KP_LEFT_EAR], w, h)
            r_ear = kp_to_pixel(keypoints[KP_RIGHT_EAR], w, h)

            shoulder_mid = midpoint(l_shoulder, r_shoulder)
            ear_mid = midpoint(l_ear, r_ear)

            shoulder_width = distance(l_shoulder, r_shoulder)

            # If person is too far away / bad detection
            if shoulder_width < 40:
                posture = "Move closer / landmarks weak"
                color = (255, 255, 0)
                handle_alert(False)

                cv2.putText(
                    frame,
                    posture,
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2
                )

                cv2.imshow("Posture Monitor (q to quit)", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                continue

            # Normalized posture features
            raw_neck = (shoulder_mid[1] - ear_mid[1]) / shoulder_width
            raw_offset = (ear_mid[0] - shoulder_mid[0]) / shoulder_width
            raw_tilt = abs(l_shoulder[1] - r_shoulder[1]) / shoulder_width

            # Smooth signals
            smooth_neck = ema(globals()["smooth_neck"], raw_neck)
            smooth_offset = ema(globals()["smooth_offset"], raw_offset)
            smooth_tilt = ema(globals()["smooth_tilt"], raw_tilt)

            globals()["smooth_neck"] = smooth_neck
            globals()["smooth_offset"] = smooth_offset
            globals()["smooth_tilt"] = smooth_tilt

            # ====================================================
            # CALIBRATION MODE
            # ====================================================

            if calibrating:
                elapsed_calib = time.time() - calibration_start
                remaining_calib = max(0, CALIBRATION_TIME - elapsed_calib)

                calib_neck_samples.append(smooth_neck)
                calib_offset_samples.append(smooth_offset)
                calib_tilt_samples.append(smooth_tilt)

                draw_upper_body(frame, keypoints, w, h, (0, 255, 255))

                cv2.putText(
                    frame,
                    f"Sit upright! Calibrating: {remaining_calib:.1f}s",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"neck={smooth_neck:.2f}  offset={smooth_offset:.2f}  tilt={smooth_tilt:.2f}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )

                if elapsed_calib >= CALIBRATION_TIME:
                    baseline_neck = float(np.median(calib_neck_samples))
                    baseline_offset = float(np.median(calib_offset_samples))
                    baseline_tilt = float(np.median(calib_tilt_samples))

                    calibrating = False

                    print("Calibration done.")
                    print(f"  neck={baseline_neck:.2f}")
                    print(f"  offset={baseline_offset:.2f}")
                    print(f"  tilt={baseline_tilt:.2f}")

            # ====================================================
            # POSTURE DETECTION MODE
            # ====================================================

            else:
                is_bad, issues, neck_diff, offset_diff, tilt_diff = classify_posture(
                    smooth_neck,
                    smooth_offset,
                    smooth_tilt,
                    baseline_neck,
                    baseline_offset,
                    baseline_tilt
                )

                if is_bad:
                    posture = "Bad: " + ", ".join(issues)
                    color = (0, 0, 255)
                else:
                    posture = "Good posture"
                    color = (0, 255, 0)

                draw_upper_body(frame, keypoints, w, h, color)
                handle_alert(is_bad)

                cv2.putText(
                    frame,
                    posture,
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2
                )

                now = time.time()
                fps = 1 / (now - prev) if now != prev else 0
                prev = now

                cv2.putText(
                    frame,
                    f"FPS: {fps:.0f}",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"neck:{smooth_neck:.2f}({neck_diff:+.2f})  "
                    f"off:{smooth_offset:.2f}({offset_diff:+.2f})  "
                    f"tilt:{smooth_tilt:.2f}({tilt_diff:+.2f})",
                    (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1
                )

                status = get_status_text(is_bad)

                if status:
                    cv2.putText(
                        frame,
                        status,
                        (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 100, 255),
                        2
                    )

        else:
            posture = "Landmarks not visible"
            color = (255, 255, 0)

            bad_posture_since = None
            handle_alert(False)

            cv2.putText(
                frame,
                posture,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

        cv2.imshow("Posture Monitor (q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


finally:
    print("Shutting down...")

    trigger_relays_off()

    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()