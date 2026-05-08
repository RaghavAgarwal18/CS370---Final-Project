import cv2, time, math
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

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
KP_LEFT_HIP       = 11
KP_RIGHT_HIP      = 12

# ===== CAMERA =====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

prev = time.time()

print("Running debug mode. Sit UPRIGHT for 10s, then SLOUCH for 10s.")
print("Watch the terminal for values. Press Q to quit.")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        keypoints = run_inference(frame)

        shoulders_ok = conf_ok(keypoints, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
        hips_ok      = conf_ok(keypoints, KP_LEFT_HIP, KP_RIGHT_HIP)
        ears_ok      = conf_ok(keypoints, KP_LEFT_EAR, KP_RIGHT_EAR)

        if shoulders_ok:
            l_shoulder   = kp_to_pixel(keypoints[KP_LEFT_SHOULDER], w, h)
            r_shoulder   = kp_to_pixel(keypoints[KP_RIGHT_SHOULDER], w, h)
            shoulder_mid = midpoint(l_shoulder, r_shoulder)

            # Draw shoulders
            cv2.circle(frame, l_shoulder,   6, (255, 0, 0), -1)
            cv2.circle(frame, r_shoulder,   6, (255, 0, 0), -1)
            cv2.circle(frame, shoulder_mid, 8, (0,   0, 255), -1)

            forward_offset = 0
            spine_ratio    = 0
            tilt_ratio     = 0
            neck_y_offset  = 0

            # Ear offset
            if ears_ok:
                l_ear    = kp_to_pixel(keypoints[KP_LEFT_EAR],  w, h)
                r_ear    = kp_to_pixel(keypoints[KP_RIGHT_EAR], w, h)
                ear_mid  = midpoint(l_ear, r_ear)
                forward_offset = ear_mid[0] - shoulder_mid[0]
                neck_y_offset  = shoulder_mid[1] - ear_mid[1]  # positive = ear above shoulder
                cv2.circle(frame, ear_mid, 8, (0, 255, 0), -1)
                cv2.line(frame, shoulder_mid, ear_mid, (255, 255, 0), 2)

            # Spine ratio
            if hips_ok:
                l_hip    = kp_to_pixel(keypoints[KP_LEFT_HIP],  w, h)
                r_hip    = kp_to_pixel(keypoints[KP_RIGHT_HIP], w, h)
                hip_mid  = midpoint(l_hip, r_hip)
                spine_height = hip_mid[1] - shoulder_mid[1]
                spine_ratio  = spine_height / h
                cv2.circle(frame, hip_mid, 8, (0, 0, 255), -1)
                cv2.line(frame, shoulder_mid, hip_mid, (255, 255, 0), 2)

            # Shoulder tilt
            shoulder_w    = distance(l_shoulder, r_shoulder)
            shoulder_tilt = abs(l_shoulder[1] - r_shoulder[1])
            tilt_ratio    = (shoulder_tilt / shoulder_w) if shoulder_w > 0 else 0

            # Print to terminal
            print(f"offset={forward_offset:+4d}  "
                  f"spine_ratio={spine_ratio:.3f}  "
                  f"neck_y={neck_y_offset:+4d}  "
                  f"tilt={tilt_ratio:.3f}")

            # Show all values on screen
            cv2.putText(frame, f"X offset: {forward_offset:+d}",
                (10, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Spine ratio: {spine_ratio:.3f}",
                (10, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Neck Y: {neck_y_offset:+d}",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Tilt: {tilt_ratio:.3f}",
                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        else:
            cv2.putText(frame, "Landmarks not visible",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # FPS
        now = time.time()
        fps  = 1 / (now - prev) if now != prev else 0
        prev = now
        cv2.putText(frame, f"FPS: {fps:.0f}",
            (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.imshow("Debug - Posture Values (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()