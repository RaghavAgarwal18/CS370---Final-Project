import cv2, time
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

prev = time.time()

def get_point(lm, idx, w, h):
    return int(lm[idx].x * w), int(lm[idx].y * h)

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    posture = "No person detected"
    color = (255,255,255)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # ===== KEY POINTS =====
        left_shoulder  = get_point(lm, 11, w, h)
        right_shoulder = get_point(lm, 12, w, h)
        left_ear       = get_point(lm, 7, w, h)
        right_ear      = get_point(lm, 8, w, h)

        # ===== MIDPOINTS =====
        shoulder_mid = (
            (left_shoulder[0] + right_shoulder[0]) // 2,
            (left_shoulder[1] + right_shoulder[1]) // 2
        )

        ear_mid = (
            (left_ear[0] + right_ear[0]) // 2,
            (left_ear[1] + right_ear[1]) // 2
        )

        # draw key points
        cv2.circle(frame, shoulder_mid, 8, (255,0,0), -1)
        cv2.circle(frame, ear_mid, 8, (0,255,0), -1)

        # ===== POSTURE LOGIC =====
        # if head is too far forward → bad posture
        forward_offset = ear_mid[0] - shoulder_mid[0]

        if abs(forward_offset) < 40:
            posture = "Good posture"
            color = (0,255,0)
        else:
            posture = "Bad posture (leaning forward)"
            color = (0,0,255)

        # draw line from shoulders to head
        cv2.line(frame, shoulder_mid, ear_mid, color, 3)

        cv2.putText(frame, f"Offset: {forward_offset}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # ===== FPS =====
    now = time.time()
    fps = 1/(now - prev) if now != prev else 0
    prev = now

    cv2.putText(frame, posture, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Posture Detection (press q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()