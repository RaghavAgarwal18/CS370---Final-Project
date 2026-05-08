from __future__ import annotations

import os
import time

import cv2


def _backend_candidates() -> list[tuple[int, str]]:
    backend_name = os.getenv("CAMERA_BACKEND", "").strip().upper()
    if backend_name:
        backend_map = {
            "CAP_DSHOW": cv2.CAP_DSHOW,
            "CAP_MSMF": cv2.CAP_MSMF,
            "CAP_ANY": cv2.CAP_ANY,
        }
        if backend_name not in backend_map:
            raise ValueError(f"Unsupported CAMERA_BACKEND={backend_name}. Use CAP_DSHOW, CAP_MSMF, or CAP_ANY.")
        return [(backend_map[backend_name], backend_name)]

    return [
        (cv2.CAP_DSHOW, "CAP_DSHOW"),
        (cv2.CAP_MSMF, "CAP_MSMF"),
        (cv2.CAP_ANY, "CAP_ANY"),
    ]


def _frame_ready(capture: cv2.VideoCapture) -> bool:
    for _ in range(3):
        ret, frame = capture.read()
        if ret and frame is not None and frame.size > 0:
            return True
        time.sleep(0.05)
    return False


def open_camera() -> tuple[cv2.VideoCapture, str]:
    preferred_index = int(os.getenv("CAMERA_INDEX", "0"))
    indices = [preferred_index] + [i for i in range(5) if i != preferred_index]

    for backend, backend_label in _backend_candidates():
        for index in indices:
            capture = cv2.VideoCapture(index, backend)
            if not capture.isOpened():
                capture.release()
                continue

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if _frame_ready(capture):
                print(f"Opened camera index {index} with {backend_label}")
                return capture, f"index={index} backend={backend_label}"

            print(f"Camera index {index} with {backend_label} opened but returned no frames; trying next option.")
            capture.release()

    raise RuntimeError(
        "Could not open any camera. Try closing Teams/Zoom/Chrome, or set CAMERA_INDEX=1 and CAMERA_BACKEND=CAP_DSHOW."
    )
