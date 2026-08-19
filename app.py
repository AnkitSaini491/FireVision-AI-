import cv2
import os
import numpy as np
import time

# ============================================================
# FIREVISION AI
# ============================================================

VIDEO_PATH = r"C:\Users\DELL\Downloads\fire_video.mp4.mp4"

WINDOW_NAME = "FireVision AI - Fire Detection"

MIN_FIRE_AREA = 250

MAX_WIDTH = 650
MAX_HEIGHT = 900


# ============================================================
# CHECK VIDEO
# ============================================================

print("=" * 60)
print("             FIREVISION AI")
print("       FIRE DETECTION SYSTEM")
print("=" * 60)

print("\nChecking video...")
print("Video:", VIDEO_PATH)

if not os.path.exists(VIDEO_PATH):

    print("\n❌ VIDEO NOT FOUND!")
    print("Check that the video exists in Downloads.")
    print("Expected filename: fire_video.mp4.mp4")

    input("\nPress Enter to exit...")
    raise SystemExit

print("✅ Video found!")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("❌ Unable to open video.")

    input("Press Enter to exit...")
    raise SystemExit


fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

delay = max(
    1,
    int(1000 / fps)
)


# ============================================================
# VIDEO INFORMATION
# ============================================================

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

print("Resolution:", width, "x", height)
print("FPS:", fps)


# ============================================================
# WINDOW
# ============================================================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    MAX_WIDTH,
    MAX_HEIGHT
)


# ============================================================
# FIRE DETECTION FUNCTION
# ============================================================

def detect_fire(frame):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------------
    # FIRE-LIKE COLOR RANGE
    # --------------------------------------------------------

    lower_fire = np.array(
        [0, 100, 120],
        dtype=np.uint8
    )

    upper_fire = np.array(
        [45, 255, 255],
        dtype=np.uint8
    )

    mask = cv2.inRange(
        hsv,
        lower_fire,
        upper_fire
    )

    # --------------------------------------------------------
    # CLEAN MASK
    # --------------------------------------------------------

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------------
    # FIND CONTOURS
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_FIRE_AREA:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if w < 10 or h < 10:
            continue

        detections.append(
            (x, y, w, h, area)
        )

    return detections


# ============================================================
# MAIN LOOP
# ============================================================

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:

        print("\n✅ Video finished.")
        break

    frame_number += 1

    height, width = frame.shape[:2]


    # ========================================================
    # FIRE DETECTION
    # ========================================================

    detections = detect_fire(frame)

    fire_count = len(detections)

    fire_detected = fire_count > 0


    # ========================================================
    # STATUS COLOR
    # ========================================================

    if fire_detected:

        status_color = (0, 0, 255)

    else:

        status_color = (0, 255, 0)


    # ========================================================
    # DRAW FIRE BOXES
    # ========================================================

    for x, y, w, h, area in detections:

        # RED BOX
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3
        )

        # FIRE LABEL
        label = "FIRE"

        cv2.rectangle(
            frame,
            (
                x,
                max(0, y - 35)
            ),
            (
                x + 100,
                y
            ),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            label,
            (
                x + 8,
                y - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


    # ========================================================
    # HEADER
    # ========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (width, 85),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "FIREVISION AI",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "FIRE DETECTION SYSTEM",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1
    )


    # ========================================================
    # ALERT BOX
    # ========================================================

    if fire_detected:

        # Flashing alert
        if int(time.time() * 5) % 2 == 0:

            cv2.rectangle(
                frame,
                (width - 205, 15),
                (width - 15, 70),
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                "FIRE ALERT",
                (width - 190, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2
            )

    else:

        cv2.rectangle(
            frame,
            (width - 205, 15),
            (width - 15, 70),
            (0, 100, 0),
            -1
        )

        cv2.putText(
            frame,
            "NORMAL",
            (width - 180, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2
        )


    # ========================================================
    # INFORMATION
    # ========================================================

    panel_y = 120

    cv2.putText(
        frame,
        f"FIRE REGIONS : {fire_count}",
        (20, panel_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        2
    )

    cv2.putText(
        frame,
        "CAMERA : ONLINE",
        (20, panel_y + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "AI MONITOR : ACTIVE",
        (20, panel_y + 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 0),
        2
    )


    # ========================================================
    # CENTER CROSSHAIR
    # ========================================================

    center_x = width // 2
    center_y = height // 2

    cv2.circle(
        frame,
        (center_x, center_y),
        30,
        status_color,
        1
    )

    cv2.line(
        frame,
        (center_x - 45, center_y),
        (center_x - 8, center_y),
        status_color,
        1
    )

    cv2.line(
        frame,
        (center_x + 8, center_y),
        (center_x + 45, center_y),
        status_color,
        1
    )

    cv2.line(
        frame,
        (center_x, center_y - 45),
        (center_x, center_y - 8),
        status_color,
        1
    )

    cv2.line(
        frame,
        (center_x, center_y + 8),
        (center_x, center_y + 45),
        status_color,
        1
    )


    # ========================================================
    # BORDER
    # ========================================================

    cv2.rectangle(
        frame,
        (8, 8),
        (width - 8, height - 8),
        status_color,
        2
    )


    # ========================================================
    # BOTTOM BAR
    # ========================================================

    cv2.rectangle(
        frame,
        (0, height - 65),
        (width, height),
        (0, 0, 0),
        -1
    )

    if fire_detected:

        bottom_text = "WARNING: FIRE DETECTED"

    else:

        bottom_text = "SYSTEM NORMAL"


    cv2.putText(
        frame,
        bottom_text,
        (15, height - 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        2
    )

    cv2.putText(
        frame,
        f"FRAME: {frame_number}",
        (width - 120, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1
    )


    # ========================================================
    # FIT SCREEN
    # ========================================================

    scale = min(
        MAX_WIDTH / width,
        MAX_HEIGHT / height,
        1.0
    )

    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
    )

    display_frame = cv2.resize(
        frame,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        display_frame
    )


    # ========================================================
    # EXIT
    # ========================================================

    key = cv2.waitKey(delay) & 0xFF

    if key == 27 or key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("\nFIREVISION AI STOPPED.")
