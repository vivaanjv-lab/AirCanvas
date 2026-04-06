import cv2
import numpy as np
import mediapipe as mp
import time
import os
import urllib.request
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision as mpv

MODEL = "hand_landmarker.task"
if not os.path.exists(MODEL):
    print("Downloading model, one time only...")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", MODEL)

latest_result = [None]
def save_result(result, _, __):
    latest_result[0] = result

detector = mpv.HandLandmarker.create_from_options(
    mpv.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=MODEL),
        running_mode=mpv.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        result_callback=save_result,
    )
)

cam = cv2.VideoCapture(0)
_, first_frame = cam.read()
first_frame = cv2.flip(first_frame, 1)
height, width = first_frame.shape[:2]
drawing = np.zeros((height, width, 3), np.uint8)

colors      = [(230,80,30), (30,200,30), (30,30,220), (0,220,220), (255,255,255)]
color_names = ["Blue", "Green", "Red", "Yellow", "White"]
brush_sizes = [4, 8, 14, 22]

chosen_color = 0
chosen_brush = 1
last_point   = None
hovering_btn = -1
hover_start  = 0

TIPS = [4, 8, 12, 16, 20]
KNUX = [3, 6, 10, 14, 18]
BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),
         (10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
TOOLBAR_HEIGHT = 65
BTN_WIDTH = 100

def get_gesture(landmarks, hand):
    fingers_up = [1 if landmarks[TIPS[i]].y < landmarks[KNUX[i]].y else 0 for i in range(1, 5)]
    if fingers_up[0] and not any(fingers_up[1:]):      return "draw"
    if fingers_up[0] and fingers_up[1] and not any(fingers_up[2:]): return "select"
    return "idle"

def show_toolbar(frame, chosen_color, chosen_brush, hovering_btn, hover_progress):
    cv2.rectangle(frame, (0, 0), (width, TOOLBAR_HEIGHT), (20, 20, 30), -1)
    for i, (name, col) in enumerate(zip(color_names, colors)):
        x1, x2 = i * BTN_WIDTH + 4, i * BTN_WIDTH + 96
        cv2.rectangle(frame, (x1, 8), (x2, 57), col, -1)
        border = (255, 255, 255) if i == chosen_color else (70, 70, 100)
        cv2.rectangle(frame, (x1, 8), (x2, 57), border, 3 if i == chosen_color else 1)
        cv2.putText(frame, name, (x1+5, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (10,10,10) if name in ("White","Yellow") else (230,230,230), 1)
        if i == hovering_btn and hover_progress > 0:
            cv2.ellipse(frame, ((x1+x2)//2, 32), (18, 18), -90, 0, int(360 * hover_progress), (255, 255, 255), 3)
    brush_px = brush_sizes[chosen_brush] * 2
    cv2.putText(frame, f"Brush: {brush_px}px  b=resize  c=clear  s=save  q=quit", (len(colors)*BTN_WIDTH+10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160,160,200), 1)

print("Air Canvas ready! ☝ draw  ✌ pick color  ✊ pen up")

while True:
    ret, frame = cam.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    detector.detect_async(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
        int(time.time() * 1000)
    )

    fingertip = None
    gesture   = "idle"

    if latest_result[0] and latest_result[0].hand_landmarks:
        lm    = latest_result[0].hand_landmarks[0]
        hand  = latest_result[0].handedness[0][0].category_name if latest_result[0].handedness else "Right"
        pts   = [(int(p.x * width), int(p.y * height)) for p in lm]
        gesture   = get_gesture(lm, hand)
        fingertip = pts[8]

        for a, b in BONES:
            cv2.line(frame, pts[a], pts[b], (0, 220, 100), 1)
        for i, p in enumerate(pts):
            cv2.circle(frame, p, 5 if i in TIPS else 3, (255, 255, 255), -1)

        if gesture == "draw" and fingertip[1] > TOOLBAR_HEIGHT:
            if last_point and last_point[1] > TOOLBAR_HEIGHT:
                r = brush_sizes[chosen_brush]
                cv2.line(drawing, last_point, fingertip, colors[chosen_color], r * 2)
                cv2.circle(drawing, fingertip, r, colors[chosen_color], -1)
            last_point = fingertip
        else:
            last_point = None

    hover_progress = 0
    if gesture == "select" and fingertip and 0 < fingertip[1] <= TOOLBAR_HEIGHT:
        btn = fingertip[0] // BTN_WIDTH
        if 0 <= btn < len(colors):
            if btn == hovering_btn:
                hover_progress = min((time.time() - hover_start) / 0.6, 1.0)
                if hover_progress >= 1.0:
                    chosen_color = btn
                    hovering_btn = -1
            else:
                hovering_btn = btn
                hover_start  = time.time()
        else:
            hovering_btn = -1
    else:
        hovering_btn = -1

    mask = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)
    output = np.where(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0, cv2.addWeighted(frame, 0.3, drawing, 0.7, 0), frame)

    show_toolbar(output, chosen_color, chosen_brush, hovering_btn, hover_progress)

    if fingertip and fingertip[1] > TOOLBAR_HEIGHT:
        cv2.circle(output, fingertip, brush_sizes[chosen_brush] + 5, (255, 255, 255), 2)
        if gesture == "draw":
            cv2.circle(output, fingertip, brush_sizes[chosen_brush], colors[chosen_color], -1)

    cv2.imshow("Air Canvas", output)

    key = cv2.waitKey(1) & 0xFF
    if   key == ord('q'): break
    elif key == ord('c'): drawing[:] = 0; last_point = None
    elif key == ord('b'): chosen_brush = (chosen_brush + 1) % len(brush_sizes)
    elif key == ord('s'): cv2.imwrite(f"drawing_{int(time.time())}.png", drawing); print("Saved!")

cam.release()
cv2.destroyAllWindows()
detector.close()
