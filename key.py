import os
import time
import math
import tkinter as tk

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pynput.keyboard import Controller, Key
from pynput.mouse import Controller as MouseController, Button as MouseButton

# ---------------- Settings ----------------
CAMERA_INDEX = 0          # try 0, 1, 2 if the window is blank or closes
FRAME_W, FRAME_H = 1280, 720
SEND_TO_OS = False        # True = also type into whatever app has focus (Notepad etc.)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Download it from:\n"
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )

# Real screen size
_root = tk.Tk()
_root.withdraw()
screen_w, screen_h = _root.winfo_screenwidth(), _root.winfo_screenheight()
_root.destroy()

# ---------------- Camera ----------------
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # CAP_DSHOW is Windows-only
cap.set(3, FRAME_W)
cap.set(4, FRAME_H)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}. Try another index.")

cv2.namedWindow('Virtual Keyboard', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Virtual Keyboard', 1600, 900)

# ---------------- Hand landmarker ----------------
options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.8,
)
landmarker = vision.HandLandmarker.create_from_options(options)
t0 = time.time()
last_ts = -1

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

def draw_hand(img, pts):
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(img, p, 4, (0, 0, 255), cv2.FILLED)

# ---------------- Controllers ----------------
keyboard = Controller()
os_mouse = MouseController()

frameR = 100
plocX, plocY = 0, 0
clocX, clocY = 0, 0
smoothing = 5

PINCH_CONFIRM_FRAMES = 5
CLICK_COOLDOWN_FRAMES = 20

pinch_frame_count = 0
mouse_click_cooldown = 0

# ---------------- Keyboard layout ----------------
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

class Button():
    def __init__(self, pos, text, size=[50, 50]):
        self.pos = pos
        self.size = size
        self.text = text

buttonList = []

x_start = 640
y_start = 180

for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([60 * j + x_start, 60 * i + y_start], key, size=[50, 50]))

row4_y = y_start + 60 * 3
buttonList.append(Button([x_start, row4_y], "Clear", size=[110, 50]))
buttonList.append(Button([x_start + 120, row4_y], "Space", size=[350, 50]))
buttonList.append(Button([x_start + 480, row4_y], "Back", size=[110, 50]))

kb_x = x_start - 20
kb_y = y_start - 20
kb_w = 60 * 10 - 10 + 40
kb_h = 60 * 4 - 10 + 40

# Text display box (above the keyboard)
tb_x, tb_y, tb_w, tb_h = kb_x, 40, kb_w, 100

typed_text = ""

def draw_glass(img, x, y, w, h):
    if y + h > img.shape[0] or x + w > img.shape[1] or y < 0 or x < 0:
        return
    roi = img[y:y+h, x:x+w]
    blur = cv2.GaussianBlur(roi, (45, 45), 0)
    color_rect = np.full(roi.shape, (255, 255, 255), dtype=np.uint8)
    glass = cv2.addWeighted(blur, 0.95, color_rect, 0.05, 0)
    img[y:y+h, x:x+w] = glass
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

def draw_text_box(img, text):
    draw_glass(img, tb_x, tb_y, tb_w, tb_h)
    cv2.putText(img, "Typed text:", (tb_x + 10, tb_y + 25),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 200, 200), 1)

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    max_w = tb_w - 40
    shown = text
    # keep only the end of the text that fits in the box
    while shown and cv2.getTextSize(shown, font, scale, thick)[0][0] > max_w:
        shown = shown[1:]

    cv2.putText(img, shown, (tb_x + 10, tb_y + 72), font, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, shown, (tb_x + 10, tb_y + 72), font, scale, (255, 255, 255), thick)

    # blinking cursor
    if int(time.time() * 2) % 2 == 0:
        tw = cv2.getTextSize(shown, font, scale, thick)[0][0]
        cx = tb_x + 14 + tw
        cv2.line(img, (cx, tb_y + 45), (cx, tb_y + 80), (0, 255, 255), 2)

clicked = False
delay_counter = 0
printed_shape = False

# ---------------- Main loop ----------------
while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame - check camera index")
        break

    img = cv2.flip(img, 1)
    img = cv2.resize(img, (FRAME_W, FRAME_H))

    if not printed_shape:
        print("Frame shape:", img.shape)
        printed_shape = True

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)

    timestamp_ms = int((time.time() - t0) * 1000)
    if timestamp_ms <= last_ts:
        timestamp_ms = last_ts + 1
    last_ts = timestamp_ms

    results = landmarker.detect_for_video(mp_image, timestamp_ms)

    hovered_button = None
    is_kb_pinching = False

    h, w, c = img.shape

    for handLms, handedness in zip(results.hand_landmarks, results.handedness):
        label = handedness[0].category_name

        pts = [(int(p.x * w), int(p.y * h)) for p in handLms]
        draw_hand(img, pts)

        x1, y1 = pts[8]   # index fingertip
        x2, y2 = pts[4]   # thumb tip
        distance = math.hypot(x2 - x1, y2 - y1)

        if label == "Right":
            for button in buttonList:
                bx, by = button.pos
                bw, bh = button.size
                if bx < x1 < bx + bw and by < y1 < by + bh:
                    hovered_button = button
                    break

            pointer_color = (255, 255, 255)
            if distance < 40:
                is_kb_pinching = True
                pointer_color = (0, 255, 0)

            cv2.circle(img, (x1, y1), 8, pointer_color, cv2.FILLED)
            cv2.circle(img, (x2, y2), 8, pointer_color, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), pointer_color, 3)

        elif label == "Left":
            screen_x = np.interp(x1, (frameR, w - frameR), (0, screen_w))
            screen_y = np.interp(y1, (frameR, h - frameR), (0, screen_h))

            clocX = plocX + (screen_x - plocX) / smoothing
            clocY = plocY + (screen_y - plocY) / smoothing

            os_mouse.position = (clocX, clocY)
            plocX, plocY = clocX, clocY

            mouse_color = (255, 200, 0)
            is_pinching_now = distance < 40

            if mouse_click_cooldown > 0:
                mouse_click_cooldown -= 1

            if is_pinching_now:
                pinch_frame_count += 1
                mouse_color = (0, 100, 255)

                if pinch_frame_count == PINCH_CONFIRM_FRAMES and mouse_click_cooldown == 0:
                    mouse_color = (0, 0, 255)
                    os_mouse.click(MouseButton.left, 1)
                    mouse_click_cooldown = CLICK_COOLDOWN_FRAMES
                elif pinch_frame_count >= PINCH_CONFIRM_FRAMES:
                    mouse_color = (0, 0, 200)
            else:
                pinch_frame_count = 0

            cv2.circle(img, (x1, y1), 10, mouse_color, cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, mouse_color, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), mouse_color, 4)

    draw_glass(img, kb_x, kb_y, kb_w, kb_h)
    draw_text_box(img, typed_text)

    for button in buttonList:
        bx, by = button.pos
        bw, bh = button.size

        is_hovered = (button == hovered_button)
        btn_clicked = (is_hovered and is_kb_pinching)

        alpha = 0.0
        color = (255, 255, 255)
        if btn_clicked:
            color = (0, 255, 0)
            alpha = 0.4
        elif is_hovered:
            alpha = 0.2

        if alpha > 0 and (by + bh <= img.shape[0] and bx + bw <= img.shape[1]):
            overlay = img[by:by+bh, bx:bx+bw].copy()
            color_rect = np.full(overlay.shape, color, dtype=np.uint8)
            blend = cv2.addWeighted(overlay, 1 - alpha, color_rect, alpha, 0)
            img[by:by+bh, bx:bx+bw] = blend
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
        else:
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (255, 255, 255), 1)

        if len(button.text) > 1:   # Space, Clear, Back (centered text)
            tw = cv2.getTextSize(button.text, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0][0]
            text_x = bx + (bw - tw) // 2
            cv2.putText(img, button.text, (text_x + 2, by + 34), cv2.FONT_HERSHEY_PLAIN, 2, (100, 100, 100), 2)
            cv2.putText(img, button.text, (text_x, by + 32), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        else:
            cv2.putText(img, button.text, (bx + 14, by + 36), cv2.FONT_HERSHEY_PLAIN, 2, (100, 100, 100), 2)
            cv2.putText(img, button.text, (bx + 12, by + 34), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        if btn_clicked and not clicked:
            if button.text == "Space":
                typed_text += " "
                if SEND_TO_OS:
                    keyboard.press(Key.space)
                    keyboard.release(Key.space)
            elif button.text == "Back":
                typed_text = typed_text[:-1]
                if SEND_TO_OS:
                    keyboard.press(Key.backspace)
                    keyboard.release(Key.backspace)
            elif button.text == "Clear":
                typed_text = ""
            else:
                ch = button.text.lower()
                typed_text += ch
                if SEND_TO_OS:
                    keyboard.press(ch)
                    keyboard.release(ch)

            clicked = True
            delay_counter = 0

    if clicked:
        delay_counter += 1
        if delay_counter > 15:
            clicked = False

    cv2.imshow("Virtual Keyboard", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()