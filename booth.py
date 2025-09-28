# booth.py — five filters: none, b&w, rainbow, sepia, teal-orange (plus original = 0)
import os, time, math, cv2, numpy as np
from datetime import datetime
from cvzone.HandTrackingModule import HandDetector


# ----------------- setup -----------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.35, maxHands=1)

save_dir = "booth"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

shots = []
filter_id = 0
show_grid = True
pinching = False
t_pinch = 0.0
t_count = None
status = "idle"

# ----------------- filters -----------------
def apply_filter(img, fid):
    if fid == 0:
        # original
        return img
    elif fid == 1:
        # black & white
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    elif fid == 2:
        # rainbow
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        return cv2.applyColorMap(g, cv2.COLORMAP_RAINBOW)
    elif fid == 3:
        # sepia-like (using autumn colormap)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        return cv2.applyColorMap(g, cv2.COLORMAP_AUTUMN)
    elif fid == 4:
        # cool/teal (using winter colormap)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        return cv2.applyColorMap(g, cv2.COLORMAP_WINTER)
    elif fid == 5:
        # hot/fire look
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        return cv2.applyColorMap(g, cv2.COLORMAP_HOT)
    else:
        return img


# ----------------- draw helpers -----------------
def draw_grid(img, n=3, color=(140, 140, 140)):
    h, w = img.shape[:2]
    i = 1
    while i < n:
        x = w * i // n
        y = h * i // n
        cv2.line(img, (x, 0), (x, h), color, 1, cv2.LINE_AA)
        cv2.line(img, (0, y), (w, y), color, 1, cv2.LINE_AA)
        i += 1


def draw_hud(img, filter_id_value, status_text, shots_len):
    lines = [
        "pinch=shutter  thumbs-up=filter  palm=grid  fist=save",
        "filter:%s shots:%s  status:%s" % (str(filter_id_value), str(shots_len % 4), status_text),
        "save to: %s/   q:quit" % save_dir
    ]
    y = 34
    for t in lines:
        cv2.putText(img, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 220, 60), 2, cv2.LINE_AA)
        y += 32


def draw_thumbs(view, base_w, base_h, frames):
    tw = 150
    x0 = base_w - (tw + 10)
    last = frames[-3:][::-1]
    i = 0
    for fr in last:
        th = cv2.resize(fr, (tw, int(tw * base_h / base_w)))
        y = 10 + i * (th.shape[0] + 10)
        view[y:y + th.shape[0], x0:x0 + tw] = th
        i += 1


# ----------------- io helpers -----------------
def save_collage(imgs, path):
    s = 380
    pad = 10
    tiles = [cv2.resize(im, (s, s)) for im in imgs[:4]]

    canvas_h = 2 * s + 3 * pad
    canvas_w = 2 * s + 3 * pad
    canvas = np.empty((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = 245

    positions = [(pad, pad), (pad, 2 * pad + s), (2 * pad + s, pad), (2 * pad + s, 2 * pad + s)]
    for im, pos in zip(tiles, positions):
        y = pos[0]
        x = pos[1]
        canvas[y:y + s, x:x + s] = im

    cv2.imwrite(path, canvas)


def split_findhands_result(result, fallback_frame):
    if isinstance(result, tuple):
        a = result[0]
        if len(result) > 1:
            b = result[1]
        else:
            b = None

        if isinstance(a, np.ndarray):
            return a, b
        elif isinstance(b, np.ndarray):
            return b, a
        else:
            if b is not None:
                return fallback_frame, b
            else:
                return fallback_frame, a

    if isinstance(result, np.ndarray):
        return result, []

    return fallback_frame, result


# ----------------- loop -----------------
while True:
    ok, frame = cap.read()
    if not ok:
        blank = np.zeros((480, 640, 3), np.uint8)
        cv2.imshow("Gesture Photo Booth", blank)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        continue

    frame = cv2.flip(frame, 1)
    base = frame.copy()

    result = detector.findHands(frame, draw=True)
    frame_drawn, hands = split_findhands_result(result, frame)

    view = apply_filter(base, filter_id)
    if show_grid:
        draw_grid(view)

    if isinstance(hands, list) and len(hands) > 0:
        hand = hands[0]
    else:
        hand = None

    H, W = frame.shape[:2]
    diag = math.hypot(W, H)
    PINCH_ON = 0.055 * diag
    PINCH_OFF = 0.075 * diag

    now = time.time()
    status = "idle"

    if hand is None:
        cv2.putText(view, "no hand detected", (12, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        lm = hand.get("lmList", None)
        if lm is not None and len(lm) >= 9:
            p4 = (int(lm[4][0]), int(lm[4][1]))
            p8 = (int(lm[8][0]), int(lm[8][1]))
            cv2.circle(view, p4, 10, (0, 255, 255), 2)
            cv2.circle(view, p8, 10, (255, 255, 0), 2)

            dbg = detector.fingersUp(hand)
            cv2.putText(view, "fingers:%s" % str(dbg), (12, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

            # pinch → countdown
            d, _, _ = detector.findDistance(lm[4][:2], lm[8][:2], frame_drawn)
            if (not pinching) and (d < PINCH_ON):
                pinching = True
                t_pinch = now
                t_count = now
                status = "countdown"
            if pinching and (d > PINCH_OFF):
                pinching = False

            # gestures
            fingers = detector.fingersUp(hand)
            if fingers == [1, 0, 0, 0, 0]:
                # thumbs-up --> next filter
                filter_id = (filter_id + 1) % 6  # cycle 0..5
                status = "filter"
                time.sleep(1.0)
            elif fingers == [1, 1, 1, 1, 1]:
                # palm --> toggle grid
                show_grid = not show_grid
                status = "grid"
                time.sleep(1.0)
            elif fingers == [0, 0, 0, 0, 0] and len(shots) > 0:
                # fist --> save collage
                imgs = shots[-4:]
                while len(imgs) < 4:
                    imgs.append(imgs[-1])
                name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_manual.jpg"
                path = os.path.join(save_dir, name)
                save_collage(imgs, path)
                status = "saved"
                time.sleep(0.25)

    # countdown / capture
    if t_count is not None:
    # countdown display
        secs = int(now - t_count)
        left = max(1, 3 - secs)
        overlay = view.copy()
        cv2.putText(overlay, str(left), (W // 2 - 30, H // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 10, cv2.LINE_AA)
        view = cv2.addWeighted(overlay, 0.35, view, 0.65, 0)

        # take shot after 3 seconds
        if secs >= 3:
            shots.append(apply_filter(base, filter_id))
            t_count = None
            status = "shot %d/4" % ((len(shots) % 4) or 4)

            if len(shots) >= 4:
                fname = datetime.now().strftime("%Y%m%d_%H%M%S") + "_collage.jpg"
                save_collage(shots[-4:], os.path.join(save_dir, fname))
                status = "saved"
                time.sleep(0.25)


    draw_thumbs(view, W, H, shots)
    draw_hud(view, filter_id, status, len(shots))

    cv2.imshow("Gesture Photo Booth", view)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
