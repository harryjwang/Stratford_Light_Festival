import cv2
import numpy as np
from ultralytics import YOLO

MODEL_NAME   = "yolov8n-seg.pt"   # segmentation model
SOURCE       = 0                   # webcam index or "video.mp4"
CONF_THRESH  = 0.5
LINE_THICK   = 2

# -------- Display / performance --------
FULLSCREEN      = True
DISPLAY_SIZE    = (1920, 1080)   # final window size; set to None to skip resize

# -------- Fixed non-repeating color palette (BGR) --------
PALETTE = [
    (0, 0, 255),     # Red
    (0, 51, 255),    # Bright Red
    (0, 255, 255),   # Yellow
    (255, 165, 0),   # Orange
    (0, 128, 255),   # Light Orange
    (0, 100, 255),   # Deep Orange
]

def distinct_color_from_index(k: int):
    """If we run out of PALETTE colors, generate a new distinct color deterministically."""
    hue = (k * 37) % 180  # OpenCV HSV hue in [0,180)
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr[0, 0])

def mask_centroid(mask_u8):
    m = cv2.moments(mask_u8, binaryImage=True)
    if m["m00"] < 1e-3:
        return None
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return (cx, cy)

def smoothstep(edge0, edge1, x):
    """Smooth S-curve mapping x from [edge0,edge1] to [0,1]."""
    if edge1 == edge0:
        return 0.0
    t = (x - edge0) / (edge1 - edge0)
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)

def main():
    cv2.setUseOptimized(True)

    # Create a fullscreen window once (prevents flicker)
    win_name = "People Outlines (Tracked + Outer Gradient Aura)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if FULLSCREEN:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    model = YOLO(MODEL_NAME)
    # model.to('cuda')  # optional GPU

    # Persistent color per tracked ID and pointer to next unused palette index
    id_colors = {}
    next_color_idx = 0

    # Per-ID state: centroid + smoothed brightness + smoothed radius
    # track_id -> {"last_center": (x,y), "strength": float, "radius": float}
    id_state = {}

    # ---- Gradient aura tuning ----
    MOTION_THRESH    = 2.0    # ignore tiny jitter (< px)
    MOTION_LOW       = 2.0    # motion where aura starts to grow (px)
    MOTION_HIGH      = 18.0   # motion where aura maxes out (px)

    STRENGTH_MIN     = 0.18
    STRENGTH_MAX     = 0.80
    STRENGTH_EMA     = 0.85   # higher = smoother/slower brightness changes

    RADIUS_BASE      = 30.0
    RADIUS_MAX       = 300.0
    RADIUS_EMA       = 0.55   # higher = smoother/slower radius changes

    FALL_OFF_LEN     = 40.0   # larger = slower fade
    GAMMA            = 1.35   # >1 softens near silhouette edge

    # ✅ Use camera index here (not a VideoCapture object)
    for result in model.track(
        source=SOURCE,
        classes=[0],           # person only
        conf=CONF_THRESH,
        persist=True,          # stable IDs across frames
        stream=True,
        verbose=False
    ):
        H, W = result.orig_img.shape[:2]
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            out = cv2.resize(frame, DISPLAY_SIZE) if DISPLAY_SIZE else frame
            cv2.imshow(win_name, out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            continue

        ids = result.boxes.id
        if ids is None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            temp_ids = [int(((x1 + x2) / 2) // 20 * 10000 + ((y1 + y2) / 2) // 20)
                        for x1, y1, x2, y2 in boxes_xyxy]
            ids = np.array(temp_ids)
        else:
            ids = ids.cpu().numpy().astype(int)

        masks = result.masks.data.cpu().numpy()  # [N, H, W]

        aura_layer = np.zeros((H, W, 3), dtype=np.float32)

        for i, mask in enumerate(masks):
            track_id = int(ids[i])

            if track_id not in id_colors:
                if next_color_idx < len(PALETTE):
                    id_colors[track_id] = PALETTE[next_color_idx]
                else:
                    id_colors[track_id] = distinct_color_from_index(next_color_idx - len(PALETTE))
                next_color_idx += 1
            color = np.array(id_colors[track_id], dtype=np.float32)

            mask_u8 = (mask * 255).astype(np.uint8)

            center = mask_centroid(mask_u8)
            if track_id not in id_state:
                id_state[track_id] = {
                    "last_center": center,
                    "strength": STRENGTH_MIN,
                    "radius": RADIUS_BASE
                }

            last_center = id_state[track_id]["last_center"]
            motion = 0.0
            if center is not None and last_center is not None:
                dx = center[0] - last_center[0]
                dy = center[1] - last_center[1]
                motion = (dx*dx + dy*dy) ** 0.5

            if motion < MOTION_THRESH:
                motion = 0.0

            t = smoothstep(MOTION_LOW, MOTION_HIGH, motion)
            target_strength = STRENGTH_MIN + t * (STRENGTH_MAX - STRENGTH_MIN)
            prev_strength = id_state[track_id]["strength"]
            strength = STRENGTH_EMA * prev_strength + (1.0 - STRENGTH_EMA) * target_strength
            id_state[track_id]["strength"] = strength

            target_radius = RADIUS_BASE + t * (RADIUS_MAX - RADIUS_BASE)
            prev_radius = id_state[track_id]["radius"]
            radius = RADIUS_EMA * prev_radius + (1.0 - RADIUS_EMA) * target_radius
            id_state[track_id]["radius"] = radius

            id_state[track_id]["last_center"] = center

            # Outer gradient aura only (inside stays black)
            inv = 255 - mask_u8
            dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)
            alpha = np.exp(-dist / FALL_OFF_LEN)
            alpha[dist > radius] = 0.0
            alpha[mask_u8 == 255] = 0.0
            if GAMMA != 1.0:
                alpha = np.power(alpha, GAMMA)

            alpha *= strength

            aura_layer[..., 0] += alpha * color[0]
            aura_layer[..., 1] += alpha * color[1]
            aura_layer[..., 2] += alpha * color[2]

            # Black interior + colored outline
            frame[mask_u8 == 255] = (0, 0, 0)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, tuple(int(c) for c in color), LINE_THICK)

        aura_layer = np.clip(aura_layer, 0, 255).astype(np.uint8)
        frame = np.maximum(frame, aura_layer)

        # Resize to fill the screen (keeps it smooth even if capture is small)
        out = cv2.resize(frame, DISPLAY_SIZE) if DISPLAY_SIZE else frame
        cv2.imshow(win_name, out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
