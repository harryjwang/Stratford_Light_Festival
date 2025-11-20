import cv2
import numpy as np
from ultralytics import YOLO

# ------------------------------ Tunables ------------------------------
MODEL_NAME   = "yolov8n-seg.pt"   # YOLOv8 segmentation model
SOURCE       = 0                  # camera index or "video.mp4"
CONF_THRESH  = 0.5
LINE_THICK   = 2

# Display
FULLSCREEN      = True
DISPLAY_SIZE    = (1920, 1080)    # final window size; set None to skip resize

# Performance knobs for Ultralytics
IMGSZ       = 640                 # inference size (smaller = faster)
VID_STRIDE  = 1                   # process every Nth frame (2/3 for speed)
MAX_DET     = 8                   # cap number of people per frame

# Color palette (BGR) - non-repeating per track ID
PALETTE = [
    (191, 131, 82),     # Light Blue-ish
    (100, 194, 236),    # Faint Yellow-ish
    (80, 132, 233),     # Faint Orange-ish
    (208, 79, 135),     # Light Purple
    (76, 157, 60),      # Magenta / Greenish
]

# ------------------------------ Helpers ------------------------------
def distinct_color_from_index(k: int):
    """If we run out of PALETTE colors, generate a new distinct color deterministically."""
    hue = (k * 37) % 180  # OpenCV HSV hue [0,180)
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
    if edge1 == edge0:
        return 0.0
    t = (x - edge0) / (edge1 - edge0)
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)

def draw_filled_human_silhouette(mask_u8, top_left, size):
    """
    Draw a filled, human-like silhouette into a binary mask (uint8, 0/255).

    - mask_u8: HxW uint8 array (modified in-place)
    - top_left: (x, y) top-left of the bounding area
    - size: (w, h) width and height of the bounding area
    """
    x, y = top_left
    w, h = size

    H, W = mask_u8.shape
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(10, min(w, W - x))
    h = max(30, min(h, H - y))

    cx = x + w // 2

    # Proportions relative to height
    head_r   = int(0.14 * h)
    head_cy  = y + int(0.2 * h)
    torso_top_y    = head_cy + head_r + int(0.05 * h)
    torso_bottom_y = y + int(0.65 * h)
    torso_w  = int(0.35 * w)

    # Head
    cv2.circle(mask_u8, (cx, head_cy), head_r, 255, -1)

    # Torso
    torso_left  = cx - torso_w // 2
    torso_right = cx + torso_w // 2
    cv2.rectangle(mask_u8,
                  (torso_left, torso_top_y),
                  (torso_right, torso_bottom_y),
                  255, -1)

    # Shoulders (wider bar)
    shoulder_y = torso_top_y + int(0.1 * h)
    shoulder_w = int(0.55 * w)
    shoulder_left  = cx - shoulder_w // 2
    shoulder_right = cx + shoulder_w // 2
    cv2.rectangle(mask_u8,
                  (shoulder_left, shoulder_y - int(0.05*h)),
                  (shoulder_right, shoulder_y + int(0.08*h)),
                  255, -1)

    # Hips
    hip_y = torso_bottom_y
    hip_w = int(0.3 * w)
    hip_left  = cx - hip_w // 2
    hip_right = cx + hip_w // 2
    cv2.rectangle(mask_u8,
                  (hip_left, hip_y),
                  (hip_right, hip_y + int(0.06*h)),
                  255, -1)

    # Legs
    leg_len = int(0.25 * h)
    leg_w   = int(0.18 * w)
    left_leg_x1  = cx - int(0.2 * w) - leg_w // 2
    left_leg_x2  = left_leg_x1 + leg_w
    right_leg_x1 = cx + int(0.2 * w) - leg_w // 2
    right_leg_x2 = right_leg_x1 + leg_w

    cv2.rectangle(mask_u8,
                  (left_leg_x1, hip_y),
                  (left_leg_x2, hip_y + leg_len),
                  255, -1)
    cv2.rectangle(mask_u8,
                  (right_leg_x1, hip_y),
                  (right_leg_x2, hip_y + leg_len),
                  255, -1)

# ------------------------------ Main ------------------------------
def main():
    cv2.setUseOptimized(True)

    # Fullscreen window
    win_name = "People Outlines (Tracked + Full-Screen Ripples + Ghost)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if FULLSCREEN:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    model = YOLO(MODEL_NAME)
    # model.to('cuda')  # optional if you have CUDA

    # Persistent state
    id_colors = {}
    next_color_idx = 0

    # track_id -> {"last_center": (x,y), "strength": float, "radius": float, "prev_mask_small": np.ndarray}
    id_state = {}

    # -------------------------- Motion/Aura tuning --------------------------
    MOTION_THRESH_CENTROID = 2.0
    MOTION_LOW             = 1.5
    MOTION_HIGH            = 8.0
    W_CENTER               = 0.4
    W_SHAPE                = 0.6
    SHAPE_TO_PX            = 80.0
    SHAPE_MASK_SIZE        = (64, 64)  # (w, h)

    STRENGTH_MIN     = 0.18
    STRENGTH_MAX     = 0.90
    STRENGTH_EMA     = 0.85

    RADIUS_BASE      = 30.0
    RADIUS_EMA       = 0.55

    FALL_OFF_LEN     = 45.0
    GAMMA            = 1.25

    # -------------------------- Ripple tuning (Gaussian-band rings) --------------------------
    WAVELENGTH_MAX = 150.0
    WAVELENGTH_MIN = 10.0
    AMP_MIN        = 0.30
    AMP_MAX        = 1.00
    SHARP_MIN      = 1.3
    SHARP_MAX      = 3.5
    SPEED_BASE     = 0.4
    SPEED_MAX      = 2.5
    BANDWIDTH_MIN  = 0.12
    BANDWIDTH_MAX  = 0.30

    # Ghost silhouette physics (when no humans)
    ghost_id = -1
    ghost_state = {
        "pos": (300, 300),
        "size": (160, 280),
        "velocity": (2, 2),
    }

    # Timer for phase animation
    t0 = cv2.getTickCount()

    def process_entity(mask_u8, track_id, color, aura_layer, frame,
                       H, W, t_sec, DIAG, RADIUS_MAX_DYNAMIC, ATTEN_MIN, ATTEN_MAX_DYNAMIC):
        """Apply motion-based ripples + aura + outline for a single mask."""
        nonlocal id_state

        # --- Motion: centroid + shape change ---
        center = mask_centroid(mask_u8)
        center_motion = 0.0
        m_small = cv2.resize(mask_u8, SHAPE_MASK_SIZE, interpolation=cv2.INTER_NEAREST)
        m_small = (m_small > 127).astype(np.uint8)

        if track_id not in id_state:
            id_state[track_id] = {
                "last_center": center,
                "strength": STRENGTH_MAX,
                "radius":   RADIUS_MAX_DYNAMIC,
                "prev_mask_small": m_small
            }
            prev_center = None
            prev_small  = None
        else:
            prev_center = id_state[track_id]["last_center"]
            prev_small  = id_state[track_id]["prev_mask_small"]

        if center is not None and prev_center is not None:
            dx = center[0] - prev_center[0]
            dy = center[1] - prev_center[1]
            center_motion = (dx*dx + dy*dy) ** 0.5
            if center_motion < MOTION_THRESH_CENTROID:
                center_motion = 0.0

        shape_motion_px = 0.0
        if prev_small is not None and prev_small.shape == m_small.shape:
            inter = (m_small & prev_small).sum()
            union = (m_small | prev_small).sum()
            if union > 0:
                shape_change = 1.0 - (inter / union)
                shape_motion_px = shape_change * SHAPE_TO_PX

        combined_motion = W_CENTER * center_motion + W_SHAPE * shape_motion_px

        # --- Map motion to brightness and radius (with smoothing) ---
        t = smoothstep(MOTION_LOW, MOTION_HIGH, combined_motion)

        target_strength = STRENGTH_MIN + t * (STRENGTH_MAX - STRENGTH_MIN)
        prev_strength = id_state[track_id]["strength"]
        strength = STRENGTH_EMA * prev_strength + (1.0 - STRENGTH_EMA) * target_strength
        id_state[track_id]["strength"] = strength

        target_radius = RADIUS_BASE + t * (RADIUS_MAX_DYNAMIC - RADIUS_BASE)
        prev_radius = id_state[track_id]["radius"]
        radius = RADIUS_EMA * prev_radius + (1.0 - RADIUS_EMA) * target_radius
        id_state[track_id]["radius"] = radius

        id_state[track_id]["last_center"] = center
        id_state[track_id]["prev_mask_small"] = m_small

        # --- Base gradient outside silhouette ---
        inv = 255 - mask_u8
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)

        alpha = np.exp(-dist / FALL_OFF_LEN)
        alpha[dist > radius] = 0.0
        alpha[mask_u8 == 255] = 0.0
        if GAMMA != 1.0:
            alpha = np.power(alpha, GAMMA)

        # --- Motion-scaled ripple parameters ---
        wavelength = WAVELENGTH_MAX - t * (WAVELENGTH_MAX - WAVELENGTH_MIN)
        amplitude  = AMP_MIN        + t * (AMP_MAX        - AMP_MIN)
        sharpness  = SHARP_MIN      + t * (SHARP_MAX      - SHARP_MIN)
        speed      = SPEED_BASE     + t * (SPEED_MAX      - SPEED_BASE)
        atten_len  = ATTEN_MIN      + t * (ATTEN_MAX_DYNAMIC - ATTEN_MIN)
        band_frac  = BANDWIDTH_MIN  + t * (BANDWIDTH_MAX  - BANDWIDTH_MIN)
        sigma_px   = max(1.0, band_frac * wavelength / 2.355)

        # Gaussian-band ripples
        phase_frac = (dist / max(1e-6, wavelength) - speed * t_sec) % 1.0
        d_px = np.minimum(phase_frac, 1.0 - phase_frac) * wavelength
        ring_profile = np.exp(-(d_px * d_px) / (2.0 * sigma_px * sigma_px))
        ring_profile *= np.exp(-dist / max(1.0, atten_len))

        alpha = (1.0 - amplitude) * alpha + amplitude * (alpha * ring_profile)
        alpha *= strength

        # Additive colorized aura
        color = np.array(color, dtype=np.float32)
        aura_layer[..., 0] += alpha * color[0]
        aura_layer[..., 1] += alpha * color[1]
        aura_layer[..., 2] += alpha * color[2]

        # Black interior + crisp colored outline
        frame[mask_u8 == 255] = (0, 0, 0)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, tuple(int(c) for c in color), LINE_THICK)

    # Tracking loop
    for result in model.track(
        source=SOURCE,
        classes=[0],
        conf=CONF_THRESH,
        imgsz=IMGSZ,
        vid_stride=VID_STRIDE,
        max_det=MAX_DET,
        persist=True,
        stream=True,
        verbose=False
    ):
        t_sec = (cv2.getTickCount() - t0) / cv2.getTickFrequency()

        H, W = result.orig_img.shape[:2]
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        aura_layer = np.zeros((H, W, 3), dtype=np.float32)

        DIAG = float(np.hypot(H, W))
        RADIUS_MAX_DYNAMIC = DIAG
        ATTEN_MIN          = 500.0
        ATTEN_MAX_DYNAMIC  = DIAG * 2.0

        # --------- CASE 1: No detections -> ghost human with ripples ----------
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            x, y = ghost_state["pos"]
            vx, vy = ghost_state["velocity"]
            w, h = ghost_state["size"]

            x += vx
            y += vy

            if x < 0 or x > W - w:
                vx = -vx
                x = max(0, min(x, W - w))
            if y < 0 or y > H - h:
                vy = -vy
                y = max(0, min(y, H - h))

            ghost_state["pos"] = (x, y)
            ghost_state["velocity"] = (vx, vy)

            ghost_mask = np.zeros((H, W), dtype=np.uint8)
            draw_filled_human_silhouette(ghost_mask, (x, y), (w, h))

            ghost_color = (230, 230, 255)  # pale bluish white
            process_entity(ghost_mask, ghost_id, ghost_color, aura_layer, frame,
                           H, W, t_sec, DIAG, RADIUS_MAX_DYNAMIC, ATTEN_MIN, ATTEN_MAX_DYNAMIC)

        # --------- CASE 2: Real detections -> original ripple logic ----------
        else:
            ids = result.boxes.id
            if ids is None:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                temp_ids = [int(((x1 + x2) / 2) // 20 * 10000 + ((y1 + y2) / 2) // 20)
                            for x1, y1, x2, y2 in boxes_xyxy]
                ids = np.array(temp_ids)
            else:
                ids = ids.cpu().numpy().astype(int)

            masks = result.masks.data.cpu().numpy()

            for i, mask in enumerate(masks):
                track_id = int(ids[i])

                if track_id not in id_colors:
                    if next_color_idx < len(PALETTE):
                        id_colors[track_id] = PALETTE[next_color_idx]
                    else:
                        id_colors[track_id] = distinct_color_from_index(next_color_idx - len(PALETTE))
                    next_color_idx += 1

                color = id_colors[track_id]
                mask_u8 = (mask * 255).astype(np.uint8)

                process_entity(mask_u8, track_id, color, aura_layer, frame,
                               H, W, t_sec, DIAG, RADIUS_MAX_DYNAMIC, ATTEN_MIN, ATTEN_MAX_DYNAMIC)

        # Composite aura over the black background + outlines
        aura_layer = np.clip(aura_layer, 0, 255).astype(np.uint8)
        frame = np.maximum(frame, aura_layer)

        out = cv2.resize(frame, DISPLAY_SIZE) if DISPLAY_SIZE else frame
        cv2.imshow(win_name, out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
