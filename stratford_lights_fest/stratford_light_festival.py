import cv2
import numpy as np
from ultralytics import YOLO

MODEL_NAME   = "yolov8n-seg.pt"   # segmentation model
SOURCE       = 1                   # webcam index or "video.mp4"
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

    win_name = "People Outlines (Tracked + Motion-Scaled Ripples)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if FULLSCREEN:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    model = YOLO(MODEL_NAME)
    # model.to('cuda')  # optional GPU

    # Persistent color per tracked ID and pointer to next unused palette index
    id_colors = {}
    next_color_idx = 0

    # Per-ID state: centroid + smoothed brightness + smoothed radius + prev small mask
    #   track_id -> {"last_center": (x,y),
    #   "strength": float,
    #   "radius": float,
    #   "prev_mask_small": np.ndarray[uint8] (h_s, w_s)}
    id_state = {}

    # ---- Motion/Aura tuning ----
    # Combine center motion (pixels) + shape motion (mask IoU change)
    MOTION_THRESH_CENTROID = 2.0   # px: ignore tiny centroid jitter
    MOTION_LOW             = 2.0   # combined motion where aura starts to grow
    MOTION_HIGH            = 18.0  # combined motion where aura maxes out

    # Balance center vs. shape motion
    W_CENTER = 0.4
    W_SHAPE  = 0.6

    # Convert shape change (0..1) to "pixel-equivalent" scale so it mixes well with centroid px
    SHAPE_TO_PX = 40.0  # ↑ makes hand/arm motion influence larger

    # Small mask used for shape motion (fast + robust)
    SHAPE_MASK_SIZE = (64, 64)  # (w, h). Keep small for speed.

    # Brightness (opacity) smoothing
    STRENGTH_MIN     = 0.18
    STRENGTH_MAX     = 0.90   # a bit brighter to make ripples pop
    STRENGTH_EMA     = 0.85   # higher = smoother/slower brightness changes

    # Radius grows with motion, also smoothed
    RADIUS_BASE      = 30.0
    RADIUS_MAX       = 320.0  # allow a bit wider reach
    RADIUS_EMA       = 0.55   # higher = smoother/slower radius changes

    # Base gradient shape
    FALL_OFF_LEN     = 45.0   # larger = slower fade
    GAMMA            = 1.25   # >1 softens near silhouette edge

    # -------- Ripple animation timer --------
    t0 = cv2.getTickCount()

    # Built-in tracker with persistent IDs
    for result in model.track(
        source=SOURCE,
        classes=[0],           # person only
        conf=CONF_THRESH,
        persist=True,          # stable IDs across frames
        stream=True,
        verbose=False
    ):
        # time (s) for ripple phase
        t_sec = (cv2.getTickCount() - t0) / cv2.getTickFrequency()

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

        masks = result.masks.data.cpu().numpy()  # [N, H, W], ~{0,1}
        aura_layer = np.zeros((H, W, 3), dtype=np.float32)

        for i, mask in enumerate(masks):
            track_id = int(ids[i])

            # Assign color (no reuse)
            if track_id not in id_colors:
                if next_color_idx < len(PALETTE):
                    id_colors[track_id] = PALETTE[next_color_idx]
                else:
                    id_colors[track_id] = distinct_color_from_index(next_color_idx - len(PALETTE))
                next_color_idx += 1
            color = np.array(id_colors[track_id], dtype=np.float32)

            # Binary mask (uint8)
            mask_u8 = (mask * 255).astype(np.uint8)

            # --- Compute center motion (px) ---
            center = mask_centroid(mask_u8)
            center_motion = 0.0
            prev_center = None

            # --- Compute shape motion via small-mask IoU change ---
            m_small = cv2.resize(mask_u8, SHAPE_MASK_SIZE, interpolation=cv2.INTER_NEAREST)
            m_small = (m_small > 127).astype(np.uint8)  # binarize 0/1

            shape_motion_px = 0.0
            prev_small = None

            if track_id not in id_state:
                id_state[track_id] = {
                    "last_center": center,
                    "strength": STRENGTH_MIN,
                    "radius": RADIUS_BASE,
                    "prev_mask_small": m_small
                }
            else:
                prev_center = id_state[track_id]["last_center"]
                prev_small  = id_state[track_id]["prev_mask_small"]

            # Center motion
            if center is not None and prev_center is not None:
                dx = center[0] - prev_center[0]
                dy = center[1] - prev_center[1]
                center_motion = (dx*dx + dy*dy) ** 0.5
                if center_motion < MOTION_THRESH_CENTROID:
                    center_motion = 0.0

            # Shape motion: 1 - IoU between current and previous small masks
            if prev_small is not None and prev_small.shape == m_small.shape:
                inter = (m_small & prev_small).sum()
                union = (m_small | prev_small).sum()
                if union > 0:
                    shape_change = 1.0 - (inter / union)  # 0=no change, 1=totally different
                    shape_motion_px = shape_change * SHAPE_TO_PX

            # Combined motion
            combined_motion = W_CENTER * center_motion + W_SHAPE * shape_motion_px

            # ---- Map combined_motion -> strength & radius (with smoothing) ----
            t = smoothstep(MOTION_LOW, MOTION_HIGH, combined_motion)

            # Brightness
            target_strength = STRENGTH_MIN + t * (STRENGTH_MAX - STRENGTH_MIN)
            prev_strength = id_state[track_id]["strength"]
            strength = STRENGTH_EMA * prev_strength + (1.0 - STRENGTH_EMA) * target_strength
            id_state[track_id]["strength"] = strength

            # Radius
            target_radius = RADIUS_BASE + t * (RADIUS_MAX - RADIUS_BASE)
            prev_radius = id_state[track_id]["radius"]
            radius = RADIUS_EMA * prev_radius + (1.0 - RADIUS_EMA) * target_radius
            id_state[track_id]["radius"] = radius

            # Update state
            id_state[track_id]["last_center"] = center
            id_state[track_id]["prev_mask_small"] = m_small

            # --- Outer gradient aura only (inside stays black) ---
            inv = 255 - mask_u8  # background=255, inside=0
            dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)

            # Base gradient alpha (decays with distance, clamped by radius)
            alpha = np.exp(-dist / FALL_OFF_LEN)
            alpha[dist > radius] = 0.0
            alpha[mask_u8 == 255] = 0.0  # inside stays black
            if GAMMA != 1.0:
                alpha = np.power(alpha, GAMMA)

            # -------- Motion-scaled ripple parameters (THIS is the big change) --------
            # More motion -> more rings (smaller wavelength), stronger & crisper ripples, a bit faster
            # -------- Motion-scaled ripple parameters (updated for more rings & propagation) --------
            # More motion -> tighter wavelength (more rings), higher amplitude, sharper edges, faster speed, rings spread farther

            WAVELENGTH_MAX = 25.0   # very wide spacing at rest
            WAVELENGTH_MIN = 10.0    # many tight rings when moving a lot  (smaller = more rings)
            AMP_MIN        = 0.85    # subtle at rest
            AMP_MAX        = 0.95    # very strong at high motion
            SHARP_MIN      = 1.2     # soft edges at rest
            SHARP_MAX      = 3.0     # crisp edges at high motion
            SPEED_BASE     = 1.0     # Hz at rest
            SPEED_MAX      = 3.5     # much faster ripples at high motion
            ATTEN_MIN      = 750.0   # rings fade sooner when still
            ATTEN_MAX      = 2500.0   # rings propagate much farther when moving

            # Scale parameters by motion "t" (0..1)
            wavelength = WAVELENGTH_MAX - t * (WAVELENGTH_MAX - WAVELENGTH_MIN)
            amplitude  = AMP_MIN        + t * (AMP_MAX        - AMP_MIN)
            sharpness  = SHARP_MIN      + t * (SHARP_MAX      - SHARP_MIN)
            speed      = SPEED_BASE     + t * (SPEED_MAX      - SPEED_BASE)
            atten_len  = ATTEN_MIN      + t * (ATTEN_MAX      - ATTEN_MIN)

            # --- Ripple modulation ---
            phase = (2.0 * np.pi) * ((dist / wavelength) - (speed * t_sec))
            rings = 0.5 * (1.0 + np.cos(phase))          # 0..1
            rings = np.power(rings, sharpness)           # sharpen edges
            rings *= np.exp(-dist / max(1.0, atten_len)) # fade with distance

            # Blend rings into base gradient
            alpha = (1.0 - amplitude) * alpha + amplitude * (alpha * rings)


            # Scale by smoothed per-ID brightness
            alpha *= strength  # (H, W) in [0,1]

            # Additive colorized aura
            aura_layer[..., 0] += alpha * color[0]
            aura_layer[..., 1] += alpha * color[1]
            aura_layer[..., 2] += alpha * color[2]

            # --- Black silhouette + colored outline (crisp) ---
            frame[mask_u8 == 255] = (0, 0, 0)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, tuple(int(c) for c in color), LINE_THICK)

        aura_layer = np.clip(aura_layer, 0, 255).astype(np.uint8)
        frame = np.maximum(frame, aura_layer)

        out = cv2.resize(frame, DISPLAY_SIZE) if DISPLAY_SIZE else frame
        cv2.imshow(win_name, out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()