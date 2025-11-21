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
VID_STRIDE  = 1                   # process every Nth frame
MAX_DET     = 8                   # cap number of people per frame

# Color palette (BGR) - non-repeating per track ID
PALETTE = [
    (191, 131, 82),     # Light Blue-ish / Warm tone
    (100, 194, 236),    # Faint Yellow / light cyan
    (80, 132, 233),     # Faint Orange-ish blue
    (208, 79, 135),     # Light Purple
    (76, 157, 60),      # Magenta-ish green
]

# Ghost walking / animation
GHOST_WALK_SPEED  = 3.0   # radians per second for limb swing
GHOST_STEP_AMPL   = 0.7   # how much arms/legs swing (0.3–1.0)
GHOST_COLOR_SPEED = 0.15  # how fast hue cycles (bigger = faster color change)

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

def draw_walking_human_silhouette(mask_u8, top_left, size, phase, facing=1):
    """
    Draw a side-view walking human silhouette (thick-limbed stick figure) into mask_u8.

    - mask_u8: HxW uint8 mask, modified in-place (0/255)
    - top_left: (x, y) top-left of bounding region
    - size: (w, h) width/height of region
    - phase: float, walking phase in radians
    - facing: +1 for facing right, -1 for facing left
    """
    if facing == 0:
        facing = 1
    facing = 1 if facing > 0 else -1

    x, y = top_left
    w, h = size
    H, W = mask_u8.shape

    # Clamp region
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(40, min(w, W - x))
    h = max(80, min(h, H - y))

    cx = x + w // 2

    # Core body proportions (side view)
    head_radius  = int(0.12 * h)
    head_cy      = y + int(0.18 * h)

    shoulder_y   = y + int(0.32 * h)
    hip_y        = y + int(0.58 * h)
    foot_y       = y + int(0.90 * h)

    torso_thick  = max(4, int(0.06 * h))
    limb_thick   = max(4, int(0.05 * h))

    # Limb lengths
    arm_len      = (hip_y - shoulder_y) * 0.95
    leg_len      = foot_y - hip_y

    # Walking cycle: arms/legs swing with phase (criss-cross profile)
    step = GHOST_STEP_AMPL
    # Legs: one forward, one back
    leg_angle_front  = step * np.sin(phase)          # front leg
    leg_angle_back   = -step * np.sin(phase)         # back leg
    # Arms: opposite to legs for natural walk
    arm_angle_front  = -step * np.sin(phase + np.pi) # front arm
    arm_angle_back   = step * np.sin(phase + np.pi)  # back arm

    def limb_endpoint(root_x, root_y, length, angle_from_vertical):
        # angle_from_vertical > 0 = forward in facing direction
        dx = length * np.sin(angle_from_vertical) * facing
        dy = length * np.cos(angle_from_vertical)
        return int(root_x + dx), int(root_y + dy)

    roi = mask_u8

    # Slight head offset in facing direction so it "leans" into motion
    head_cx = cx + facing * int(0.06 * w)

    # Head
    cv2.circle(roi, (head_cx, head_cy), head_radius, 255, -1)

    # Torso (single vertical line for profile)
    cv2.line(roi, (cx, shoulder_y), (cx, hip_y), 255, torso_thick)

    # Root joints for arms and legs (profile: both from center)
    shoulder_x = cx
    hip_x      = cx

    # --- Compute arm endpoints ---
    front_hand_x, front_hand_y = limb_endpoint(shoulder_x, shoulder_y,
                                               arm_len, arm_angle_front)
    back_hand_x,  back_hand_y  = limb_endpoint(shoulder_x, shoulder_y,
                                               arm_len, arm_angle_back)

    # --- Compute leg endpoints ---
    front_foot_x, front_foot_y = limb_endpoint(hip_x, hip_y,
                                               leg_len, leg_angle_front)
    back_foot_x,  back_foot_y  = limb_endpoint(hip_x, hip_y,
                                               leg_len, leg_angle_back)

    foot_r = max(3, int(0.03 * h))

    # Draw order: back limbs first, then torso, then front limbs
    # so they visually "criss-cross" over the body

    # Back arm and leg
    cv2.line(roi, (shoulder_x, shoulder_y), (back_hand_x, back_hand_y), 255, limb_thick)
    cv2.line(roi, (hip_x, hip_y), (back_foot_x, back_foot_y),           255, limb_thick)
    cv2.circle(roi, (back_foot_x, back_foot_y), foot_r, 255, -1)

    # Torso again to reinforce overlap
    cv2.line(roi, (cx, shoulder_y), (cx, hip_y), 255, torso_thick)

    # Front arm and leg (on top)
    cv2.line(roi, (shoulder_x, shoulder_y), (front_hand_x, front_hand_y), 255, limb_thick)
    cv2.line(roi, (hip_x, hip_y), (front_foot_x, front_foot_y),          255, limb_thick)
    cv2.circle(roi, (front_foot_x, front_foot_y), foot_r, 255, -1)

# ------------------------------ Main ------------------------------
def main():
    cv2.setUseOptimized(True)

    win_name = "People Outlines (Tracked + Full-Screen Ripples + Walking Ghost)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if FULLSCREEN:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    model = YOLO(MODEL_NAME)

    id_colors = {}
    next_color_idx = 0
    id_state = {}

    # -------------------------- Motion/Aura tuning --------------------------
    MOTION_THRESH_CENTROID = 2.0
    MOTION_LOW             = 1.5
    MOTION_HIGH            = 8.0
    W_CENTER               = 0.4
    W_SHAPE                = 0.6
    SHAPE_TO_PX            = 80.0
    SHAPE_MASK_SIZE        = (64, 64)

    STRENGTH_MIN     = 0.18
    STRENGTH_MAX     = 0.90
    STRENGTH_EMA     = 0.85

    RADIUS_BASE      = 30.0
    RADIUS_EMA       = 0.55

    FALL_OFF_LEN     = 45.0
    GAMMA            = 1.25

    # Ripple tuning
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

    ghost_id = -1
    # Horizontal-only ghost:
    ghost_state = {
        "x": 300,             # current x (in pixels)
        "vx": 3.0,            # horizontal speed (pixels/frame)
        "size": (160, 280),   # (w, h)
        "y_frac": 0.6,        # vertical position as fraction of H (0=top, 1=bottom)
    }

    t0 = cv2.getTickCount()

    def process_entity(mask_u8, track_id, color, aura_layer, frame,
                       H, W, t_sec, DIAG, RADIUS_MAX_DYNAMIC, ATTEN_MIN, ATTEN_MAX_DYNAMIC):
        nonlocal id_state

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

        inv = 255 - mask_u8
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)

        alpha = np.exp(-dist / FALL_OFF_LEN)
        alpha[dist > radius] = 0.0
        alpha[mask_u8 == 255] = 0.0
        if GAMMA != 1.0:
            alpha = np.power(alpha, GAMMA)

        WAVELENGTH_MAX = 150.0
        WAVELENGTH_MIN = 10.0
        AMP_MIN        = 0.30
        AMP_MAX        = 1.00
        SPEED_BASE     = 0.4
        SPEED_MAX      = 2.5
        BANDWIDTH_MIN  = 0.12
        BANDWIDTH_MAX  = 0.30

        wavelength = WAVELENGTH_MAX - t * (WAVELENGTH_MAX - WAVELENGTH_MIN)
        amplitude  = AMP_MIN        + t * (AMP_MAX        - AMP_MIN)
        speed      = SPEED_BASE     + t * (SPEED_MAX      - SPEED_BASE)
        atten_len  = ATTEN_MIN      + t * (ATTEN_MAX_DYNAMIC - ATTEN_MIN)
        band_frac  = BANDWIDTH_MIN  + t * (BANDWIDTH_MAX  - BANDWIDTH_MIN)
        sigma_px   = max(1.0, band_frac * wavelength / 2.355)

        phase_frac = (dist / max(1e-6, wavelength) - speed * t_sec) % 1.0
        d_px = np.minimum(phase_frac, 1.0 - phase_frac) * wavelength
        ring_profile = np.exp(-(d_px * d_px) / (2.0 * sigma_px * sigma_px))
        ring_profile *= np.exp(-dist / max(1.0, atten_len))

        alpha = (1.0 - amplitude) * alpha + amplitude * (alpha * ring_profile)
        alpha *= strength

        color = np.array(color, dtype=np.float32)
        aura_layer[..., 0] += alpha * color[0]
        aura_layer[..., 1] += alpha * color[1]
        aura_layer[..., 2] += alpha * color[2]

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

        # --------- CASE 1: No detections -> walking ghost with GRADIENT color ----------
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            x = ghost_state["x"]
            vx = ghost_state["vx"]
            w, h = ghost_state["size"]
            y_frac = ghost_state["y_frac"]

            # Vertical position fixed based on frame height
            y = int(y_frac * H - h / 2)

            # Horizontal motion only
            x += vx

            if x < 0 or x > W - w:
                vx = -vx
                x = max(0, min(x, W - w))

            ghost_state["x"] = x
            ghost_state["vx"] = vx

            ghost_mask = np.zeros((H, W), dtype=np.uint8)
            phase = t_sec * GHOST_WALK_SPEED

            # Face in direction of horizontal velocity
            facing = 1 if vx >= 0 else -1

            draw_walking_human_silhouette(ghost_mask, (int(x), int(y)), (w, h), phase, facing=facing)

            # 🌈 Time-based gradient color for the ghost (hue cycles smoothly)
            hue = int((t_sec * 180 * GHOST_COLOR_SPEED) % 180)  # hue in [0,180)
            hsv = np.uint8([[[hue, 255, 255]]])
            ghost_color_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            ghost_color = (int(ghost_color_bgr[0]),
                           int(ghost_color_bgr[1]),
                           int(ghost_color_bgr[2]))

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
