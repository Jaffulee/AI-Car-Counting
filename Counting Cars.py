import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



def intersection_area_xyxy(boxA, boxB):
    """
    Compute intersection area between two bounding boxes in XYXY format.
    boxA = (x1A, y1A, x2A, y2A)
    boxB = (x1B, y1B, x2B, y2B)
    Returns intersection area as a float.
    """
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB

    interW = max(0, min(x2A, x2B) - max(x1A, x1B))
    interH = max(0, min(y2A, y2B) - max(y1A, y1B))
    return interW * interH


def iou_xyxy(boxA, boxB):
    """
    Compute IoU (intersection-over-union) between two bounding boxes in XYXY format.
    Returns IoU as a float between [0, 1].
    """
    interArea = intersection_area_xyxy(boxA, boxB)
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB

    areaA = abs(x2A - x1A) * abs(y2A - y1A)
    areaB = abs(x2B - x1B) * abs(y2B - y1B)

    union = areaA + areaB - interArea
    if union == 0:
        return 0.0
    return interArea / union


def iol_xyxy(boxA, boxB):
    """
    Compute intersection-over-left (IoL) where 'left' refers to boxA:
    IoL = intersection_area(boxA, boxB) / area(boxA)
    Returns float in [0, 1].
    """
    interArea = intersection_area_xyxy(boxA, boxB)
    x1A, y1A, x2A, y2A = boxA
    areaA = abs(x2A - x1A) * abs(y2A - y1A)
    if areaA == 0:
        return 0.0
    return interArea / areaA


def create_inner_border(width, height, border_padding_scale):
    """
    Creates an inner border for detecting vehicles leaving/entering.
    Leaves border_padding_scale percent of the image outside the inner border.
    Returns bbox in XYXY int format: (x1, y1, x2, y2).
    """
    border_scale_multiplier = 1.0 - border_padding_scale
    border_width = width * border_scale_multiplier
    border_height = height * border_scale_multiplier
    borderx1 = (width - border_width) / 2
    borderx2 = width - borderx1
    bordery1 = (height - border_height) / 2
    bordery2 = height - bordery1
    return tuple(map(int, (borderx1, bordery1, borderx2, bordery2)))


def cluster_overlaps(dets, overlap_threshold):
    """
    Build overlap clusters (connected components) based on IoU > overlap_threshold.
    Returns list of clusters, each is a list of indices into dets.
    """
    n = len(dets)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if iou_xyxy(dets[i]["xyxy"], dets[j]["xyxy"]) > overlap_threshold:
                union(i, j)

    clusters = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)
    return list(clusters.values())


def shift_xyxy(box, shift_xy):
    """Shift an xyxy box by (dx, dy)."""
    x1, y1, x2, y2 = box
    dx, dy = shift_xy
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def clamp_vec2(vec, max_norm):
    """Clamp a 2D vector magnitude to max_norm."""
    norm = float(np.linalg.norm(vec))
    if norm == 0.0 or max_norm is None:
        return vec
    if norm <= max_norm:
        return vec
    return vec * (max_norm / norm)


def predicted_xyxy_from_last_seen(
    last_state,
    current_frame_idx,
    extrapolation_alpha,
    max_gap_frames=None,
    max_shift_px=None,
):
    """
    Predict where the last_seen box would be at current_frame_idx using constant velocity.

    alpha = 0 -> no extrapolation (original last_seen box)
    alpha = 1 -> full linear extrapolation

    Stability:
      - only extrapolate if frames_gap <= max_gap_frames (if set)
      - clamp total shift magnitude to max_shift_px (if set)
    """
    base_box = last_state["xyxy"]
    frames_gap = current_frame_idx - last_state["last_frame_seen"]
    if frames_gap <= 0:
        return base_box

    if max_gap_frames is not None and frames_gap > max_gap_frames:
        return base_box

    vel = last_state.get("vel_per_frame", None)
    if vel is None:
        return base_box

    shift = extrapolation_alpha * vel * frames_gap

    if max_shift_px is not None:
        shift = clamp_vec2(shift, max_shift_px)

    dx, dy = float(shift[0]), float(shift[1])
    return tuple(map(float, shift_xyxy(base_box, (dx, dy))))


def choose_det_from_cluster_using_last_seen(
    cluster_idxs,
    dets,
    last_seen,
    continuity_iou_threshold,
    current_frame_idx,
    extrapolation_alpha,
):
    """
    Choose ONE detection from a same-frame overlap cluster using ONLY last_seen (per-ID state):

    1) If any candidate's model_id exists in last_seen:
       - keep the candidate that best overlaps its own predicted last_seen box.

    2) Else (none of the candidate IDs exist in last_seen):
       - if the cluster overlaps any predicted last_seen box sufficiently, overwrite the chosen detection's ID
         with the best-matching last_seen ID.

    3) Else:
       - keep highest confidence.

    Returns (chosen_idx, overwrite_id_or_None).
    """

    # 1) Prefer candidate with known ID in last_seen and best IoU to its own predicted box
    best_idx = None
    best_score = -1.0

    for idx in cluster_idxs:
        cid = dets[idx]["model_id"]
        if cid in last_seen:
            pred_box = predicted_xyxy_from_last_seen(
                last_seen[cid],
                current_frame_idx,
                extrapolation_alpha,
                max_gap_frames=MAX_EXTRAPOLATION_GAP_FRAMES,
                max_shift_px=MAX_EXTRAPOLATION_SHIFT_PX,
            )
            iou_score = iou_xyxy(dets[idx]["xyxy"], pred_box)
            score = iou_score + 1e-6 * dets[idx].get("conf", 0.0)  # tie-break
            if score > best_score:
                best_score = score
                best_idx = idx

    if best_idx is not None:
        return best_idx, None

    # 2) Try to match cluster to ANY last_seen box and overwrite
    best_last_id = None
    best_last_iou = 0.0

    if last_seen:  # only if we have any history at all
        for idx in cluster_idxs:
            for last_id, st in last_seen.items():
                pred_box = predicted_xyxy_from_last_seen(
                    st,  # use this last_seen state, not last_seen[cid]
                    current_frame_idx,
                    extrapolation_alpha,
                    max_gap_frames=MAX_EXTRAPOLATION_GAP_FRAMES,
                    max_shift_px=MAX_EXTRAPOLATION_SHIFT_PX,
                )
                iou_score = iou_xyxy(dets[idx]["xyxy"], pred_box)
                if iou_score > best_last_iou:
                    best_last_iou = iou_score
                    best_last_id = last_id


    if best_last_id is not None and best_last_iou > continuity_iou_threshold:
        chosen_idx = max(cluster_idxs, key=lambda k: dets[k].get("conf", 0.0))
        return chosen_idx, best_last_id

    # 3) Fallback
    chosen_idx = max(cluster_idxs, key=lambda k: dets[k].get("conf", 0.0))
    return chosen_idx, None

def estimate_lanes_elbow_curvature(
    points_xy,
    k_max=8,
    random_state=0,
    use_log_inertia=False,
):
    """
    Estimate number of clusters using KMeans inertia + max curvature.

    Parameters
    ----------
    points_xy : array-like (N,2)
        Event center coordinates.
    k_max : int
        Maximum k to test.
    random_state : int
        KMeans random seed.
    use_log_inertia : bool
        If True, compute curvature on log(inertia).
        If False, use raw inertia.

    Returns
    -------
    lanes_est : float
        Possibly non-integer estimated number of lanes.
    chosen_k : int
        Integer k at maximum curvature.
    inertia_by_k : dict[int, float]
        Raw inertia values for diagnostics.
    """
    points_xy = np.asarray(points_xy, dtype=float)
    n = points_xy.shape[0]

    if n < 3:
        return np.nan, 0, {}

    k_max_eff = int(min(k_max, n))
    if k_max_eff < 2:
        return np.nan, 0, {}

    # Standardise coordinates
    X = StandardScaler().fit_transform(points_xy)

    ks = list(range(1, k_max_eff + 1))
    inertias = []

    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(X)
        inertias.append(float(km.inertia_))

    inertias = np.array(inertias, dtype=float)
    inertia_by_k = {k: v for k, v in zip(ks, inertias)}

    # Choose curve for curvature analysis
    if use_log_inertia:
        curve = np.log(inertias + 1e-9)  # numerical safety
    else:
        curve = inertias

    # Need at least k >= 3 for curvature
    if k_max_eff < 3:
        return float(1.0), 1, inertia_by_k

    # Second discrete difference (curvature proxy)
    d2 = np.zeros(len(curve))
    for i in range(1, len(curve) - 1):
        d2[i] = curve[i - 1] - 2 * curve[i] + curve[i + 1]

    best_i = int(np.argmax(np.abs(d2)))
    k_star = ks[best_i]

    # ---- Non-integer refinement via quadratic fit ----
    lanes_est = float(k_star)
    if 1 <= best_i <= len(curve) - 2:
        y1, y2, y3 = curve[best_i - 1], curve[best_i], curve[best_i + 1]
        denom = (y1 - 2 * y2 + y3)
        if denom != 0:
            x_vertex = 0.5 * (y1 - y3) / denom
            lanes_est = k_star + x_vertex

    lanes_est = float(np.clip(lanes_est, 1.0, k_max_eff))
    return lanes_est, k_star, inertia_by_k


# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------

# Project-relative folders (assumes you run from repo root)
DATA_DIR = Path("data")
OUTPUT_DATA_DIR = Path("output_data")
OUTPUT_VIDEO_DIR = Path("output_video")

OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

DRAW_VELOCITY_ARROW = True
ARROW_SCALE = 2.5          # pixels of arrow length per (px/frame) speed unit
ARROW_MIN_LEN = 6          # minimum arrow length to be visible
ARROW_MAX_LEN = 80         # cap so fast cars don't draw huge arrows

EXTRAPOLATION_ALPHA = 1.0  # 0.0 = original, 1.0 = fully extrapolated

# -------------------------
# Stability improvements
# -------------------------

# Only extrapolate if last_seen is recent (frames gap <= this)
MAX_EXTRAPOLATION_GAP_FRAMES = 15

# Clamp the total extrapolation shift magnitude (pixels)
MAX_EXTRAPOLATION_SHIFT_PX = 250.0

# Velocity smoothing (EMA). 0 = no smoothing, 0.7-0.9 = strong smoothing
VEL_EMA_ALPHA = 0.7

# Reject velocity spikes: cap speed in px/frame (None to disable)
MAX_SPEED_PX_PER_FRAME = 80.0

# Optional: require a minimum IoU to overwrite an ID from last_seen when cluster IDs are all new
MIN_OVERWRITE_IOU = 0.5  # keep using your existing threshold

# Loop over all videos in data/
video_paths = sorted(DATA_DIR.glob("*.mp4"))
if not video_paths:
    raise FileNotFoundError(f"No .mp4 files found in {DATA_DIR.resolve()}")

# YOLO model (uses tracking)
model = YOLO("yolov10x.pt")

allowed_labels = ["car", "truck", "bus", "motorcycle", "train"]

# Overlap logic (same-frame)
same_frame_overlap_threshold = 0.6

# Continuity logic (use last_seen overlap, not last-frame)
continuity_iou_threshold = MIN_OVERWRITE_IOU # used when deciding to overwrite ID from last_seen

# Border logic
border_padding_scale = 0.4
border_overlap_threshold = 0.1  # threshold on IoL to decide in_box

# Flush IDs not detected for N frames
FLUSH_FRAMES = 90  # e.g., ~3s at 30fps

# Cull/metrics
cull_time_seconds = 60 * 60  # kept for parity; your counting uses frames-since-event too

# # Video writer codec
# FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

all_df_full = []
all_df_fact = []
all_df_dim = []
all_df_agg = []

for output_ID, video_path in enumerate(video_paths, start=1):
    video_name = video_path.stem
    print(f"\nProcessing video {output_ID}: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length_s = (frame_count - 1) / fps if fps else 0.0  # signpost-safe
    print(f"fps={fps}, width={width}, height={height}, frames={frame_count}")

    # Border bbox
    bbox = create_inner_border(width, height, border_padding_scale)
    bx1, by1, bx2, by2 = bbox

    # Output paths (prefix with original filename)
    output_video_path = OUTPUT_VIDEO_DIR / f"{video_name}_tracked.mp4"
    output_fact_csv_path = OUTPUT_DATA_DIR / f"{video_name}_Cars_detected_fact.csv"
    output_dim_csv_path = OUTPUT_DATA_DIR / f"{video_name}_Cars_detected_dim.csv"
    output_total_csv_path = OUTPUT_DATA_DIR / f"{video_name}_Cars_detected.csv"
    output_agg_csv_path = OUTPUT_DATA_DIR / f"{video_name}_Cars_detected_agg.csv"

    # Defensive fps fallback (important for MP4 validity)
    if not fps or fps <= 0:
        fps = 30.0

    # Base output path (without extension fixed yet)
    base_output_path = OUTPUT_VIDEO_DIR / f"{video_name}_tracked"

    writer_opened = False
    out = None

    # Try codecs in order of preference
    for ext, fourcc_str in [
        # ("mp4", "avc1"),   # H.264 (best compatibility if available)
        ("mp4", "mp4v"),   # MPEG-4 Part 2 (your previous behaviour)
        ("avi", "MJPG"),   # Very robust fallback
    ]:
        candidate_path = base_output_path.with_suffix(f".{ext}")
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out = cv2.VideoWriter(str(candidate_path), fourcc, fps, (width, height), True)

        if out.isOpened():
            output_video_path = candidate_path
            print(f"VideoWriter opened: {output_video_path.name} (fourcc={fourcc_str})")
            writer_opened = True
            break

    if not writer_opened:
        cap.release()
        raise OSError("Could not open VideoWriter with avc1/mp4v/MJPG")


    # --------------------------------------------------------------------------------
    # Per-video state
    # --------------------------------------------------------------------------------

    # last_seen is the ONLY continuity source (per-ID state)
    # last_seen[id] = {
    #   "xyxy":..., "in_box":..., "label":..., "frames_seen":...,
    #   "frames_since_last_detection_event":..., "last_frame_seen":...
    # }
    last_seen = {}

    # event logs
    inout = []
    inoutframes = []
    inoutid = []
    inout_anglevec = []
    inout_detection_time = []
    inout_angle_degree = []
    inout_vehicle_type = []
    inout_frames_seen = []
    inout_frames_since_last_detection_event = []
    inout_time_seen = []
    inout_time_since_last_detection_event = []
    inout_speed_px_per_frame = []
    inout_speed_px_per_second = []
    inout_center_x = []
    inout_center_y = []


    # fallback id assignment when YOLO does not provide an ID
    my_global_id_counter = -2

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_seconds = frame_idx / fps if fps else 0.0

        # 1) YOLO tracking on this frame
        results = model.track(frame, conf=0.25, persist=True)
        if not results:
            out.write(frame)
            frame_idx += 1
            continue

        frame_res = results[0]
        boxes = frame_res.boxes

        # 2) Gather detections (use YOLO IDs as primary)
        current_dets = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            conf = float(box.conf[0].item()) if box.conf is not None else 0.0

            if box.id is not None:
                model_id = int(box.id[0].item())
            else:
                model_id = -1

            if box.cls is not None:
                cls_idx = int(box.cls[0].item())
                label = model.names[cls_idx]
            else:
                label = "unknown"

            if label not in allowed_labels:
                continue

            # if no model_id, assign a stable fallback id for this detection instance
            # (still not great without tracking, but avoids crashing)
            if model_id == -1:
                my_global_id_counter -= 1
                model_id = my_global_id_counter

            current_dets.append(
                {
                    "xyxy": (x1, y1, x2, y2),
                    "model_id": model_id,
                    "label": label,
                    "conf": conf,
                }
            )

        # 3) Only intervene when there are overlapping boxes in the same frame
        #    and resolve them using last_seen continuity only.
        if len(current_dets) > 1:
            clusters = cluster_overlaps(current_dets, same_frame_overlap_threshold)
            kept = []
            for cluster_idxs in clusters:
                chosen_idx, overwrite_id = choose_det_from_cluster_using_last_seen(
                    cluster_idxs=cluster_idxs,
                    dets=current_dets,
                    last_seen=last_seen,
                    continuity_iou_threshold=continuity_iou_threshold,
                    current_frame_idx=frame_idx,
                    extrapolation_alpha=EXTRAPOLATION_ALPHA,
                )

                det = current_dets[chosen_idx]
                if overwrite_id is not None:
                    det["model_id"] = overwrite_id
                kept.append(det)
            current_dets = kept

        # 4) Counting and last_seen updates operate on last_seen state per ID (last-seen frame)
        for cd in current_dets:
            vid = cd["model_id"]

            in_border_value = iol_xyxy(cd["xyxy"], bbox)
            in_box = not (in_border_value < border_overlap_threshold)

            if vid not in last_seen:
                last_seen[vid] = {
                    "xyxy": cd["xyxy"],
                    "in_box": in_box,
                    "label": cd["label"],
                    "frames_seen": 1,
                    "frames_since_last_detection_event": 0,
                    "last_frame_seen": frame_idx,
                    "vel_per_frame": np.array([0.0, 0.0], dtype=float)
                }
                continue

            prev = last_seen[vid]

            # --- velocity from last-seen box to current box ---
            x1a, y1a, x2a, y2a = prev["xyxy"]
            x1b, y1b, x2b, y2b = cd["xyxy"]

            prev_center = np.array([(x1a + x2a) / 2, (y1a + y2a) / 2], dtype=float)
            curr_center = np.array([(x1b + x2b) / 2, (y1b + y2b) / 2], dtype=float)

            delta = curr_center - prev_center  # (dx, dy) in pixels

            frames_delta = frame_idx - prev["last_frame_seen"]
            if frames_delta <= 0:
                frames_delta = 1
            
            raw_vel = delta / frames_delta  # px/frame
            vel_per_frame = raw_vel # aliasing

            # Optional: clamp raw velocity magnitude (reject spikes)
            if MAX_SPEED_PX_PER_FRAME is not None:
                raw_vel = clamp_vec2(raw_vel, MAX_SPEED_PX_PER_FRAME)

            # EMA smoothing
            old_vel = prev.get("vel_per_frame", np.array([0.0, 0.0], dtype=float))
            prev["vel_per_frame"] = (VEL_EMA_ALPHA * old_vel) + ((1.0 - VEL_EMA_ALPHA) * raw_vel)

            speed_px_per_frame = float(np.linalg.norm(delta)) / frames_delta

            if np.linalg.norm(delta) > 0:
                direction_unit = delta / np.linalg.norm(delta)
            else:
                direction_unit = np.array([0.0, 0.0], dtype=float)

            # store for drawing
            cd["center_xy"] = (int(curr_center[0]), int(curr_center[1]))
            cd["vel_dir_unit"] = direction_unit
            cd["speed_px_per_frame"] = speed_px_per_frame


            # detection-event logic based on last-seen state for this ID
            if prev["in_box"] != in_box:
                inout.append(in_box)
                inoutframes.append(frame_idx)
                inoutid.append(vid)
                inout_detection_time.append(current_seconds)
                x1c, y1c, x2c, y2c = cd["xyxy"]
                cx = (x1c + x2c) / 2.0
                cy = (y1c + y2c) / 2.0
                inout_center_x.append(cx)
                inout_center_y.append(cy)

                x1a, y1a, x2a, y2a = prev["xyxy"]
                x1b, y1b, x2b, y2b = cd["xyxy"]

                prev_center = np.array([(x1a + x2a) / 2, - (y1a + y2a) / 2])
                curr_center = np.array([(x1b + x2b) / 2, - (y1b + y2b) / 2])
                direction = curr_center - prev_center
                frames_delta = frame_idx - prev["last_frame_seen"]
                if frames_delta <= 0:
                    frames_delta = 1  # safety

                dist_px = float(np.linalg.norm(direction))
                speed_ppf = dist_px / frames_delta
                speed_pps = speed_ppf * fps if fps else 0.0

                inout_speed_px_per_frame.append(speed_ppf)
                inout_speed_px_per_second.append(speed_pps)


                if np.linalg.norm(direction) == 0:
                    anglevec = np.array([0.0, 0.0])
                    angle_degree = 0.0
                else:
                    anglevec = direction / np.linalg.norm(direction)
                    angle_degree = float(np.angle(anglevec[0] + anglevec[1] * 1j, deg=True))

                inout_anglevec.append(anglevec)
                inout_angle_degree.append(angle_degree)
                inout_vehicle_type.append(cd["label"])
                inout_frames_seen.append(prev["frames_seen"])
                inout_frames_since_last_detection_event.append(prev["frames_since_last_detection_event"])

                time_seen = (prev["frames_seen"] - 1) / fps if fps else 0.0
                inout_time_seen.append(time_seen)

                time_since_last_event = (prev["frames_since_last_detection_event"] / fps) if fps else 0.0
                inout_time_since_last_detection_event.append(time_since_last_event)

                prev["frames_since_last_detection_event"] = 0

            # update per-ID last_seen state
            prev["xyxy"] = cd["xyxy"]
            prev["in_box"] = in_box
            prev["label"] = cd["label"]
            prev["frames_seen"] += 1
            prev["last_frame_seen"] = frame_idx

        # increment frames_since_last_detection_event counters, flush stale IDs
        for vid, st in list(last_seen.items()):
            st["frames_since_last_detection_event"] += 1

            # optional parity with old "cull_time" concept (seconds-based) - not required for flush
            if fps and (st["frames_since_last_detection_event"] / fps) > cull_time_seconds:
                pass

            if (frame_idx - st["last_frame_seen"]) > FLUSH_FRAMES:
                del last_seen[vid]

        # 5) Draw boxes + border
        cv2.putText(
            frame,
            str(frame_idx),
            (width - 60, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (240, 240, 240),
            2,
        )

        bcolor = (0, 0, 255)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), bcolor, 3)

        for cd in current_dets:
            (x1, y1, x2, y2) = cd["xyxy"]
            label = cd["label"]
            vid = cd["model_id"]

            in_border_value = iol_xyxy(cd["xyxy"], bbox)
            in_box = not (in_border_value < border_overlap_threshold)
            color = (0, 255, 0) if in_box else bcolor

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if DRAW_VELOCITY_ARROW and "vel_dir_unit" in cd and "center_xy" in cd:
                cx, cy = cd["center_xy"]
                dir_u = cd["vel_dir_unit"]
                spf = float(cd.get("speed_px_per_frame", 0.0))

                arrow_len = int(np.clip(spf * ARROW_SCALE, ARROW_MIN_LEN, ARROW_MAX_LEN))
                end_x = int(cx + dir_u[0] * arrow_len)
                end_y = int(cy + dir_u[1] * arrow_len)

                cv2.arrowedLine(
                    frame,
                    (cx, cy),
                    (end_x, end_y),
                    color,
                    2,
                    tipLength=0.3,
                )

            spf = float(cd.get("speed_px_per_frame", 0.0))
            speed_txt = f"{spf:.1f}px/f"

            text = f"ID:{vid} {label} {speed_txt} in_box:{round(in_border_value*100)}%"

            cv2.putText(
                frame,
                text,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        out.write(frame)
        frame_idx += 1

    # --------------------------------------------------------------------------------
    # Build outputs for this video
    # --------------------------------------------------------------------------------

    inout_remapped = ["In" if v else "Out" for v in inout]

    df = pd.DataFrame(
        {
            "Vehicle_ID": inoutid,
            "In_Out": inout_remapped,
            "Filename_ID": output_ID,
            "Frame_Number_0_index": inoutframes,
            "Frames_Seen": inout_frames_seen,
            "Frames_Since_Last_Detection_Event": inout_frames_since_last_detection_event,
            "Time_of_Detection (s)": inout_detection_time,
            "Time_Seen (s)": inout_time_seen,
            "Time_Since_Last_Detection_Event (s)": inout_time_since_last_detection_event,
            "Vehicle_Type": inout_vehicle_type,
            "Angle": inout_anglevec,
            "Angle_Degrees": inout_angle_degree,
            "Speed_Px_Per_Frame": inout_speed_px_per_frame,
            "Speed_Px_Per_Second": inout_speed_px_per_second,
            "Event_Center_X": inout_center_x,
            "Event_Center_Y": inout_center_y,
        }
    )

    # Keep only one entrance/exit for each car (same intent as your original)
    df = df.drop_duplicates(
        subset=["In_Out", "Frames_Seen", "Vehicle_ID", "Filename_ID"],
        keep="first",
    )

    # Lane estimation based on event centers
    k_max_lanes = 15

    df_in = df[df["In_Out"] == "In"]
    df_out = df[df["In_Out"] == "Out"]

    pts_in = df_in[["Event_Center_X", "Event_Center_Y"]].dropna().to_numpy()
    pts_out = df_out[["Event_Center_X", "Event_Center_Y"]].dropna().to_numpy()

    lanes_in_est, lanes_in_k, lanes_in_inertia = estimate_lanes_elbow_curvature(pts_in, k_max=k_max_lanes)
    lanes_out_est, lanes_out_k, lanes_out_inertia = estimate_lanes_elbow_curvature(pts_out, k_max=k_max_lanes)

    df_fileinfo_lookup = pd.DataFrame(
        {
            "Filename_ID": [output_ID],
            "Video_Length (s)": [video_length_s],
            "Video_Total_Frames": [frame_count],
            "Video_Fps": [fps],
            "Filename": [video_path.name],
            "Output_Video": [output_video_path.name],
            "Estimated_Lanes_In": [lanes_in_est],
            "Estimated_Lanes_Out": [lanes_out_est],
            "Estimated_Lanes_Method": ["kmeans_inertia_curvature"],
            "Estimated_Lanes_kmax": [k_max_lanes]
        }
    )

    df_full = pd.merge(df, df_fileinfo_lookup, how="left", on="Filename_ID")
    df_full["Estimated_Lanes"] = np.where(
        df_full["In_Out"].eq("In"),
        df_full["Estimated_Lanes_In"],
        df_full["Estimated_Lanes_Out"],
    )

    def sql_groupby(df_, as_index=False, **kwargs):
        groupby_cols = kwargs.pop("groupby")
        return df_.groupby(groupby_cols, as_index=as_index).agg(**kwargs)

    df_total_in_out = sql_groupby(
        df_full,
        groupby=["In_Out", "Filename_ID", "Video_Length (s)"],
        Count_Cars=("In_Out", "count"),
        Avg_Angle=("Angle_Degrees", "mean"),
        Avg_Speed_Px_Per_Second=("Speed_Px_Per_Second", "mean"),
        Estimated_Lanes=("Estimated_Lanes","max")
    )
    if "Video_Length (s)" in df_total_in_out.columns and (df_total_in_out["Video_Length (s)"] != 0).all():
        df_total_in_out["Cars_per_Second"] = df_total_in_out["Count_Cars"] / df_total_in_out["Video_Length (s)"]
        lanes = df_total_in_out["Estimated_Lanes"].replace(0, np.nan)
        df_total_in_out["Cars_per_Lane_per_Second"] = (
            df_total_in_out["Count_Cars"] / (df_total_in_out["Video_Length (s)"] * lanes)
        )
    else:
        df_total_in_out["Cars_per_Second"] = np.nan
        df_total_in_out["Cars_per_Lane_per_Second"] = np.nan

    # Write CSVs
    df_full.to_csv(output_total_csv_path, index=False)
    df.to_csv(output_fact_csv_path, index=False)
    df_fileinfo_lookup.to_csv(output_dim_csv_path, index=False)
    df_total_in_out.to_csv(output_agg_csv_path, index=False)

    cap.release()
    out.release()
    print(f"Output size bytes: {output_video_path.stat().st_size}")

    # Try to reset Ultralytics tracker state between videos (best-effort)
    try:
        if hasattr(model, "predictor") and hasattr(model.predictor, "tracker") and model.predictor.tracker is not None:
            model.predictor.tracker.reset()
    except Exception:
        pass

    print(f"Done. Output csvs to {OUTPUT_DATA_DIR}")

    print(f"Done. Output video: {output_video_path}")

    print(df_full)
    print(df)
    print(df_fileinfo_lookup)
    print(df_total_in_out)
    all_df_full.append(df_full.copy())
    all_df_fact.append(df.copy())
    all_df_dim.append(df_fileinfo_lookup.copy())
    all_df_agg.append(df_total_in_out.reset_index().copy()) 

OUTPUT_MASTER_DIR = OUTPUT_DATA_DIR / "all_videos"
OUTPUT_MASTER_DIR.mkdir(parents=True, exist_ok=True)

df_full_all = pd.concat(all_df_full, ignore_index=True) if all_df_full else pd.DataFrame()
df_fact_all = pd.concat(all_df_fact, ignore_index=True) if all_df_fact else pd.DataFrame()
df_dim_all  = pd.concat(all_df_dim,  ignore_index=True) if all_df_dim  else pd.DataFrame()
df_agg_all  = pd.concat(all_df_agg,  ignore_index=True) if all_df_agg  else pd.DataFrame()

df_full_all.to_csv(OUTPUT_MASTER_DIR / "Cars_detected.csv", index=False)
df_fact_all.to_csv(OUTPUT_MASTER_DIR / "Cars_detected_fact.csv", index=False)
df_dim_all.to_csv(OUTPUT_MASTER_DIR / "Cars_detected_dim.csv", index=False)
df_agg_all.to_csv(OUTPUT_MASTER_DIR / "Cars_detected_agg.csv", index=False)