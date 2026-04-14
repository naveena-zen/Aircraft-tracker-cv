# encoding: utf-8
"""
analytics.py
------------
Unified analytics module for the Optical Flow Based Aircraft Feature
Tracking System.

This file consolidates the following original modules (unchanged logic):
  - corners_detector.py  → detect_corners()
  - chart_generator.py   → create_charts()
  - heatmap_generator.py → generate_heatmap(), _jet_cmap()
  - scratch.py           → survivor-index test utility (kept as reference)
"""

# ---------------------------------------------------------------------------
# Standard / third-party imports (union of all original module imports)
# ---------------------------------------------------------------------------
import os

import cv2
import numpy as np

import matplotlib
# Use 'Agg' backend to avoid requiring an X server or opening GUI windows
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors   # noqa: F401  (kept from heatmap_generator)


# ===========================================================================
# Section 1 — Corner Detection
# (originally: corners_detector.py)
# ===========================================================================

def detect_corners(frame_bgr: np.ndarray,
                   max_corners: int = 200,
                   quality_level: float = 0.3,
                   min_distance: float = 7.0,
                   block_size: int = 7) -> np.ndarray:
    """
    Detect strong corner points in a colour frame using the Shi-Tomasi
    algorithm (cv2.goodFeaturesToTrack).

    Parameters
    ----------
    frame_bgr   : Input colour (BGR) frame.
    max_corners : Maximum number of corners to return.
    quality_level : Minimal accepted quality of image corners (0–1).
    min_distance  : Minimum possible Euclidean distance between corners.
    block_size    : Size of an average block for computing derivative
                    covariation matrix.

    Returns
    -------
    corners : numpy array of shape (N, 1, 2), dtype float32.
              Ready to pass directly to calcOpticalFlowPyrLK.
              Returns None if no corners are found.
    """
    # Convert to greyscale – goodFeaturesToTrack requires a single-channel image
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Shi-Tomasi corner detection
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size
    )

    return corners  # shape (N, 1, 2) float32, or None


# ===========================================================================
# Section 2 — Statistical Charts
# (originally: chart_generator.py)
# ===========================================================================

def create_charts(video_stem, output_dir, tracking_counts, displacements, corner_scores):
    """
    Generate and save statistical charts for the tracking pipeline.
    """
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # ensure no interactive plot shows up
    plt.ioff()

    # -----------------------------------------------------------------
    # Chart 1: Tracking Performance  array format: [(frame_idx, count)]
    # -----------------------------------------------------------------
    if tracking_counts:
        frames = [item[0] for item in tracking_counts]
        counts = [item[1] for item in tracking_counts]
        initial_count = counts[0] if counts else 0

        plt.figure(figsize=(10, 6))
        plt.plot(frames, counts, marker='o', linestyle='-', color='b', label='Tracked Points')
        plt.axhline(y=initial_count, color='r', linestyle='--', label=f'Initial Count ({initial_count})')
        plt.title('Tracked Points Count Over Frames')
        plt.xlabel('Frame Number')
        plt.ylabel('Number of Successfully Tracked Points')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f"{video_stem}_tracking_performance.png"), dpi=150)
        plt.close()

    # -----------------------------------------------------------------
    # Chart 2: Point Displacement Over Frames
    # -----------------------------------------------------------------
    if displacements:
        frames = [item[0] for item in displacements]
        avg_disp = [item[1] for item in displacements]

        plt.figure(figsize=(10, 6))
        plt.plot(frames, avg_disp, linestyle='-', color='m', label='Average Displacement')
        plt.fill_between(frames, avg_disp, color='m', alpha=0.2)
        plt.title('Average Point Displacement Over Frames')
        plt.xlabel('Frame Number')
        plt.ylabel('Average Displacement Distance (pixels)')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f"{video_stem}_displacement.png"), dpi=150)
        plt.close()

    # -----------------------------------------------------------------
    # Chart 3: Corner Detection Confidence Distribution
    # -----------------------------------------------------------------
    if corner_scores is not None and len(corner_scores) > 0:
        plt.figure(figsize=(10, 6))
        n, bins, patches = plt.hist(corner_scores, bins=20, edgecolor='black', alpha=0.7)

        # Apply different colours for each bar
        cmap = plt.get_cmap('viridis')
        for i, p in enumerate(patches):
            p.set_facecolor(cmap(i / float(max(1, len(patches) - 1))))

        plt.title('Corner Detection Confidence Distribution')
        plt.xlabel('Corner Quality Score')
        plt.ylabel('Number of Corners')
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f"{video_stem}_corner_distribution.png"), dpi=150)
        plt.close()


# ===========================================================================
# Section 3 — Motion Heatmap
# (originally: heatmap_generator.py)
# ===========================================================================

def generate_heatmap(video_stem, first_frame, all_tracked_points, output_dir,
                     speed_avg=0.0, speed_max=0.0, speed_min=0.0):
    """
    Build and save a motion heatmap for one video.

    Parameters
    ----------
    video_stem        : str   — base name of the video (no extension)
    first_frame       : ndarray (H, W, 3) BGR — first video frame
    all_tracked_points: list of (x, y) float tuples — every tracked position
                        accumulated across all frames
    output_dir        : str   — root output directory (e.g. "output")
    speed_avg         : float — average pixel-per-frame speed across all frames
    speed_max         : float — maximum per-frame average pixel speed recorded
    speed_min         : float — minimum per-frame average pixel speed recorded
    """

    # ------------------------------------------------------------------
    # 0. Guard: nothing to draw
    # ------------------------------------------------------------------
    if not all_tracked_points:
        print("  [Heatmap] No tracked points to plot — skipping.")
        return

    h, w = first_frame.shape[:2]

    # ------------------------------------------------------------------
    # 1. Accumulate density on a float accumulation map
    # ------------------------------------------------------------------
    density = np.zeros((h, w), dtype=np.float32)

    for (x, y) in all_tracked_points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            density[yi, xi] += 1.0

    # ------------------------------------------------------------------
    # 2. Gaussian blur to spread / smooth the density
    # ------------------------------------------------------------------
    density = cv2.GaussianBlur(density, (0, 0), sigmaX=15, sigmaY=15)

    # ------------------------------------------------------------------
    # 3. Normalise to [0, 255] uint8
    # ------------------------------------------------------------------
    max_val = density.max()
    if max_val > 0:
        density_norm = (density / max_val * 255).astype(np.uint8)
    else:
        density_norm = density.astype(np.uint8)

    # ------------------------------------------------------------------
    # 4. Apply COLORMAP_JET  (blue = low, red = high)
    # ------------------------------------------------------------------
    heatmap_bgr = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)

    # ------------------------------------------------------------------
    # 5. Blend: 60% heatmap + 40% original first frame
    # ------------------------------------------------------------------
    blended_bgr = cv2.addWeighted(heatmap_bgr, 0.60, first_frame, 0.40, 0)

    # ------------------------------------------------------------------
    # 5b. Overlay speed stats text in the bottom-left corner (OpenCV)
    # ------------------------------------------------------------------
    lines = [
        f"Avg Speed : {speed_avg:.2f} px/frame",
        f"Max Speed : {speed_max:.2f} px/frame",
        f"Min Speed : {speed_min:.2f} px/frame",
    ]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(h, w) / 1000.0)   # scales with image size
    thickness  = 1
    padding    = 8
    line_gap   = 6

    # Measure text to size the background box
    text_sizes = [cv2.getTextSize(ln, font, font_scale, thickness)[0] for ln in lines]
    box_w = max(tw for tw, _ in text_sizes) + padding * 2
    line_h = max(th for _, th in text_sizes)
    box_h = len(lines) * (line_h + line_gap) + padding * 2 - line_gap

    margin = 10
    x0 = margin
    y0 = h - margin - box_h

    # Dark semi-transparent background box
    overlay = blended_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.65, blended_bgr, 0.35, 0, blended_bgr)

    # White text lines
    text_x = x0 + padding
    text_y = y0 + padding + line_h
    for ln in lines:
        cv2.putText(blended_bgr, ln, (text_x, text_y),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        text_y += line_h + line_gap

    # Final conversion to RGB for matplotlib (after overlay)
    blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # 6. Compose figure with heatmap and colorbar legend
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 6),
        gridspec_kw={"width_ratios": [20, 1]},
        facecolor="#0d0d0d"
    )

    ax_img, ax_cb = axes

    # Main heatmap image
    ax_img.imshow(blended_rgb)
    ax_img.axis("off")

    # Title
    display_name = video_stem.replace("_", " ").title()
    ax_img.set_title(
        f"Motion Heatmap \u2014 {display_name}",
        color="white",
        fontsize=13,
        fontweight="bold",
        pad=10
    )

    # ------------------------------------------------------------------
    # 7. Colorbar legend strip (low → high activity)
    # ------------------------------------------------------------------
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    ax_cb.imshow(
        gradient,
        aspect="auto",
        cmap=_jet_cmap(),
        origin="lower"
    )
    ax_cb.set_xticks([])
    ax_cb.set_yticks([0, 255])
    ax_cb.set_yticklabels(["Low", "High"], color="white", fontsize=9)
    ax_cb.tick_params(colors="white", length=0)
    for spine in ax_cb.spines.values():
        spine.set_edgecolor("#555555")

    ax_cb.set_ylabel("Activity", color="white", fontsize=9, labelpad=6)
    ax_cb.yaxis.set_label_position("right")
    ax_cb.yaxis.tick_right()

    fig.patch.set_facecolor("#0d0d0d")
    plt.tight_layout(pad=0.5)

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    heatmap_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    save_path = os.path.join(heatmap_dir, video_stem + "_heatmap.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"  [Heatmap] Saved -> {save_path}")


def _jet_cmap():
    """Return a matplotlib colormap that matches cv2.COLORMAP_JET ordering."""
    return plt.cm.jet


# ===========================================================================
# Section 4 — Turbulence Detection
# ===========================================================================

def detect_turbulence(per_frame_speeds):
    """
    Detect turbulent frames from per-frame speed data.

    A frame is flagged as turbulent when its average pixel displacement
    exceeds  mean + 2 × standard_deviation  of all recorded per-frame speeds.
    This statistical threshold catches sudden abnormal spikes in motion while
    ignoring normal variation.

    Parameters
    ----------
    per_frame_speeds : list of float
        Average pixel displacement recorded for every tracked frame
        (as produced inside the optical-flow loop in main.py).
        Index 0 corresponds to frame 1 (the first frame after the corners
        frame), index k corresponds to frame k+1.

    Returns
    -------
    turbulent_frame_indices : set of int
        Frame indices (1-based, matching frame_idx in main.py) that are
        flagged as turbulent.  Empty set if fewer than 2 frames are
        available or if no spikes are detected.
    stats : dict
        Diagnostic information:
          'mean'   – mean speed across all frames
          'std'    – standard deviation of speeds
          'threshold' – the spike threshold (mean + 2*std)
          'count'  – number of turbulent frames detected
    """
    turbulent_frame_indices = set()
    stats = {"mean": 0.0, "std": 0.0, "threshold": 0.0, "count": 0}

    if len(per_frame_speeds) < 2:
        return turbulent_frame_indices, stats

    speeds = np.array(per_frame_speeds, dtype=np.float32)
    mean_speed = float(np.mean(speeds))
    std_speed  = float(np.std(speeds))
    threshold  = mean_speed + 2.0 * std_speed

    stats["mean"]      = mean_speed
    stats["std"]       = std_speed
    stats["threshold"] = threshold

    # per_frame_speeds[0] was collected at frame_idx == 1
    for i, spd in enumerate(per_frame_speeds):
        if spd > threshold:
            turbulent_frame_indices.add(i + 1)   # 1-based frame index

    stats["count"] = len(turbulent_frame_indices)
    return turbulent_frame_indices, stats


def draw_turbulence_warning(frame_bgr):
    """
    Overlay a red "⚠ Turbulence Detected" warning in the top-right corner
    of *frame_bgr* (modified **in-place**).

    Parameters
    ----------
    frame_bgr : np.ndarray  (H, W, 3) — BGR image to annotate.

    Returns
    -------
    frame_bgr : the same array, now annotated.
    """
    h, w = frame_bgr.shape[:2]

    label      = "\u26a0 Turbulence Detected"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(h, w) / 900.0)
    thickness  = max(1, int(font_scale * 2))
    padding    = 10
    margin     = 12

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Position: top-right corner
    box_w = tw + padding * 2
    box_h = th + baseline + padding * 2
    x0 = w - margin - box_w
    y0 = margin

    # Dark semi-transparent background box
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h),
                  (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.65, frame_bgr, 0.35, 0, frame_bgr)

    # Red warning text
    text_x = x0 + padding
    text_y = y0 + padding + th
    cv2.putText(frame_bgr, label, (text_x, text_y),
                font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

    return frame_bgr


# ===========================================================================
# Section 5 — Scratch / Test Utility
# (originally: scratch.py)
# Kept as a runnable utility; produces output only when this file is run
# directly (python analytics.py).
# ===========================================================================

def _run_survivor_test():
    """
    Test np.array_equal-based survivor-index logic used inside main.py.
    Prints 'survivors: [0, 2]' when working correctly.
    """
    points = np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]], dtype=np.float32)
    status = np.array([1, 0, 1], dtype=np.uint8)

    good_old = points[status == 1]

    survivors = []
    old_idx = 0
    for go in good_old:
        while old_idx < len(points) and not np.array_equal(points[old_idx], go):
            old_idx += 1
        if old_idx < len(points):
            survivors.append(old_idx)
            old_idx += 1

    print("survivors:", survivors)


if __name__ == "__main__":
    _run_survivor_test()
