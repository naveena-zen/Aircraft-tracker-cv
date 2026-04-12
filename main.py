# encoding: utf-8
"""
main.py
-------
Entry point for the Shi-Tomasi + Lucas-Kanade Optical Flow tracker.

Pipeline (per video)
--------------------
1. Detect corner points on the FIRST frame using Shi-Tomasi
   (goodFeaturesToTrack) and save it as  output_corners.jpg.
2. Track those corners across all remaining frames with Lucas-Kanade
   optical flow (calcOpticalFlowPyrLK).
3. Save exactly 2 unique mid-point snapshots at ~1/3 and ~2/3 of the
   video duration, plus the final frame.

Total output per video: 4 images
  <stem>_output_corners.jpg   -- frame 0 with detected corners
  <stem>_output_frame_mid1.jpg -- ~33% through the video
  <stem>_output_frame_mid2.jpg -- ~67% through the video
  <stem>_output_final.jpg      -- last frame

No live video window is opened (headless / batch mode).
"""

import os
import sys
import glob
import cv2
import numpy as np

from analytics import detect_corners, detect_turbulence, draw_turbulence_warning
from tracker import track_points, draw_tracked_points
import analytics as chart_generator
import analytics as heatmap_generator

# Force UTF-8 output so any Unicode in filenames prints cleanly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VIDEOS_DIR       = "videos"        # folder containing input video files
OUTPUT_DIR       = "output"        # folder where result images will be saved
CORNER_COLOR     = (0, 255, 0)     # green  -- detected corners on frame 0
TRACK_COLOR      = (0, 255, 255)   # yellow -- tracked points on later frames
CIRCLE_RADIUS    = 5
CIRCLE_THICKNESS = 2


# ---------------------------------------------------------------------------
# Process a single video
# ---------------------------------------------------------------------------

def process_video(video_path, output_dir):
    """
    Run corner detection + optical flow tracking on one video file.
    Saves exactly 4 unique output images per video.

    Parameters
    ----------
    video_path : str -- path to the input video (.mp4 / .avi)
    output_dir : str -- directory where output images are saved
    """
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    print("\n" + "=" * 60)
    print("Processing: " + video_path)
    print("=" * 60)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("  [ERROR] Cannot open video: " + video_path)
        return

    # ------------------------------------------------------------------
    # Get total frame count so we can compute unique save points
    # ------------------------------------------------------------------
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Fallback: count manually (some codecs don't report frame count)
        total_frames = None

    # ------------------------------------------------------------------
    # Step 1 : Read the first frame
    # ------------------------------------------------------------------
    ret, first_frame = cap.read()
    if not ret:
        print("  [ERROR] Cannot read first frame.")
        cap.release()
        return

    # ------------------------------------------------------------------
    # Step 2 : Detect corners on the first frame using Shi-Tomasi
    # ------------------------------------------------------------------
    corners = detect_corners(first_frame)

    if corners is None or len(corners) == 0:
        print("  [WARNING] No corners detected in first frame. Skipping video.")
        cap.release()
        return

    print("  Detected " + str(len(corners)) + " corner points on frame 0.")

    # Calculate corner scores for chart 3
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    eigen = cv2.cornerMinEigenVal(first_frame_gray, 7)
    corner_scores = []
    for pt in corners:
        x, y = int(pt[0][0]), int(pt[0][1])
        corner_scores.append(eigen[y, x])

    # Draw and save first frame with detected corners
    first_annotated = first_frame.copy()
    for pt in corners:
        x, y = int(pt[0][0]), int(pt[0][1])
        cv2.circle(first_annotated, (x, y),
                   CIRCLE_RADIUS, CORNER_COLOR, CIRCLE_THICKNESS)

    corners_path = os.path.join(output_dir, video_stem + "_output_corners.jpg")
    cv2.imwrite(corners_path, first_annotated)
    print("  [1/4] Saved corners image   -> " + corners_path)

    # ------------------------------------------------------------------
    # Step 3 : Determine save points (1/3 and 2/3 through the video)
    # ------------------------------------------------------------------
    if total_frames and total_frames > 3:
        mid1_frame = total_frames // 3        # ~33 %
        mid2_frame = (2 * total_frames) // 3  # ~67 %
    else:
        # Fallback for very short videos or unknown length
        mid1_frame = 30
        mid2_frame = 60

    print("  Save milestones: mid1=frame " + str(mid1_frame) +
          "  mid2=frame " + str(mid2_frame) +
          "  final=last frame")

    # ------------------------------------------------------------------
    # Step 4 : Optical flow tracking loop
    # ------------------------------------------------------------------
    old_gray  = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    track_pts = corners.copy()   # shape (N, 1, 2) float32

    # New code for charts
    orig_pts = corners.copy().reshape(-1, 2)
    tracking_counts = [(0, len(orig_pts))]
    displacements = [(0, 0.0)]

    # Accumulate every tracked (x, y) position for the motion heatmap
    all_tracked_positions = []
    # Seed with the initial corner positions from frame 0
    for pt in corners:
        all_tracked_positions.append((float(pt[0][0]), float(pt[0][1])))

    # Speed estimation: per-frame average pixel movement
    per_frame_speeds = []

    frame_idx    = 1
    final_frame  = None
    mid1_saved   = False
    mid2_saved   = False
    mid1_annotated   = None
    mid2_annotated   = None
    mid1_snapshot_idx = 0
    mid2_snapshot_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Track corners from previous frame to current frame
        good_new, good_old = track_points(old_gray, frame_gray, track_pts)

        # New code for charts
        if len(good_new) > 0:
            track_pts_flat = track_pts.reshape(-1, 2)
            survivor_indices = []
            old_idx = 0
            for go in good_old:
                while old_idx < len(track_pts_flat) and not np.array_equal(track_pts_flat[old_idx], go):
                    old_idx += 1
                if old_idx < len(track_pts_flat):
                    survivor_indices.append(old_idx)
                    old_idx += 1
            
            orig_pts = orig_pts[survivor_indices]
            dists = np.linalg.norm(good_new - orig_pts, axis=1)
            avg_disp = float(np.mean(dists))
            
            tracking_counts.append((frame_idx, len(good_new)))
            displacements.append((frame_idx, avg_disp))

            # Speed: pixel distance each point moved from PREVIOUS frame
            prev_pts_flat = track_pts.reshape(-1, 2)  # positions in previous frame
            # good_old contains the matched subset of prev_pts_flat
            frame_dists = np.linalg.norm(good_new - good_old, axis=1)
            per_frame_speeds.append(float(np.mean(frame_dists)))
        else:
            tracking_counts.append((frame_idx, 0))
            displacements.append((frame_idx, 0.0))

        if len(good_new) == 0:
            print("  [WARNING] All points lost at frame " + str(frame_idx) +
                  ". Stopping tracking.")
            break

        # Draw tracked points on the current frame
        annotated = frame.copy()
        annotated = draw_tracked_points(
            annotated, good_new,
            color=TRACK_COLOR,
            radius=CIRCLE_RADIUS,
            thickness=CIRCLE_THICKNESS
        )

        # Save mid1 snapshot (~1/3 through)
        if not mid1_saved and frame_idx >= mid1_frame:
            mid1_snapshot_idx = frame_idx          # remember for turbulence check
            mid1_annotated    = annotated.copy()   # keep a copy; overlay applied later
            mid1_saved = True

        # Save mid2 snapshot (~2/3 through)
        if not mid2_saved and frame_idx >= mid2_frame:
            mid2_snapshot_idx = frame_idx
            mid2_annotated    = annotated.copy()
            mid2_saved = True

        # Accumulate tracked positions for heatmap (non-intrusive)
        for pt in good_new:
            all_tracked_positions.append((float(pt[0]), float(pt[1])))

        # Update state for next iteration
        old_gray    = frame_gray.copy()
        track_pts   = good_new.reshape(-1, 1, 2)
        final_frame = annotated
        frame_idx  += 1

    # ------------------------------------------------------------------
    # Step 5 : Detect turbulence from per-frame speed data
    # ------------------------------------------------------------------
    turbulent_indices, turb_stats = detect_turbulence(per_frame_speeds)
    if turb_stats["count"] > 0:
        print("  [Turbulence] Detected " + str(turb_stats["count"]) +
              " turbulent frame(s) | threshold: " +
              "{:.2f}".format(turb_stats["threshold"]) + " px/frame")
    else:
        print("  [Turbulence] No turbulence detected in this video.")

    # ------------------------------------------------------------------
    # Step 6 : Save snapshot frames (with turbulence overlay if needed)
    # ------------------------------------------------------------------
    if mid1_saved and mid1_annotated is not None:
        if mid1_snapshot_idx in turbulent_indices:
            draw_turbulence_warning(mid1_annotated)
        path = os.path.join(output_dir, video_stem + "_output_frame_mid1.jpg")
        cv2.imwrite(path, mid1_annotated)
        print("  [2/4] Saved mid-1 frame " + str(mid1_snapshot_idx) + " -> " + path)

    if mid2_saved and mid2_annotated is not None:
        if mid2_snapshot_idx in turbulent_indices:
            draw_turbulence_warning(mid2_annotated)
        path = os.path.join(output_dir, video_stem + "_output_frame_mid2.jpg")
        cv2.imwrite(path, mid2_annotated)
        print("  [3/4] Saved mid-2 frame " + str(mid2_snapshot_idx) + " -> " + path)

    if final_frame is not None:
        if (frame_idx - 1) in turbulent_indices:
            draw_turbulence_warning(final_frame)
        final_path = os.path.join(output_dir,
                                  video_stem + "_output_final.jpg")
        cv2.imwrite(final_path, final_frame)
        print("  [4/4] Saved final frame " + str(frame_idx - 1) +
              " -> " + final_path)
    else:
        print("  [WARNING] No frames were processed after frame 0.")

    # ------------------------------------------------------------------
    # Step 7 : Generate and save charts
    # ------------------------------------------------------------------
    print("  Generating statistical charts...")
    chart_generator.create_charts(video_stem, output_dir, tracking_counts, displacements, corner_scores)

    # ------------------------------------------------------------------
    # Step 8 : Generate and save motion heatmap  (with speed stats)
    # ------------------------------------------------------------------
    print("  Generating motion heatmap...")
    if per_frame_speeds:
        speed_avg = float(np.mean(per_frame_speeds))
        speed_max = float(np.max(per_frame_speeds))
        speed_min = float(np.min(per_frame_speeds))
    else:
        speed_avg = speed_max = speed_min = 0.0
    heatmap_generator.generate_heatmap(
        video_stem, first_frame, all_tracked_positions, output_dir,
        speed_avg=speed_avg, speed_max=speed_max, speed_min=speed_min
    )

    cap.release()
    print("  Done. Total frames: " + str(frame_idx) +
          "  |  Output images: 4 + 3 charts + 1 heatmap")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Gather all video files from the videos/ folder
    patterns = [
        os.path.join(VIDEOS_DIR, "*.mp4"),
        os.path.join(VIDEOS_DIR, "*.avi"),
    ]
    video_files = []
    for p in patterns:
        video_files.extend(glob.glob(p))

    if not video_files:
        print("[ERROR] No video files found in '" + VIDEOS_DIR + "/' folder.")
        return

    print("Found " + str(len(video_files)) + " video file(s) to process.")

    for video_path in sorted(video_files):
        process_video(video_path, OUTPUT_DIR)

    total_imgs = len(glob.glob(os.path.join(OUTPUT_DIR, "*.jpg")))
    print("\nAll videos processed successfully.")
    print("Total output images: " + str(total_imgs) +
          "  (4 per video -- corners, mid1, mid2, final)")
    print("Saved to: " + os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
