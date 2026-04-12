# Optical Flow Based Aircraft Feature Tracking System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.x-orange?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-red?logoColor=white)
![Domain](https://img.shields.io/badge/Domain-Computer%20Vision-purple)

---

## Table of Contents

- [Project Description](#project-description)
- [Domain](#domain)
- [Algorithms Used](#algorithms-used)
- [What the Project Detects](#what-the-project-detects)
- [Output Images](#output-images)
- [Results](#results)
- [Enhancements](#enhancements)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)

---

## Project Description

This project demonstrates two fundamental Computer Vision techniques — **Shi-Tomasi Corner Detection** and **Lucas-Kanade Optical Flow Tracking** — applied to real aerial (aircraft) video footage.

The system reads a video file, identifies the most significant corner/feature points in the very first frame using the Shi-Tomasi algorithm, and then continuously tracks those points across all subsequent frames using the Lucas-Kanade Optical Flow method. Instead of displaying a live video window, the results are saved as annotated images at key points in the video — giving a clear visual record of what was detected and how it was tracked over time. Additionally, the system generates analytical charts (tracking performance, displacement averages, and confidence distributions) using Matplotlib to offer deeper statistical insight into the tracking pipeline. The project processes multiple videos in batch mode and requires no deep learning models or pre-trained weights — it relies entirely on classical, interpretable Computer Vision algorithms.

---

## Domain

**Image Analytics and Computer Vision**

This project sits at the intersection of two closely related areas:

| Domain               | What this project does in that domain                          |
|----------------------|----------------------------------------------------------------|
| **Image Analytics**  | Extracts meaningful spatial features (corners) from image frames and analyses how they change over time |
| **Computer Vision**  | Applies classical CV algorithms to understand motion and structure in video sequences of real-world scenes |

---

## Algorithms Used

### 1. Shi-Tomasi Corner Detection (`cv2.goodFeaturesToTrack`)

**What it does:**
Shi-Tomasi Corner Detection is a feature detection algorithm that identifies the strongest "corner" points in an image — locations where intensity changes significantly in multiple directions. These points are considered the most trackable and distinguishable regions in a frame.

It is an improvement over the earlier Harris Corner Detector, using a more reliable scoring function to select only the best-quality corners up to a user-defined maximum.

**How it works in this project:**
- Applied only on **Frame 0** (the very first frame of each video)
- The frame is converted to **greyscale** before detection
- Parameters used:
  - `maxCorners = 200` — detect up to 200 strong corner points
  - `qualityLevel = 0.3` — only accept corners with at least 30% of the best corner's quality score
  - `minDistance = 7.0` — enforce minimum spacing between corners to avoid clustering
  - `blockSize = 7` — neighbourhood size used to compute the corner response
- The detected corners are returned as an array of `(x, y)` coordinates
- These coordinates are then used as the **starting points** for optical flow tracking

**Visual result:** Green circles drawn on the first frame mark all detected corner points.

---

### 2. Lucas-Kanade Optical Flow (`cv2.calcOpticalFlowPyrLK`)

**What it does:**
Lucas-Kanade Optical Flow is a sparse motion estimation algorithm. Given a set of points in one frame, it tries to find where those same points have moved to in the next frame by assuming that the appearance of each point's local neighbourhood stays roughly constant between frames (the "brightness constancy" assumption).

The **pyramidal** version (`PyrLK`) extends this to handle larger motions by analysing the video at multiple image scales (a Gaussian pyramid).

**How it works in this project:**
- Applied on **every frame after Frame 0**, using the corners detected by Shi-Tomasi as starting points
- Parameters used:
  - `winSize = (70, 70)` — large search window to handle fast-moving objects like aircraft
  - `maxLevel = 2` — two pyramid levels (original + two downscaled versions) for multi-scale search
  - `criteria` — stops iterating after 10 iterations or when change drops below 0.03
- For each frame, only **successfully tracked points** (where `status == 1`) are kept
- The tracked points are updated frame-by-frame and passed forward to the next frame
- **Yellow circles** are drawn on each frame at each successfully tracked point position

---

## What the Project Detects

### Corner / Feature Points
In the context of aerial videos, corner points correspond to **visually distinctive locations** on the aircraft and background — such as:
- Wing edges and tips
- Engine nacelle outlines
- Fuselage corners and windows
- Structural joints or colour boundaries on the aircraft body
- Horizon lines and cloud edges in the background

These are regions where the image intensity changes sharply in two directions — making them reliable and stable for tracking.

### Tracking Results
As the video progresses:
- The originally detected corner points are tracked across frames
- The algorithm follows the motion of each tracked point, even as the aircraft moves, rotates, or the camera angle changes
- Points that can no longer be reliably matched between frames are dropped (status = 0)
- The remaining points continue to be tracked and displayed

### What the output images represent
| Output Image | What it shows |
|---|---|
| `output_corners.jpg` | The first frame of the video with **green circles** marking all Shi-Tomasi detected corner points — the starting seeds for tracking |
| `output_frame_mid1.jpg` | A frame at approximately 1/3 through the video, showing **yellow circles** at the current positions of all successfully tracked points |
| `output_frame_mid2.jpg` | A frame at approximately 2/3 through the video, showing the continued tracking of surviving points |
| `output_final.jpg` | The last frame of the video, showing **yellow circles** at the final tracked positions — some original points may have been lost by this stage |

### Analytical Charts
Alongside video frame annotations, three statistical charts are generated per video inside the `output/charts/` directory:
- **Tracking Performance Over Frames:** A line graph demonstrating how tracking degrades over time as feature points are lost.
- **Average Point Displacement:** A line graph depicting the average distance tracked points have moved from their original detected position.
- **Corner Detection Confidence Distribution:** A histogram representing the distribution of strengths for the originally detected corners.

In addition, the **Motion Heatmap** (saved to `output/heatmaps/`) includes a speed statistics overlay in the bottom-left corner showing **Avg, Max, and Min speed** in px/frame across all tracked frames.

---

## Results

### What was successfully detected
- Shi-Tomasi reliably detected between **5 and 38 corner points** per video on the first frame, depending on the scene complexity
- The detected corners consistently fell on visually distinctive regions — edges of the aircraft body, structural joints, and high-contrast boundaries

### What tracking showed
- Lucas-Kanade Optical Flow successfully tracked the detected corners across hundreds of frames per video
- Yellow circles in mid-frame and final-frame images visually confirm that the points followed real motion in the scene
- The circles shift position between frames — demonstrating that the algorithm is detecting and following actual movement, not static markers

### How the rings represent tracked motion
Each **circle drawn on the frame represents a single feature point** being actively tracked by the optical flow algorithm:
- **Green circles** (Frame 0): These are the original Shi-Tomasi detections — the seed points
- **Yellow circles** (subsequent frames): These show where each seed point has moved to by that frame, as computed by Lucas-Kanade
- Circles that disappear in later frames correspond to points the algorithm could no longer track reliably (e.g., the feature moved out of frame or became occluded)

The spatial shift of circle positions between the corners image and the final image visually demonstrates the **motion of the aircraft** (or camera) over the video duration.

---

## Enhancements

The following capabilities have been progressively added to the core tracking pipeline:

### 1. Motion Heatmap
- Generates **1 heatmap image per video**, saved in `output/heatmaps/`
- Accumulates every tracked `(x, y)` position across all frames into a 2D density map
- Brighter / warmer regions represent areas of **highest motion activity**
- Applies `cv2.COLORMAP_JET` (blue = low, red = high) for intuitive visualisation
- Blends the coloured density map with the **first frame at 60/40 opacity** so the aircraft context remains visible
- Includes a colour-bar legend strip (Low → High activity) and a descriptive title

### 2. Aircraft Speed Estimation
- Calculates **average pixel displacement per frame** for all successfully tracked points
- Computes **Avg, Max, and Min** speed (in px/frame) across the entire video
- Results are overlaid as **white text on a semi-transparent dark box** in the **bottom-left corner** of the heatmap image
- Provides a quantitative measure of how fast the tracked features are moving between consecutive frames

### 3. Turbulence Detection

#### What it does
Flags individual frames where the tracked feature movement spikes abnormally — a proxy for turbulent conditions in the captured footage.

#### How it is calculated
- Reuses the **per-frame average pixel displacement** already computed during speed estimation (no extra data collection)
- Computes the **mean** and **standard deviation** of all per-frame displacements
- Any frame whose displacement exceeds **mean + 2 × standard deviation** is classified as turbulent
- This is a classic statistical outlier test — only frames with genuinely anomalous motion are flagged

#### What the warning overlay shows
- On every snapshot image (`_mid1`, `_mid2`, `_final`) that corresponds to a turbulent frame:
  - A **red `⚠ Turbulence Detected` text label** is drawn in the **top-right corner**
  - The label sits on a **dark semi-transparent background box** for readability
- Frames with no detected turbulence receive **no extra overlay** — existing output is unchanged
- No new output files are created; only the standard 4 images per video are affected

#### Real-world relevance to aviation analytics
In aviation, turbulence causes sudden, unexpected accelerations of the airframe.  
When analysing aerial video, turbulent episodes appear as sharp spikes in optical flow magnitude — exactly what the mean + 2σ threshold captures.  
Automatically tagging these frames allows analysts to:
- Quickly identify high-stress moments in a flight recording without scrubbing through hours of footage
- Correlate visual turbulence events with flight data recorder outputs
- Apply targeted stabilisation or further analysis only to flagged segments

### 4. File Structure Cleanup
- Merged `chart_generator.py`, `heatmap_generator.py`, `corners_detector.py`, and `scratch.py` into a single **`analytics.py`** file
- All public functions (`detect_corners`, `create_charts`, `generate_heatmap`, `detect_turbulence`, `draw_turbulence_warning`) are importable from one place
- Reduces the number of top-level files and makes the codebase easier to navigate and maintain
- All existing functionality, output filenames, and results are fully preserved

---

## Project Structure

```
aircraft-tracker/
│
├── main.py               # Entry point — opens videos, runs the full pipeline,
│                         #   saves annotated output images
│
├── analytics.py          # Unified analytics module:
│                         #   detect_corners()         — Shi-Tomasi corner detection
│                         #   create_charts()          — Matplotlib statistical charts
│                         #   generate_heatmap()       — Motion heatmap with speed overlay
│                         #   detect_turbulence()      — Statistical turbulence detection
│                         #   draw_turbulence_warning()— Warning overlay on turbulent frames
│
├── tracker.py            # Lucas-Kanade optical flow tracking logic
│                         #   track_points(old_gray, new_gray, points)
│                         #   draw_tracked_points(frame, points)
│
├── app.py                # (auxiliary / standalone runner)
│
├── requirements.txt      # Python dependencies (opencv-python, numpy, matplotlib)
│
├── videos/               # Input video files (.mp4 / .avi)
│   ├── flying.mp4
│   └── ...
│
└── output/               # Generated output images and analytics
    ├── charts/           # 3 statistical tracking charts per video
    ├── heatmaps/         # 1 motion heatmap per video (with speed stats overlay)
    ├── <video>_output_corners.jpg
    ├── <video>_output_frame_mid1.jpg   ← ⚠ overlay shown if turbulent
    ├── <video>_output_frame_mid2.jpg   ← ⚠ overlay shown if turbulent
    └── <video>_output_final.jpg        ← ⚠ overlay shown if turbulent
```

---

## How to Run

### 1. Clone or download the project

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **requirements.txt** contains:
> ```
> opencv-python
> numpy
> matplotlib
> ```

### 3. Place input videos

Put your `.mp4` or `.avi` video files inside the `videos/` folder:

```
videos/
├── flying.mp4
└── your_video.mp4
```

### 4. Run the project

```bash
python main.py
```

### 5. View output

Results are saved to the `output/` folder — **4 images + 3 analytical charts + 1 motion heatmap per video**:

```
output/
├── charts/
│   ├── flying_corner_distribution.png
│   ├── flying_displacement.png
│   └── flying_tracking_performance.png
├── heatmaps/
│   └── flying_heatmap.png              ← Motion heatmap with speed stats overlay
├── flying_output_corners.jpg           ← Frame 0: detected corners (green)
├── flying_output_frame_mid1.jpg        ← ~33%: tracked points (yellow) [⚠ if turbulent]
├── flying_output_frame_mid2.jpg        ← ~67%: tracked points (yellow) [⚠ if turbulent]
└── flying_output_final.jpg             ← Final frame: tracked points (yellow) [⚠ if turbulent]
```

No GUI window will open. All output is saved directly as image files.

---

## Tech Stack

| **Technology** | **Version** | **Purpose** |
|---|---|---|
| **Python** | 3.10+ | Core programming language |
| **OpenCV (`cv2`)** | 4.x | Image processing, corner detection, optical flow |
| **NumPy** | 1.x | Array operations and numerical computation |
| **Matplotlib** | 3.x | Generating analytical and statistical tracking charts |

> No deep learning frameworks, no pre-trained models, no external APIs.  
> Pure classical Computer Vision — fully explainable and lightweight.
