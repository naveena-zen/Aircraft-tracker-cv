# How to Run

### 1. Clone or download the project

```bash
git clone https://github.com/<naveena-zen>/<Aircraft-tracker-cv>.git
cd <Aircraft-tracker-cv>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

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
