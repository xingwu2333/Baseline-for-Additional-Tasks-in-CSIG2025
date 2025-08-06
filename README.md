# 🎯 Baseline for additional tasks 
**The 6th Chinese Conference on China Society of Image and Graphics (CSIG) 2025 – Additional Tasks**

This repository provides a baseline for the additional tasks of the **CSIG 2025 Challenge**, using a **Detection-Based Tracking** approach. The detection results are obtained via the official toolkit of the challenge(https://github.com/TinaLRJ/DeepPro).

---

## 📁 File Descriptions

- **`track.py`**  
  Performs object tracking based on the provided detection results.

  **Key Parameters:**
  - `max_age = 30` → Maximum number of frames an unmatched track is kept.
  - `min_hits = 1` → Minimum number of detections before a track is confirmed.
  - `iou_threshold = 11` → IOU threshold used for data association.
  - `max_gap = 1` → Maximum gap (in frames) allowed between associations.

- **`merge_tracks.py`**  
  Optionally merge fragmented tracks to enhance continuity and improve scoring. Use as needed.

- **`val_scores.txt`**  
  Validation set performance scores one the official competition platform.

---

## 💡 Requirements

Install the following Python packages:

```bash
pip install filterpy scipy collections
