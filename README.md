# Autonomous Racing Path Planner & Cone Sorter

![0403](https://github.com/user-attachments/assets/88eece45-8fee-49ac-bf58-fb49eda2af8e)


## Project Overview
This project features an autonomous racing simulation that maps unknown environments in real-time. Using a simulated **Field of View (FOV)** sensor, the vehicle detects surrounding cones, classifies them into left/right boundaries, and dynamically generates a smooth racing line (centerline) using spline interpolation.

### Key Features:
* **Dynamic FOV Mapping:** Only cones within the 120° sensor range are "perceived" and mapped.
* **Coupled Cone Sorting:** An interleaved growth algorithm that builds track boundaries simultaneously to prevent lane crossing in sharp turns.
* **Path Smoothing:** Utilizes `scipy.interpolate.splprep` to generate a drivable path from raw midpoint data.
* **Temporal Filtering:** Smooths path transitions between frames to ensure stable vehicle movement.

---

## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
