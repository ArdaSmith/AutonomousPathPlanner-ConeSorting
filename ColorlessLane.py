import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Wedge
from scipy.interpolate import splprep, splev

# -----------------------------
# BASIC UTILS
# -----------------------------
def distance(a, b):
    return np.linalg.norm(a - b)

def angle_between(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

# -----------------------------
# DEDUPLICATION
# -----------------------------
def deduplicate(cones, thresh=0.5):
    unique = []
    for c in cones:
        if not any(distance(c, u) < thresh for u in unique):
            unique.append(c)
    return np.array(unique)

# -----------------------------
# HEADING-BASED SPLIT
# -----------------------------
def split_cones(cones, veh_pos, veh_heading):
    left, right = [], []

    forward = np.array([np.cos(veh_heading), np.sin(veh_heading)])
    right_vec = np.array([forward[1], -forward[0]])

    for c in cones:
        rel = c - veh_pos
        side = np.dot(rel, right_vec)

        if side > 0:
            right.append(c)
        else:
            left.append(c)

    return left, right

# -----------------------------
# ADAPTIVE DISTANCE
# -----------------------------
def estimate_spacing(cones):
    if len(cones) < 2:
        return 5.0

    dists = []
    for i in range(len(cones)):
        for j in range(i+1, len(cones)):
            dists.append(distance(cones[i], cones[j]))

    return np.median(dists)

# -----------------------------
# GREEDY STEP
# -----------------------------
def find_next_cone(current, prev, candidates, max_dist, max_angle=np.pi/2):

    best = None
    best_score = float('inf')

    direction = current - prev if prev is not None else None

    for c in candidates:
        d = distance(current, c)
        if d > max_dist:
            continue

        if direction is not None:
            new_dir = c - current
            ang = angle_between(direction, new_dir)

            if ang > max_angle:
                continue
        else:
            ang = 0

        score = d + 2.5 * ang

        if score < best_score:
            best_score = score
            best = c

    return best

# -----------------------------
# PATH GROWTH
# -----------------------------
def grow_path(seed, cones, max_dist, veh_heading):
    path = [seed]
    
    remaining = [c for c in cones if not np.array_equal(c, seed)]

    forward_vec = np.array([np.cos(veh_heading), np.sin(veh_heading)])
    prev = seed - forward_vec 
    current = seed

    while True:
        nxt = find_next_cone(current, prev, remaining, max_dist)

        if nxt is None:
            break

        path.append(nxt)
        remaining = [c for c in remaining if not np.array_equal(c, nxt)]

        prev = current
        current = nxt

    return path

# -----------------------------
# RESAMPLING (COUPLING CORE)
# -----------------------------
def resample_path(path, n=60):
    path = np.array(path)

    if len(path) < 2:
        return path

    dists = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    dists = np.insert(dists, 0, 0)

    if dists[-1] == 0:
        return path

    t = dists / dists[-1]
    t_new = np.linspace(0, 1, n)

    new = np.vstack([
        np.interp(t_new, t, path[:, 0]),
        np.interp(t_new, t, path[:, 1])
    ]).T

    return new

# -----------------------------
# MAIN SORTING
# -----------------------------
def coupled_cone_sorting(cones, veh_pos, veh_heading):

    cones = deduplicate(cones)

    if len(cones) < 4:
        return [], []

    left_cones, right_cones = split_cones(cones, veh_pos, veh_heading)

    if len(left_cones) < 2 or len(right_cones) < 2:
        return [], []

    left_seed = min(left_cones, key=lambda c: distance(c, veh_pos))
    right_seed = min(right_cones, key=lambda c: distance(c, veh_pos))

    spacing = estimate_spacing(cones)
    max_dist = spacing * 1.5

    left_path = grow_path(left_seed, left_cones, max_dist, veh_heading)
    right_path = grow_path(right_seed, right_cones, max_dist, veh_heading)

    left_np = resample_path(left_path)
    right_np = resample_path(right_path)

    min_len = min(len(left_np), len(right_np))

    return left_np[:min_len], right_np[:min_len]

# -----------------------------
# SMOOTHING
# -----------------------------
def safe_smooth_path(path_points, num_points=150):
    if len(path_points) < 3:
        return path_points

    try:
        tck, u = splprep([path_points[:, 0], path_points[:, 1]], s=5.0, k=2)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        return np.vstack([x_new, y_new]).T
    except:
        return path_points

# -----------------------------
# TEMPORAL FILTER
# -----------------------------
def temporal_smooth(prev, new, alpha=0.7):
    if prev is None or len(prev) != len(new):
        return new
    return alpha * prev + (1 - alpha) * new

# -----------------------------
# FOV DETECTION
# -----------------------------
def update_seen_cones(all_cones, seen_flags, veh_pos, veh_heading, fov_radius, fov_angle):

    for i, cone in enumerate(all_cones):

        dist = np.linalg.norm(cone - veh_pos)
        if dist > fov_radius:
            continue

        angle_to_cone = np.arctan2(cone[1] - veh_pos[1], cone[0] - veh_pos[0])
        angle_diff = (angle_to_cone - veh_heading + np.pi) % (2 * np.pi) - np.pi

        if abs(angle_diff) <= fov_angle:
            seen_flags[i] = True

    return seen_flags

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    df = pd.read_csv("gt_cones.csv", skiprows=2)
    all_cones = df[['x', 'y']].to_numpy()

    # If the cones are not located around origin(0,0),make your starting point all_cones[0].
    veh_pos = np.array([0.0, 0.0]) 

    # Start the vehicle heading (X axis / right side)
    veh_heading = 0.0

    INTERVAL = 50
    REAL_SPEED = 20
    SPEED = REAL_SPEED * (INTERVAL / 1000)

    # FOV Settings
    FOV_RADIUS = 22.0
    FOV_ANGLE = np.radians(60) # ±60 degrees (120 Total)

    seen_flags = np.zeros(len(all_cones), dtype=bool)
    prev_centerline = None

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    ax.grid(True)

    # Create the Graphics
    all_scatter, = ax.plot(all_cones[:, 0], all_cones[:, 1], 'ko', alpha=0.1, markersize=2, label="All Cones")
    seen_scatter, = ax.plot([], [], 'ro', markersize=2, label="Seen Cones")
    left_line, = ax.plot([], [], 'b-', linewidth=1, label="Left Bound")
    right_line, = ax.plot([], [], 'y-', linewidth=1, label="Right Bound")
    path_line, = ax.plot([], [], 'r-', linewidth=2, label="Centerline")
    veh_marker, = ax.plot([], [], 'ks', markersize=4, label="Vehicle")

    fov_wedge = Wedge(veh_pos, FOV_RADIUS, 0, 0, color='green', alpha=0.15, label="Sensor FOV")
    ax.add_patch(fov_wedge)
    ax.legend(loc="upper left")

    def update(frame):
        global veh_pos, veh_heading, seen_flags, prev_centerline

        seen_flags = update_seen_cones(
            all_cones, seen_flags,
            veh_pos, veh_heading,
            FOV_RADIUS, FOV_ANGLE
        )

        known = deduplicate(all_cones[seen_flags])

        if len(known) > 4:
            left, right = coupled_cone_sorting(known, veh_pos, veh_heading)

            if len(left) > 1 and len(right) > 1:
                center = (left + right) / 2
                center = safe_smooth_path(center)

                center = temporal_smooth(prev_centerline, center)
                prev_centerline = center

                path_line.set_data(center[:, 0], center[:, 1])
                left_line.set_data(left[:, 0], left[:, 1])
                right_line.set_data(right[:, 0], right[:, 1])

                closest_idx = np.argmin(np.linalg.norm(center - veh_pos, axis=1))
                target_idx = closest_idx
                dist_accum = 0.0
                
                for i in range(closest_idx, len(center) - 1):
                    dist_accum += np.linalg.norm(center[i+1] - center[i])
                    if dist_accum >= SPEED:
                        target_idx = i + 1
                        break

                target = center[target_idx]
                direction = target - veh_pos
                dist = np.linalg.norm(direction)

                if dist > 0.05:
                    direction = direction / dist
                    veh_heading = np.arctan2(direction[1], direction[0])
                    veh_pos = target 

        if len(known) > 0:
            seen_scatter.set_data(known[:, 0], known[:, 1])

        veh_marker.set_data([veh_pos[0]], [veh_pos[1]])

        fov_wedge.set_center(veh_pos)
        fov_wedge.set_theta1(np.degrees(veh_heading - FOV_ANGLE))
        fov_wedge.set_theta2(np.degrees(veh_heading + FOV_ANGLE))

        # Focus POV on the vehicle
        ax.set_xlim(veh_pos[0] - 15, veh_pos[0] + 40)
        ax.set_ylim(veh_pos[1] - 25, veh_pos[1] + 25)

        return seen_scatter, path_line, veh_marker, left_line, right_line, fov_wedge

    ani = FuncAnimation(fig, update, frames=1500, interval=INTERVAL, blit=False, repeat=False)
    plt.show()