import cv2
import numpy as np
import pandas as pd
import json
import itertools
from sklearn.cluster import DBSCAN
from pupil_apriltags import Detector
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple

# ----------------------------------------------------------------------------
# --- 1. CONFIGURATION & HYPERPARAMETERS ---
# ----------------------------------------------------------------------------
# --- File Paths ---
DATA_DIR = "./data/SLESP5003_1001"
METADATA_FILE = "meta_data.json"
VIDEO_FILE = "SLESP5003_1001.mp4"
RADAR_CSV_FILE = "SLESP5003_1001.csv"
OUTPUT_VIDEO_FILE = "validation_projection.mp4"
CALIBRATION_OUTPUT_FILE = "radar_camera_extrinsics.npz"

# --- Calibration Target Parameters ---
# CRITICAL: This MUST be set to the physical size of the AprilTag's
# black square edge, measured in meters.
TAG_SIZE_METERS = 0.2032  # EXAMPLE: 15cm tag

# --- Temporal Synchronization Parameters ---
# Radar is 5fps (200ms). Camera is 10fps (100ms).
# A 100ms tolerance ensures we take the nearest radar scan
# that is at most 100ms away from the video frame.
FRAME_SYNC_TOLERANCE_MS = 200

# --- Radar Filtering Parameters ---
# The percentile of intensity to use as a filter threshold.
# Trihedral reflectors should be in the top 10%.
RADAR_INTENSITY_PERCENTILE = 60
# DBSCAN: Max distance (meters) between points to be considered a cluster.
RADAR_DBSCAN_EPS = 0.2  # 20cm
# DBSCAN: Min number of points to form a dense cluster.
RADAR_DBSCAN_MIN_SAMPLES = 3

# --- Calibration Aggregation Parameters ---
# The maximum RMSD (meters) to consider a frame's
# calibration result as "valid" for aggregation.
MAX_VALID_CALIBRATION_RMSD = 0.1  # 10cm

# Camera Matrix
CAMERA_MATRIX = np.array([
    [1.36211720e+03, 0.00000000e+00, 7.45834826e+02],
    [0.00000000e+00, 1.35683586e+03, 2.43736808e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

# Distortion Coefficients
# DIST_COEFFS = np.array([
#     [-3.92408433e-01, 4.65319512e+00, -7.72263689e-03,
#      1.79067356e-02, -2.68995581e+01]
# ])

# No distortion
DIST_COEFFS = np.zeros((1, 5))


@dataclass
class SensorRigConfig:
    """Container for all loaded configuration data."""
    K: np.ndarray
    D: np.ndarray
    video_start_ms: int
    video_fps: float
    radar_start_ms: int
    at_detector_params: list
    frame_sync_tolerance_ms: int


# ----------------------------------------------------------------------------
# --- 2. DATA LOADING & SYNCHRONIZATION (PART 1) ---
# ----------------------------------------------------------------------------

def load_config(meta_path: str) -> SensorRigConfig:
    """Loads all metadata and camera priors."""
    # 1. Load Camera Intrinsics [6]
    K, D = CAMERA_MATRIX, DIST_COEFFS

    # 2. Load Session Metadata 
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    video_start_ms = int(metadata['video']['start_ms'])
    video_fps = float(metadata['video']['frame_rate'])
    radar_start_ms = int(metadata['radar']['start_ms'])
    
    # 3. Create params list for pupil-apriltags detector 
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    at_detector_params = [fx, fy, cx, cy]

    return SensorRigConfig(
        K=K,
        D=D,
        video_start_ms=video_start_ms,
        video_fps=video_fps,
        radar_start_ms=radar_start_ms,
        at_detector_params=at_detector_params,
        frame_sync_tolerance_ms=FRAME_SYNC_TOLERANCE_MS
    )

def synchronize_data(config: SensorRigConfig, video_path: str, radar_csv_path: str) -> pd.DataFrame:
    """
    Synchronizes radar scans to video frames.
    Uses video as the reference (master timeline) and assigns to each frame
    the nearest radar scan within a time tolerance.
    """
    print("Time synchronizing sensor data...")

    # -------------------------------------------------------------
    # 1. Load video information
    # -------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frame_indices = np.arange(total_frames)
    frame_interval_ms = 1000.0 / config.video_fps
    video_timestamps_ms = config.video_start_ms + (frame_indices * frame_interval_ms)

    df_video = pd.DataFrame({
        "frame_index": frame_indices,
        "video_timestamp_ms": video_timestamps_ms
    })
    df_video["timestamp_dt"] = pd.to_datetime(df_video["video_timestamp_ms"], unit="ms")

    # -------------------------------------------------------------
    # 2. Load radar data
    # -------------------------------------------------------------
    df_radar = pd.read_csv(radar_csv_path)

    # --- Identify timestamp column ---
    if "timestamp_us" in df_radar.columns:
        df_radar["radar_timestamp_ms"] = df_radar["timestamp_us"]  # already in ms even if mislabeled
        print("Using 'timestamp_us' as radar timestamp (interpreted as milliseconds).")
    elif "timestamp" in df_radar.columns:
        df_radar.rename(columns={"timestamp": "radar_timestamp_ms"}, inplace=True)
        print("Renamed 'timestamp' → 'radar_timestamp_ms'.")
    else:
        raise KeyError("Radar CSV missing timestamp column ('timestamp_us' or 'timestamp').")

    df_radar["timestamp_dt"] = pd.to_datetime(df_radar["radar_timestamp_ms"], unit="ms")

    # --- Group radar points by timestamp ---
    df_radar_grouped = (
        df_radar.groupby("radar_timestamp_ms")
        .apply(lambda x: x[["x", "y", "z", "power_snr"]].to_dict("list"))
        .to_frame("radar_points_dict")
        .reset_index()
    )

    df_radar_grouped["timestamp_dt"] = pd.to_datetime(df_radar_grouped["radar_timestamp_ms"], unit="ms")

    # -------------------------------------------------------------
    # 3. Merge: assign to each video frame the nearest radar scan
    # -------------------------------------------------------------
    FRAME_SYNC_TOLERANCE_MS = getattr(config, "frame_sync_tolerance_ms", 250)  # default: ±250 ms

    df_sync = pd.merge_asof(
        df_video.sort_values("timestamp_dt"),
        df_radar_grouped.sort_values("timestamp_dt"),
        on="timestamp_dt",
        direction="nearest",
        tolerance=pd.Timedelta(f"{FRAME_SYNC_TOLERANCE_MS}ms")
    )

    # -------------------------------------------------------------
    # 4. Cleanup and diagnostics
    # -------------------------------------------------------------
    df_sync.dropna(subset=["radar_points_dict"], inplace=True)

    df_sync["timestamp_delta_ms"] = (
        (df_sync["video_timestamp_ms"] - df_sync["radar_timestamp_ms"]).abs()
    )


    print(f"Video timestamps: {df_video['video_timestamp_ms'].min()} - {df_video['video_timestamp_ms'].max()}")
    print(f"Radar timestamps: {df_radar_grouped['radar_timestamp_ms'].min()} - {df_radar_grouped['radar_timestamp_ms'].max()}")
    print(f"Synchronization complete: {len(df_sync)} frame–scan pairs within ±{FRAME_SYNC_TOLERANCE_MS} ms tolerance.")

    print(df_sync[["frame_index", "video_timestamp_ms", "radar_timestamp_ms", "timestamp_delta_ms"]].head().astype(str))

    return df_sync[
        ["frame_index", "video_timestamp_ms", "radar_timestamp_ms", "timestamp_delta_ms", "radar_points_dict"]
    ]

# ----------------------------------------------------------------------------
# --- 3. TARGET IDENTIFICATION (PART 2) ---
# ----------------------------------------------------------------------------

def get_camera_targets(
    frame: np.ndarray, 
    detector: Detector, 
    K: np.ndarray, 
    D: np.ndarray, 
    camera_params: list
) -> Optional[np.ndarray]:
    """
    Detects AprilTags and returns the 3D coordinates of their centers.
    Returns a 3x3 array of [p_c1, p_c2, p_c3] or None if not found.
    [26, 27, 28, 29]
    """
    # 1. Undistort the image first [29, 31]
    frame_undistorted = cv2.undistort(frame, K, D)
    gray_undistorted = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
    
    # 2. Detect tags and estimate pose [27, 28]
    try:
        detections = detector.detect(
            gray_undistorted,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=TAG_SIZE_METERS
        )
    except Exception as e:
        print(f"  [Camera] Error during AprilTag detection: {e}")
        return None
    
    
    # print(f"  [Camera] Detected {len(detections)} AprilTags.")

    # # -----------------------------------------------------------------
    # # --- NEW VISUALIZATION / DEBUG BLOCK ---
    # # -----------------------------------------------------------------
    #     # Create a color copy to draw on
    # vis_frame = frame_undistorted.copy()
        
    # # Draw all found tags
    # for tag in detections:
    #     corners = tag.corners.astype(int)
    #     center = tuple(tag.center.astype(int))
        
    #     # Draw bounding box
    #     cv2.polylines(vis_frame, [corners], True, (0, 255, 0), 2)
    #     # Draw center
    #     cv2.circle(vis_frame, center, 5, (0, 0, 255), -1)
    #     # Draw Tag ID
    #     cv2.putText(vis_frame, f"ID: {tag.tag_id}",
    #                 (corners[0][0], corners[0][1] - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    # # Add a helpful status text
    # status_text = f"Found {len(detections)} tags"
    # if len(detections) != 3:
    #     status_text += " (FAIL - Need 3)"
    #     color = (0, 0, 255) # Red
    # else:
    #     status_text += " (OK)"
    #     color = (0, 255, 0) # Green
        
    # cv2.putText(vis_frame, status_text, (20, 40), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # # Show the image and WAIT for a key press
    # print("    [Debug] Showing detection window. Press any key to continue...")
    # cv2.imshow("AprilTag Detection Debug", vis_frame)
    # cv2.waitKey(0) # <-- This pauses the script for user input
    # cv2.destroyWindow("AprilTag Detection Debug") # <-- This closes the figure
    # print("    [Debug] Window closed, continuing...")
    # # -----------------------------------------------------------------
    # # --- END OF VISUALIZATION BLOCK ---
    # # -----------------------------------------------------------------

    # 3. Validate: We need exactly 3 tags for calibration
    if len(detections)!= 3:
        return None
            
    # 4. Extract 3D translation vectors (pose_t) 
    P_cam = np.array([d.pose_t.flatten() for d in detections])
    return P_cam

def get_radar_targets(radar_points_dict: dict) -> Optional[np.ndarray]:
    """
    Filters the radar point cloud for 3 trihedral reflectors.
    Returns a 3x3 array of [p_r1, p_r2, p_r3] or None if not found.
    [34, 35, 36, 40, 41]
    """

    # print("  [Radar] Processing radar points to find trihedrals...")

    points = pd.DataFrame(radar_points_dict)
    if points.empty:
        return None
    
    # print(f"  [Radar] Total points in scan: {len(points)}")
    
        
    points_3d = points[['x', 'y', 'z']].values
    intensities = points['power_snr'].values


    # 1. Intensity Filtering [32, 37]
    # Trihedrals have high RCS, so they will be high-intensity outliers
    intensity_threshold = np.percentile(intensities, RADAR_INTENSITY_PERCENTILE)
    high_intensity_mask = (intensities > intensity_threshold)
    
    high_intensity_points = points_3d[high_intensity_mask]

    # print(f"[Radar] Intensity threshold: {intensity_threshold:.2f}")
    
    if len(high_intensity_points) < 3: # Not enough points
        return None
    
    # print(f"[Radar] Points after intensity filtering: {len(high_intensity_points)}")

    # 2. Density-Based Clustering (DBSCAN) [40, 41, 43]
    # We use DBSCAN over K-Means because it doesn't require k=3
    # and can discover the true number of clusters, making it robust to noise.
    db = DBSCAN(eps=RADAR_DBSCAN_EPS, min_samples=RADAR_DBSCAN_MIN_SAMPLES).fit(high_intensity_points)
    labels = db.labels_
    
    # Get unique cluster labels, excluding noise (label -1)
    unique_labels = set(labels) - {-1}

    # print(f"[Radar] DBSCAN found {len(unique_labels)} clusters.")

    # 3. Validate: We need exactly 3 clusters
    if len(unique_labels) < 3:
        return None

    # 4. Calculate Centroid of each cluster
    # 4. Calculate size and centroid for each cluster
    cluster_info = []
    for label in unique_labels:
        cluster_points = high_intensity_points[labels == label]
        point_count = len(cluster_points)
        centroid = np.mean(cluster_points, axis=0)
        # Store all info for sorting
        cluster_info.append({
            'label': label, 
            'count': point_count, 
            'centroid': centroid
        })

    # 5. Sort clusters by point count (largest first)
    sorted_clusters = sorted(cluster_info, key=lambda x: x['count'], reverse=True)
    
    # 6. Keep only the top 3 largest clusters
    top_3_clusters = sorted_clusters[:3]

    # 7. Extract the centroids of just these top 3
    P_rad_list = [cluster['centroid'] for cluster in top_3_clusters]
    
    return np.array(P_rad_list)

# ----------------------------------------------------------------------------
# --- 4. 3D-to-3D REGISTRATION (PART 3) ---
# ----------------------------------------------------------------------------

def kabsch_algorithm(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves for the optimal rigid transformation (R, T) that aligns
    point set P to point set Q (i.e., Q = R @ P + T)
    using the Kabsch algorithm (SVD method).
    
    P, Q: 3xN numpy arrays of 3D points
    
    Returns:
    R: 3x3 Rotation Matrix
    T: 3x1 Translation Vector
    
    [48, 50, 51, 52, 53, 54, 55]
    """
    
    # 1. Calculate Centroids 
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    # 2. Center Points
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # 3. Compute Covariance Matrix H 
    # Note: We use (N, 3) format, so H = P_centered.T @ Q_centered
    H = P_centered.T @ Q_centered
    
    # 4. Compute SVD [55]
    U, S, Vt = np.linalg.svd(H)
    
    # 5. Compute Rotation R [51]
    R = Vt.T @ U.T
    
    # 6. Handle Reflection (Critical Step) 
    # If det(R) < 0, it's a reflection. We must correct it.
    if np.linalg.det(R) < 0:
        Vt_corrected = Vt.copy()
        Vt_corrected[-1, :] *= -1
        R = Vt_corrected.T @ U.T
        
    # 7. Compute Translation T 
    T = centroid_Q.T - (R @ centroid_P.T)
    
    return R, T.reshape((3, 1)) # Return T as a 3x1 column vector

def calculate_rmsd(P: np.ndarray, Q: np.ndarray, R: np.ndarray, T: np.ndarray) -> float:
    """Calculates the RMSD between two point sets given R and T."""
    P_transformed = (R @ P.T).T + T.T
    errors = Q - P_transformed
    squared_errors = np.sum(errors**2, axis=1)
    mean_squared_error = np.mean(squared_errors)
    return np.sqrt(mean_squared_error)

def find_best_transform(P_rad: np.ndarray, P_cam: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Solves the 3D-to-3D registration problem with *unknown correspondence*
    by testing all 3! = 6 permutations.
    
    [45, 46, 47]
    """
    min_rmsd = np.inf
    best_R = None
    best_T = None

    # Brute-force all 3! = 6 permutations [45]
    for P_rad_permuted in itertools.permutations(P_rad):
        P_rad_np = np.array(P_rad_permuted)
        
        # 1. Solve with Kabsch algorithm [48, 51]
        R_cand, T_cand = kabsch_algorithm(P_rad_np, P_cam)
        
        # 2. Calculate RMSD for this permutation
        rmsd = calculate_rmsd(P_rad_np, P_cam, R_cand, T_cand)
        
        # 3. Keep the best one
        if rmsd < min_rmsd:
            min_rmsd = rmsd
            best_R = R_cand
            best_T = T_cand
            
    if best_R is None or min_rmsd > MAX_VALID_CALIBRATION_RMSD:
        return None # This frame is not high-quality
        
    return best_R, best_T, min_rmsd


def preview_radar_projection_before_calibration(
    frame: np.ndarray,
    radar_points_dict: dict,
    K: np.ndarray,
    D: np.ndarray,
    R_preview: np.ndarray = np.eye(3),
    T_preview: np.ndarray = np.zeros((3, 1))
):
    """
    Projects raw radar 3D points onto the camera image using a temporary 
    (identity) transformation to visualize alignment before calibration.
    """

    print(f"number of radar points: {len(radar_points_dict['x'])}")

    radar_data = pd.DataFrame(radar_points_dict)
    if radar_data.empty:
        print("No radar points to visualize.")
        return

    # Convert radar points to 3D NumPy array
    points_3d = radar_data[['x', 'y', 'z']].values.astype(np.float32)

    # Convert rotation to Rodrigues vector for cv2.projectPoints
    rvec, _ = cv2.Rodrigues(R_preview)

    # Project radar 3D points into the 2D image plane
    image_points_2d, _ = cv2.projectPoints(points_3d, rvec, T_preview, K, D)

    # Draw the projected points
    preview_frame = frame.copy()
    for pt in image_points_2d:
        p_2d = tuple(pt.ravel().astype(int))
        if 0 <= p_2d[0] < preview_frame.shape[1] and 0 <= p_2d[1] < preview_frame.shape[0]:
            cv2.circle(preview_frame, p_2d, 3, (0, 0, 255), -1)  # red radar points

    # Add overlay text
    cv2.putText(preview_frame, "Radar projection preview (pre-calibration)",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Display
    cv2.imshow("Radar Projection Preview", preview_frame)
    cv2.waitKey(0)  # wait for key press
    cv2.destroyWindow("Radar Projection Preview")


# ----------------------------------------------------------------------------
# --- 5. MAIN CALIBRATION & VALIDATION (PART 4) ---
# ----------------------------------------------------------------------------

def main_calibration_pipeline():
    """
    Runs the full calibration pipeline:
    1. Loads and synchronizes data.
    2. Processes each (frame, scan) pair to find targets.
    3. Solves for R, T for each pair.
    4. Aggregates results to find the final R_final, T_final.
    5. Saves the final parameters.
    """
    print("--- Starting Radar-Camera Calibration Pipeline ---")
    
    # 1. Load Config
    config = load_config(
        os.path.join(DATA_DIR, METADATA_FILE),
    )
    
    # 2. Synchronize Data
    df_sync = synchronize_data(
        config,
        os.path.join(DATA_DIR, VIDEO_FILE),
        os.path.join(DATA_DIR, RADAR_CSV_FILE)
    )

    if df_sync.empty:
        print("!!! ERROR: No synchronized frames found. Check data and timestamps.")
        return
    
    # 3. Initialize Detectors
    # 
    at_detector = Detector(
        families='tagStandard41h12',
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25
    )
    
    cap = cv2.VideoCapture(os.path.join(DATA_DIR, VIDEO_FILE))

    # 4. Process Frames and Find Best Transform
    calibration_results = []
    global_min_rmsd = np.inf
    global_best_R = None
    global_best_T = None
    
    print(f"Processing {len(df_sync)} synchronized frames...")
    
    for _, row in df_sync.iterrows():
        frame_idx = int(row['frame_index'])
        
        # Seek to the correct video frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # --- Visualize radar projection before calibration ---
        print(f"\n[Frame {frame_idx}] Previewing radar projection before calibration...")
        preview_radar_projection_before_calibration(
            frame,
            row['radar_points_dict'],
            config.K,
            config.D
        )
        
        # --- Find 3D Targets in this Pair ---
        # A. Find 3 AprilTags in Camera 
        P_cam = get_camera_targets(
            frame, 
            at_detector, 
            config.K, 
            config.D, 
            config.at_detector_params
        )

        # print(f" Processing Camera Frame {frame_idx}...")

        if P_cam is None:
            # print(f"Frame {frame_idx}: Failed to find 3 camera targets.")
            continue
            
        print(f"Processing Radar Scan for Frame {frame_idx}...")
        # B. Find 3 Trihedrals in Radar [41]
        P_rad = get_radar_targets(row['radar_points_dict'])
        if P_rad is None:
            print(f"Frame {frame_idx}: Failed to find 3 radar targets.")
            continue
            
        # --- We have 3 points in both systems. Solve for (R, T) ---
        # C. Solve 3D-to-3D registration [45, 51]
        result = find_best_transform(P_rad, P_cam)
        
        if result is None:
            print(f"Frame {frame_idx}: RMSD too high. Skipping.")
            continue
            
        R_frame, T_frame, rmsd_frame = result
        calibration_results.append({'R': R_frame, 'T': T_frame, 'rmsd': rmsd_frame})
        
        # D. Find the single BEST frame
        if rmsd_frame < global_min_rmsd:
            global_min_rmsd = rmsd_frame
            global_best_R = R_frame
            global_best_T = T_frame
            
        print(f"  Frame {frame_idx}: Found valid R, T. RMSD: {rmsd_frame*1000:.2f} mm")

    cap.release()
    
    if not calibration_results:
        print("!!! CALIBRATION FAILED: No valid frames found with 3 targets in both sensors.")
        return

    # 5. Finalize Parameters
    # We use the R, T from the single frame with the lowest RMSD.
    R_final = global_best_R
    T_final = global_best_T
    # Alternative: Average all valid T vectors
    # T_avg = np.mean(np.array( for res in calibration_results]), axis=0)
    
    # Convert R to rvec for saving 
    rvec_final, _ = cv2.Rodrigues(R_final)
    
    print("\n--- CALIBRATION COMPLETE ---")
    print(f"Found {len(calibration_results)} valid calibration frames.")
    print(f"Best Frame RMSD: {global_min_rmsd*1000:.2f} mm")
    
    print(f"\nFinal Rotation Matrix (R_final):\n{R_final}")
    print(f"\nFinal Translation Vector (T_final, meters):\n{T_final}")
    print(f"\nFinal Rodrigues Vector (rvec_final):\n{rvec_final}")
    
    # 6. Save final parameters [3, 31]
    np.savez(
        os.path.join(DATA_DIR, CALIBRATION_OUTPUT_FILE),
        R_final=R_final,
        T_final=T_final,
        rvec_final=rvec_final,
        K_intrinsic=config.K,
        D_intrinsic=config.D
    )
    print(f"\nFinal calibration parameters saved to {CALIBRATION_OUTPUT_FILE}")
    
    # 7. Run Validation Projection
    validate_projection(
        R_final, 
        T_final, 
        config,
        df_sync,
        os.path.join(DATA_DIR, VIDEO_FILE)
    )

def validate_projection(
    R: np.ndarray, 
    T: np.ndarray, 
    config: SensorRigConfig, 
    df_sync: pd.DataFrame, 
    video_path: str
):
    """
    Projects all radar points onto the video using the final
    extrinsic parameters to create a validation video.
    [31, 58, 59, 60, 61]
    """
    print(f"--- Running Validation: Projecting radar points onto video ---")
    
    cap = cv2.VideoCapture(video_path)
    
    # Setup video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_format = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(
        os.path.join(DATA_DIR, OUTPUT_VIDEO_FILE), 
        out_format, 
        config.video_fps, 
        (frame_width, frame_height)
    )
    
    # Convert R to rvec for cv2.projectPoints 
    rvec, _ = cv2.Rodrigues(R)
    K, D = config.K, config.D
    
    for _, row in df_sync.iterrows():
        frame_idx = int(row['frame_index'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Get all 3D points from this radar scan
        radar_data = pd.DataFrame(row['radar_points_dict'])
        if radar_data.empty:
            out.write(frame)
            continue
            
        points_3d = radar_data[['x', 'y', 'z']].values.astype(np.float32)
        
        # Project 3D radar points to 2D image plane [58]
        image_points_2d, _ = cv2.projectPoints(points_3d, rvec, T, K, D)
        
        # Draw the projected points on the frame
        for pt in image_points_2d:
            # Squeeze to (x, y)
            p_2d = tuple(pt.astype(int))
            # Draw only points visible in the frame
            if 0 <= p_2d < frame_width and 0 <= p_2d < frame_height:
                cv2.circle(frame, p_2d, 3, (0, 0, 255), -1) # Red dots
                
        out.write(frame)
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Validation video saved to {OUTPUT_VIDEO_FILE}")

def project_camera_to_radar(
    P_cam: np.ndarray, 
    R_final: np.ndarray, 
    T_final: np.ndarray
) -> np.ndarray:
    """
    Projects 3D points from the Camera coordinate system to the
    Radar coordinate system. (Inverse of R_final, T_final)
    
    P_cam: Nx3 array of 3D points in camera coordinates
    """
    # P_cam = R @ P_rad + T
    # P_rad = R.T @ (P_cam - T) = R.T @ P_cam - R.T @ T
    
    R_inv = R_final.T
    T_inv = -R_final.T @ T_final
    
    # Handle single (3,) or (3,1) vector
    if P_cam.ndim == 1:
        P_cam = P_cam.reshape(1, 3)
        
    # Handle (N, 3) array
    P_cam_T = P_cam.T # (3, N)
    P_rad_T = (R_inv @ P_cam_T) + T_inv
    
    return P_rad_T.T # (N, 3)

# ----------------------------------------------------------------------------
# --- 6. SCRIPT EXECUTION ---
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main_calibration_pipeline()
