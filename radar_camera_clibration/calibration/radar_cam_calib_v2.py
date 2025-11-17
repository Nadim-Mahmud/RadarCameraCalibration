#!/usr/bin/env python3

import os
import json
import csv
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import cv2
import pandas as pd
import itertools
from sklearn.cluster import DBSCAN
from pupil_apriltags import Detector

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
TAG_SIZE_METERS = 0.2032  # EXAMPLE: 20.32cm tag

# --- NEW: Spatial Window Matching ---
# This maps a specific AprilTag ID to its expected 3D location window
# in the RADAR's coordinate system.
# This REPLACES the permutation-guessing logic.
#
# Format: {TAG_ID: ( (min_az_deg, max_az_deg), (min_range_m, max_range_m) )}
TARGET_MAP = {
    2: ((-30.0, -10.0), (0.5, 3.0)), # Tag 26 is "left"
    3: ((-5.0,   5.0), (0.5, 3.0)), # Tag 27 is "center"
    5: ((10.0,  30.0), (0.5, 3.0)), # Tag 28 is "right"
}
# The specific tag IDs we are looking for
REQUIRED_TAG_IDS = set(TARGET_MAP.keys())


# --- Temporal Synchronization Parameters ---
FRAME_SYNC_TOLERANCE_MS = 200

# --- Radar Filtering Parameters ---
RADAR_INTENSITY_PERCENTILE = 60
RADAR_DBSCAN_EPS = 0.2  # 20cm
RADAR_DBSCAN_MIN_SAMPLES = 3

# --- Calibration Aggregation Parameters ---
MAX_VALID_CALIBRATION_RMSD = 0.1  # 10cm

# Camera Matrix
CAMERA_MATRIX = np.array([
    [1.36211720e+03, 0.00000000e+00, 7.45834826e+02],
    [0.00000000e+00, 1.35683586e+03, 2.43736808e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

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

    # 1. Load video information
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

    # 2. Load radar data
    df_radar = pd.read_csv(radar_csv_path)

    if "timestamp_us" in df_radar.columns:
        df_radar["radar_timestamp_ms"] = df_radar["timestamp_us"]
        print("Using 'timestamp_us' as radar timestamp (interpreted as milliseconds).")
    elif "timestamp" in df_radar.columns:
        df_radar.rename(columns={"timestamp": "radar_timestamp_ms"}, inplace=True)
        print("Renamed 'timestamp' → 'radar_timestamp_ms'.")
    else:
        raise KeyError("Radar CSV missing timestamp column ('timestamp_us' or 'timestamp').")

    df_radar["timestamp_dt"] = pd.to_datetime(df_radar["radar_timestamp_ms"], unit="ms")

    # Group radar points by timestamp
    df_radar_grouped = (
        df_radar.groupby("radar_timestamp_ms")
        .apply(lambda x: x[["x", "y", "z", "power_snr"]].to_dict("list"))
        .to_frame("radar_points_dict")
        .reset_index()
    )

    df_radar_grouped["timestamp_dt"] = pd.to_datetime(df_radar_grouped["radar_timestamp_ms"], unit="ms")

    # 3. Merge: assign to each video frame the nearest radar scan
    FRAME_SYNC_TOLERANCE_MS = getattr(config, "frame_sync_tolerance_ms", 250)

    df_sync = pd.merge_asof(
        df_video.sort_values("timestamp_dt"),
        df_radar_grouped.sort_values("timestamp_dt"),
        on="timestamp_dt",
        direction="nearest",
        tolerance=pd.Timedelta(f"{FRAME_SYNC_TOLERANCE_MS}ms")
    )

    # 4. Cleanup and diagnostics
    df_sync.dropna(subset=["radar_points_dict"], inplace=True)
    df_sync["timestamp_delta_ms"] = (
        (df_sync["video_timestamp_ms"] - df_sync["radar_timestamp_ms"]).abs()
    )

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
) -> Optional[Dict[int, np.ndarray]]:
    """
    Detects AprilTags and returns a dict mapping
    {tag_id: 3D_center_coordinates} for the required tags.
    """
    # 1. Undistort the image first
    frame_undistorted = cv2.undistort(frame, K, D)
    gray_undistorted = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
    
    # 2. Detect tags and estimate pose
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
    
    if not detections:
        return None

    # 3. Build dictionary of found required tags
    P_cam_dict = {}
    for d in detections:
        if d.tag_id in REQUIRED_TAG_IDS:
            P_cam_dict[d.tag_id] = d.pose_t.flatten()

    # 4. Validate: We need all 3 required tags for calibration
    if len(P_cam_dict) < 3:
        # print(f"  [Camera] Found {len(P_cam_dict)} of {len(REQUIRED_TAG_IDS)} required tags.")
        return None
            
    # 5. Return dict of 3D translation vectors (pose_t) 
    return P_cam_dict

def get_radar_targets(radar_points_dict: dict) -> Optional[Dict[int, np.ndarray]]:
    """
    Filters the radar point cloud for 3 trihedral reflectors
    by searching inside pre-defined spatial windows (from TARGET_MAP).
    Returns a dict mapping {tag_id: 3D_radar_centroid}
    """
    points = pd.DataFrame(radar_points_dict)
    if points.empty:
        return None
        
    points_3d = points[['x', 'y', 'z']].values
    intensities = points['power_snr'].values

    # 1. Intensity Filtering
    intensity_threshold = np.percentile(intensities, RADAR_INTENSITY_PERCENTILE)
    high_intensity_mask = (intensities > intensity_threshold)
    
    if not np.any(high_intensity_mask):
        return None

    points_3d_filtered = points_3d[high_intensity_mask]
    
    # 2. Pre-calculate Azimuth and Range for all filtered points
    # Azimuth = arctan(y, x)
    azimuth_deg = np.degrees(np.arctan2(points_3d_filtered[:, 1], points_3d_filtered[:, 0]))
    # Range = sqrt(x^2 + y^2 + z^2)
    range_m = np.linalg.norm(points_3d_filtered, axis=1)

    P_rad_dict = {}

    # 3. Find target for each ID in its window
    for tag_id, (az_window, range_window) in TARGET_MAP.items():
        
        # A. Find all points within this target's 3D window
        az_mask = (azimuth_deg >= az_window[0]) & (azimuth_deg <= az_window[1])
        range_mask = (range_m >= range_window[0]) & (range_m <= range_window[1])
        
        window_mask = az_mask & range_mask
        
        target_points = points_3d_filtered[window_mask]
        
        if len(target_points) == 0:
            # print(f"  [Radar] No points found for Tag {tag_id}")
            continue # No points found for this tag
        
        # B. Run DBSCAN *only* on points in this window
        # This finds the reflector *within* the window
        db = DBSCAN(eps=RADAR_DBSCAN_EPS, min_samples=RADAR_DBSCAN_MIN_SAMPLES).fit(target_points)
        labels = db.labels_
        
        unique_labels = set(labels) - {-1}
        if not unique_labels:
            # print(f"  [Radar] No cluster found for Tag {tag_id}")
            continue # Only noise found
        
        # C. Find the largest cluster in this window
        largest_cluster_label = -1
        max_points_in_cluster = 0
        
        for label in unique_labels:
            points_in_cluster = np.sum(labels == label)
            if points_in_cluster > max_points_in_cluster:
                max_points_in_cluster = points_in_cluster
                largest_cluster_label = label
                
        # D. Calculate the centroid of that largest cluster
        cluster_points = target_points[labels == largest_cluster_label]
        centroid = np.mean(cluster_points, axis=0)
        
        P_rad_dict[tag_id] = centroid
    
    # 4. Validate: We need all 3 targets
    if len(P_rad_dict) < 3:
        # print(f"  [Radar] Found {len(P_rad_dict)} of {len(REQUIRED_TAG_IDS)} required targets.")
        return None

    return P_rad_dict

# ----------------------------------------------------------------------------
# --- 4. 3D-to-3D REGISTRATION (PART 3) ---
# ----------------------------------------------------------------------------

def kabsch_algorithm(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves for the optimal rigid transformation (R, T) that aligns
    point set P to point set Q (i.e., Q = R @ P + T)
    using the Kabsch algorithm (SVD method).
    
    P, Q: Nx3 numpy arrays of 3D points (e.g., 3x3)
    
    Returns:
    R: 3x3 Rotation Matrix
    T: 3x1 Translation Vector
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

#
# --- FUNCTION `find_best_transform` IS NO LONGER NEEDED AND HAS BEEN REMOVED ---
#

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
    # print(f"number of radar points: {len(radar_points_dict['x'])}")
    radar_data = pd.DataFrame(radar_points_dict)
    if radar_data.empty:
        return

    points_3d = radar_data[['x', 'y', 'z']].values.astype(np.float32)
    rvec, _ = cv2.Rodrigues(R_preview)
    image_points_2d, _ = cv2.projectPoints(points_3d, rvec, T_preview, K, D)

    preview_frame = frame.copy()
    for pt in image_points_2d:
        p_2d = tuple(pt.ravel().astype(int))
        if 0 <= p_2d[0] < preview_frame.shape[1] and 0 <= p_2d[1] < preview_frame.shape[0]:
            cv2.circle(preview_frame, p_2d, 3, (0, 0, 255), -1)

    cv2.putText(preview_frame, "Radar projection preview (pre-calibration)",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imshow("Radar Projection Preview", preview_frame)
    cv2.waitKey(1)  # Changed to 1ms wait to allow processing
    # cv2.destroyWindow("Radar Projection Preview") # Removed to keep window open


# ----------------------------------------------------------------------------
# --- 5. MAIN CALIBRATION & VALIDATION (PART 4) ---
# ----------------------------------------------------------------------------

def main_calibration_pipeline():
    """
    Runs the full calibration pipeline:
    1. Loads and synchronizes data.
    2. Processes each (frame, scan) pair to find targets.
    3. Solves for R, T for each pair using explicit ID matching.
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
    at_detector = Detector(
        families='tagStandard41h12', # Make sure this matches your tags
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

        # --- (Optional) Visualize radar projection before calibration ---
        # print(f"\n[Frame {frame_idx}] Previewing radar projection before calibration...")
        # preview_radar_projection_before_calibration(
        #     frame,
        #     row['radar_points_dict'],
        #     config.K,
        #     config.D
        # )
        
        # --- Find 3D Targets in this Pair ---
        # A. Find 3 AprilTags in Camera 
        P_cam_dict = get_camera_targets(
            frame, 
            at_detector, 
            config.K, 
            config.D, 
            config.at_detector_params
        )

        if P_cam_dict is None:
            # print(f"Frame {frame_idx}: Failed to find 3 camera targets.")
            continue
            
        # B. Find 3 Trihedrals in Radar
        P_rad_dict = get_radar_targets(row['radar_points_dict'])
        if P_rad_dict is None:
            # print(f"Frame {frame_idx}: Failed to find 3 radar targets.")
            continue
            
        # --- We have points in both systems. Solve for (R, T) ---
        
        # C. Align Dictionaries to create ordered point sets
        # We now have *known correspondence* based on Tag IDs
        common_ids = sorted(list(P_cam_dict.keys() & P_rad_dict.keys()))
        
        if len(common_ids) < 3:
            print(f"Frame {frame_idx}: Mismatch in found IDs. Skipping.")
            continue
            
        # Create (Nx3) arrays in corresponding order
        P_cam_ordered = np.array([P_cam_dict[id] for id in common_ids])
        P_rad_ordered = np.array([P_rad_dict[id] for id in common_ids])
        
        # D. Solve 3D-to-3D registration directly with Kabsch
        R_frame, T_frame = kabsch_algorithm(P_rad_ordered, P_cam_ordered)
        
        # E. Calculate RMSD
        rmsd_frame = calculate_rmsd(P_rad_ordered, P_cam_ordered, R_frame, T_frame)
        
        if rmsd_frame > MAX_VALID_CALIBRATION_RMSD:
            print(f"Frame {frame_idx}: RMSD too high ({rmsd_frame*1000:.2f} mm). Skipping.")
            continue
            
        calibration_results.append({'R': R_frame, 'T': T_frame, 'rmsd': rmsd_frame})
        
        # F. Find the single BEST frame
        if rmsd_frame < global_min_rmsd:
            global_min_rmsd = rmsd_frame
            global_best_R = R_frame
            global_best_T = T_frame
            
        print(f"  Frame {frame_idx}: Found valid R, T. RMSD: {rmsd_frame*1000:.2f} mm")

    cap.release()
    cv2.destroyAllWindows()
    
    if not calibration_results:
        print("!!! CALIBRATION FAILED: No valid frames found with 3 targets in both sensors.")
        print("Check your TARGET_MAP windows, RADAR_INTENSITY_PERCENTILE, or required TAG_IDs.")
        return

    # 5. Finalize Parameters
    # We use the R, T from the single frame with the lowest RMSD.
    R_final = global_best_R
    T_final = global_best_T
    
    rvec_final, _ = cv2.Rodrigues(R_final)
    
    print("\n--- CALIBRATION COMPLETE ---")
    print(f"Found {len(calibration_results)} valid calibration frames.")
    print(f"Best Frame RMSD: {global_min_rmsd*1000:.2f} mm")
    
    print(f"\nFinal Rotation Matrix (R_final):\n{R_final}")
    print(f"\nFinal Translation Vector (T_final, meters):\n{T_final}")
    
    # 6. Save final parameters
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
            
        radar_data = pd.DataFrame(row['radar_points_dict'])
        if radar_data.empty:
            out.write(frame)
            continue
            
        points_3d = radar_data[['x', 'y', 'z']].values.astype(np.float32)
        
        # Project 3D radar points to 2D image plane
        image_points_2d, _ = cv2.projectPoints(points_3d, rvec, T, K, D)
        
        # Draw the projected points on the frame
        for pt in image_points_2d.reshape(-1, 2):
            p_2d = tuple(pt.astype(int))
            # Draw only points visible in the frame
            if 0 <= p_2d[0] < frame_width and 0 <= p_2d[1] < frame_height:
                cv2.circle(frame, p_2d, 3, (0, 0, 255), -1) # Red dots
                
        out.write(frame)
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Validation video saved to {OUTPUT_VIDEO_FILE}")

# (Other helper functions like project_camera_to_radar remain unchanged)

# ----------------------------------------------------------------------------
# --- 6. SCRIPT EXECUTION ---
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main_calibration_pipeline()