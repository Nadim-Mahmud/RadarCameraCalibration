# Spatio-Temporal Calibration Workflow (Radar ↔ Camera)

[link to dataset](https://drive.google.com/drive/folders/1M5XB60L1uQoCYZoJW7_hKBz_cABmYTIN?usp=sharing)

- **Camera Calibration**

  - Capture images of a known calibration pattern (checkerboard / circle grid).
  - Use OpenCV (`cv2.findChessboardCorners`, `cv2.calibrateCamera`) to compute:
    - Intrinsic matrix `K`
    - Distortion coefficients `dist`
    - Per-frame camera pose (optional)
  - Save `K`, `dist`, and example poses for later projection.

- **Temporal Calibration**

  - Ensure both sensors record precise timestamps for each frame/detection.
  - Estimate time offset `Δt` between radar and camera:
    - Use correlated events (object motion peaks, flashes, or synchronized marker events).
    - Compute `Δt` by maximizing cross-correlation of event / motion signals or minimizing tracking error.
  - Apply `Δt` to reindex/shift frames so radar and camera frames represent the same physical instant.

- **Radar Coordinate → Camera Projection**

  - Define radar point coordinates `p_r = [x_r, y_r, z_r]^T` in radar frame.
  - Transform to camera frame using extrinsics `R` and `T`:
    ```
    p_c = R * p_r + T
    ```
  - Project to image plane using intrinsics `K` (and handle distortion with `dist`):
    ```
    p_img_homog = K * p_c
    u = p_img_homog[0] / p_img_homog[2]
    v = p_img_homog[1] / p_img_homog[2]
    ```
  - Optionally undistort or apply distortion model when reprojecting.

- **Matching (Association)**

  - Choose matching cues: spatial proximity, motion trajectory, Doppler/signature, or reflectivity.
  - We have used april tag to detect images.
  - For each camera detection (object bbox / keypoint), find nearest projected radar point(s) within a threshold (euclidean / Mahalanobis).
  - Use temporal windowing around aligned timestamps if uncertainty remains.
  - Optionally apply data-association algorithms (Nearest Neighbor, Hungarian, or probabilistic filters).

- **Extrinsic Parameter Optimization**

  - Form correspondence set `{(p_r_i, u_i,v_i)}` after matching.
  - Optimize `R` and `T` by minimizing reprojection error:
    \[
    \min*{R,T}\sum_i \big\|[u_i,v_i]^T - \pi\big(K(R p*{r_i}+T)\big)\big\|^2
    \]
  - Use nonlinear least squares (e.g., `scipy.optimize.least_squares` or Ceres) to refine extrinsics.
  - If temporal uncertainty exists, jointly optimize `Δt` with `R,T`.

- **Validation**

  - Visualize radar points projected onto camera frames across multiple timestamps.
  - Compute quantitative metrics:
    - Mean reprojection error (pixels)
    - Association precision / recall (if ground truth)
  - Test on varied scenes (different ranges, angles, and motion) and report statistics.

- **Refinement & Robustness**
  - Re-run optimization with robust loss (Huber / Tukey) to reduce outlier influence.
  - Use multi-scene / multi-target data to generalize extrinsics.
  - If available, incorporate IMU or GNSS to constrain motion and improve temporal alignment.

---

**Notes**

- Keep units consistent (meters, seconds).
- Log intermediate results (timestamps, matched pairs, reprojection error) for debugging and reproducibility.
