#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2

# Your existing parser
from mmwave_bin_parser import parse as parse_mmwave_bin

# -------------------------------------------------------------------------
# ArUco dictionaries
# -------------------------------------------------------------------------
DICT_MAP = {
    "APRILTAG_36H11": cv2.aruco.DICT_APRILTAG_36h11,
    "ARUCO_4X4_50":   cv2.aruco.DICT_4X4_50,
    "ARUCO_5X5_100":  cv2.aruco.DICT_5X5_100,
    "ARUCO_6X6_250":  cv2.aruco.DICT_6X6_250,
    "ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    # short aliases
    "4X4_50":         cv2.aruco.DICT_4X4_50,
    "5X5_100":        cv2.aruco.DICT_5X5_100,
    "6X6_250":        cv2.aruco.DICT_6X6_250,
}

AUTO_DICTS = [
    "APRILTAG_36H11",
    "ARUCO_4X4_50", "ARUCO_5X5_100", "ARUCO_6X6_250", "ARUCO_ORIGINAL",
    "4X4_50", "5X5_100", "6X6_250"
]

# -------------------------------------------------------------------------
# IO helpers
# -------------------------------------------------------------------------
def load_radar_csv(csv_path: Path):
    ts_by_frame = {}
    pts_by_frame = {}
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            fn = int(row["frame_num"])
            if fn not in pts_by_frame:
                pts_by_frame[fn] = []
                ts_by_frame[fn] = int(float(row["timestamp_us"]))
            pts_by_frame[fn].append([
                float(row["x"]), float(row["y"]), float(row["z"]),
                float(row["power_snr"]), float(row["range_m"])
            ])
    for k in list(pts_by_frame.keys()):
        pts_by_frame[k] = np.array(pts_by_frame[k], dtype=np.float64)  # x y z snr range
    return pts_by_frame, ts_by_frame

# -------------------------------------------------------------------------
# Axis mapping search
# -------------------------------------------------------------------------
def gen_axis_mappings():
    mats = []
    I = np.eye(3)
    perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
    signs = [(1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1),(-1,1,1),(-1,1,-1),(-1,-1,1),(-1,-1,-1)]
    for p in perms:
        P = I[:, list(p)]
        for s in signs:
            S = np.diag(np.array(s, float))
            M = P @ S
            if np.linalg.det(M) > 0.5:
                mats.append(M)
    return mats

def score_mapping_via_bins(M: np.ndarray, pts: np.ndarray) -> float:
    if pts.size == 0:
        return -1.0
    XYZm = (M @ pts[:,:3].T).T
    az = np.degrees(np.arctan2(XYZm[:,1], XYZm[:,0]))  # [-180,180]
    left = ((az >= -30) & (az <= -10)).sum()
    center = ((az >= -5) & (az <= 5)).sum()
    right = ((az >= 10) & (az <= 30)).sum()
    return 1.0*left + 1.2*center + 1.3*right

def pick_axis_mapping_auto(pts_by_frame: Dict[int, np.ndarray], max_frames: int = 20) -> np.ndarray:
    frames = sorted(pts_by_frame.keys(), key=lambda k: -len(pts_by_frame[k]))[:max_frames]
    Ms = gen_axis_mappings()
    best_M, best = None, -1e9
    for M in Ms:
        s = 0.0
        for f in frames:
            s += score_mapping_via_bins(M, pts_by_frame[f])
        if s > best:
            best, best_M = s, M
    return best_M

# -------------------------------------------------------------------------
# Video timestamps and sync
# -------------------------------------------------------------------------
def video_frame_timestamps(video_path: Path, start_us: int, fps: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    step_us = int(round(1e6 / fps))
    return [start_us + i*step_us for i in range(n)]

def sync_video_to_radar(video_ts: List[int], radar_ts_by_frame: Dict[int, int]):
    radar_frames = sorted(radar_ts_by_frame.items(), key=lambda kv: kv[0])
    r_keys = [k for k,_ in radar_frames]
    r_ts   = [t for _,t in radar_frames]
    map_vid_to_rad = []
    j = 0
    for t in video_ts:
        best = j
        best_err = abs(r_ts[j] - t)
        while j + 1 < len(r_ts) and abs(r_ts[j+1] - t) <= best_err:
            j += 1
            best = j
            best_err = abs(r_ts[j] - t)
        map_vid_to_rad.append(best)
    return r_keys, map_vid_to_rad


def find_frame_with_tags(video_path: Path, start_idx: int, search_frames: int, want_tags: int, dict_hint: str = "auto"):
    """
    Searches video frames starting from start_idx to find one containing at least 'want_tags' markers.
    Returns (frame_index, frame, centers, used_dict_name)
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    best = (None, None, [], None)  # index, frame, centers, dict

    for offset in range(-search_frames, search_frames + 1):
        idx = start_idx + offset
        if idx < 0 or idx >= total
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        centers, used_dict, score = detect_centers_with_auto_dict(frame, want=want_tags, dict_hint=dict_hint)
        if centers is not None and len(centers) >= want_tags:
            best = (idx, frame, centers, used_dict)
            break

    cap.release()
    return best


# -------------------------------------------------------------------------
# Tag detection and association
# -------------------------------------------------------------------------
def detector_params():
    P = cv2.aruco.DetectorParameters()
    P.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    P.adaptiveThreshWinSizeMin = 5
    P.adaptiveThreshWinSizeMax = 53
    P.adaptiveThreshWinSizeStep = 4
    P.adaptiveThreshConstant = 7
    P.minMarkerPerimeterRate = 0.02
    P.maxMarkerPerimeterRate = 4.0
    P.minCornerDistanceRate = 0.01
    return P

def detect_centers_with_auto_dict(frame: np.ndarray, want: int, dict_hint: str = "auto"):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dicts = AUTO_DICTS if dict_hint.lower() == "auto" else [dict_hint]
    best = ([], None, 0)  # centers, dict_name, area_sum
    P = detector_params()
    for dname in dicts:
        if dname not in DICT_MAP:
            continue
        ad = cv2.aruco.getPredefinedDictionary(DICT_MAP[dname])
        det = cv2.aruco.ArucoDetector(ad, P)
        corners, ids, _ = det.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            continue
        centers = np.array([c[0].mean(axis=0) for c in corners], float)
        areas = np.array([cv2.contourArea(c.astype(np.float32)) for c in corners])
        score = areas.sum()
        if (centers.shape[0] >= want and score > best[2]) or (best[1] is None and centers.shape[0] > 0):
            best = (centers, dname, score)
    return best  # centers, chosen_dict, score

def associate_by_angle_and_range(centers_uv, XYZm, snr, angle_wins, range_wins, min_snr):
    order = np.argsort(centers_uv[:,0])  # left to right
    img_pts, obj_pts = [], []
    az = np.degrees(np.arctan2(XYZm[:,1], XYZm[:,0]))
    rng = np.linalg.norm(XYZm, axis=1)
    for k, t_idx in enumerate(order.tolist()):
        win_az = angle_wins.get(k, (-60, 60))
        win_r  = range_wins.get(k, (0.2, 15.0))
        m = (az >= win_az[0]) & (az <= win_az[1]) & (rng >= win_r[0]) & (rng <= win_r[1]) & (snr >= min_snr)
        if not np.any(m):
            continue
        pick = np.argmin(np.abs(az - np.clip(az[m].mean(), -90, 90)))  # mild bias toward center of window
        img_pts.append(centers_uv[t_idx])
        obj_pts.append(XYZm[pick])
    return img_pts, obj_pts

# One largest marker center
def detect_one_center(frame: np.ndarray, dict_name: str):
    if dict_name not in DICT_MAP:
        return None
    ad = cv2.aruco.getPredefinedDictionary(DICT_MAP[dict_name])
    P = detector_params()
    det = cv2.aruco.ArucoDetector(ad, P)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None
    areas = [cv2.contourArea(c.astype(np.float32)) for c in corners]
    j = int(np.argmax(areas))
    return corners[j][0].mean(axis=0)  # (u, v)

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fx", type=float, default=1426.152525681587)
    ap.add_argument("--fy", type=float, default=1429.8315107353205)
    ap.add_argument("--cx", type=float, default=734.3439390137369)
    ap.add_argument("--cy", type=float, default=326.75890302293806)
    ap.add_argument("--dist", type=float, nargs="+", default=[0.16448811, -0.26391235, -0.0043684, 0.02723059, 0.45555558])
    ap.add_argument("--aruco_dict", default="auto")
    ap.add_argument("--frameskip", type=int, default=1)
    ap.add_argument("--min_snr", type=float, default=30.0)
    ap.add_argument("--tag_count", type=int, default=3)
    ap.add_argument("--expected_ranges", nargs="+", default=["0:0.4,8.0","1:0.4,8.0","2:0.4,8.0"])
    ap.add_argument("--expected_angles", nargs="+", default=["0:-30,-10","1:-5,5","2:10,30"])
    ap.add_argument("--manual_map", nargs=9, type=float)
    ap.add_argument("--search_frames", type=int, default=30)

    # Moving single trihedral options
    ap.add_argument("--moving_single", action="store_true", help="Accumulate 1 trihedral + 1 tag across frames")
    ap.add_argument("--az_gate_deg", type=float, default=30.0, help="azimuth gate around pixel yaw")
    ap.add_argument("--elev_gate_deg", type=float, default=12.0, help="elevation gate around 0 deg")
    ap.add_argument("--snr_pct", type=float, default=70.0, help="keep points above this SNR percentile per frame")
    ap.add_argument("--keep_every", type=int, default=1, help="subsample harvested pairs")
    ap.add_argument("--ransac_iter", type=int, default=12000)
    ap.add_argument("--ransac_err_px", type=float, default=8.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse radar to CSV if needed
    csv_path = out_dir / "radar_parsed.csv"
    if not csv_path.exists():
        parse_mmwave_bin(Path(args.bin), csv_path)
    pts_by_frame, ts_by_frame = load_radar_csv(csv_path)
    print(f"Radar frames: {len(pts_by_frame)}, CSV: {csv_path}")

    # Axis mapping
    if args.manual_map and len(args.manual_map) == 9:
        M = np.array(args.manual_map, dtype=float).reshape(3,3)
        print("[mapping] manual M=\n", M)
    else:
        M = pick_axis_mapping_auto(pts_by_frame, max_frames=20)
        print("[mapping] auto M=\n", M)

    # Timestamps and sync
    meta = json.loads(Path(args.meta).read_text())
    start_us = int(meta["video"]["start_ms"]) * 1000
    fps = float(meta["video"]["frame_rate"])
    video_ts = video_frame_timestamps(Path(args.video), start_us, fps)
    radar_keys, vid_to_rad = sync_video_to_radar(video_ts, ts_by_frame)

    # Intrinsics
    K = np.array([[args.fx, 0, args.cx],[0, args.fy, args.cy],[0,0,1]], dtype=float)
    dist = np.array(args.dist, dtype=float)

    if args.moving_single:
        print("[moving_single] harvest pairs across frames")
        cap = cv2.VideoCapture(str(args.video))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pairs = []  # (uv, XYZm, frame_idx, radar_key, az, yaw_cam, snr)
        used_dict = args.aruco_dict if args.aruco_dict.lower() != "auto" else "4X4_50"

        def harvest_at(i):
            if i < 0 or i >= n_total:
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok:
                return None
            uv = detect_one_center(fr, used_dict)
            if uv is None:
                return None
            ridx = vid_to_rad[i] if i < len(vid_to_rad) else -1
            if ridx < 0 or ridx >= len(radar_keys):
                return None
            key = radar_keys[ridx]
            P = pts_by_frame[key]  # x y z snr range

            XYZm = (M @ P[:,:3].T).T
            snr  = P[:,3]
            # per-frame SNR percentile cut
            thr = np.percentile(snr, args.snr_pct) if snr.size > 0 else -np.inf
            az   = np.degrees(np.arctan2(XYZm[:,1], XYZm[:,0]))
            rng  = np.linalg.norm(XYZm, axis=1)
            elev = np.degrees(np.arctan2(XYZm[:,2], np.maximum(1e-6, np.sqrt(XYZm[:,0]**2 + XYZm[:,1]**2))))

            # pixel yaw from u
            u = float(uv[0])
            yaw_cam = np.degrees(np.arctan((u - args.cx) / args.fx))

            mask = (snr >= thr) & (rng > 0.4) & (rng < 12.0) \
                   & (np.abs(az - yaw_cam) < args.az_gate_deg) \
                   & (np.abs(elev) < args.elev_gate_deg)

            if not np.any(mask):
                return None
            # pick point closest in az to yaw_cam with SNR as tiebreaker
            cand = np.where(mask)[0]
            j = int(cand[np.lexsort(((-snr[cand]).astype(np.float64), np.abs(az[cand] - yaw_cam)))][0])
            return (uv, XYZm[j], i, key, float(az[j]), float(yaw_cam), float(snr[j]))

        for i in range(n_total):
            got = harvest_at(i)
            if got is None and args.search_frames > 0:
                for off in range(1, args.search_frames+1):
                    got = harvest_at(i+off) or harvest_at(i-off)
                    if got is not None:
                        break
            if got is not None:
                pairs.append(got)

        cap.release()
        print(f"[moving_single] harvested pairs: {len(pairs)}")

        if len(pairs) < 8:
            raise RuntimeError(f"Only {len(pairs)} pairs. Move the trihedral more or loosen gates.")

        # Subsample if requested
        if args.keep_every > 1:
            pairs = pairs[::max(1, args.keep_every)]
            print(f"[moving_single] after subsample keep_every={args.keep_every}: {len(pairs)} pairs")

        # Pre-filter with stricter azimuth agreement
        pre = []
        for (uv, xyz, i, key, az, yaw_cam, snr) in pairs:
            if abs(az - yaw_cam) < max(6.0, 0.5 * args.az_gate_deg):
                pre.append((uv, xyz, i, key, az, yaw_cam, snr))
        if len(pre) >= 8:
            pairs = pre
            print(f"[moving_single] after azimuth prefilter: {len(pairs)} pairs")

        img_pts = np.array([p[0] for p in pairs], float).reshape(-1,1,2)
        obj_pts = np.array([p[1] for p in pairs], float).reshape(-1,1,3)

        # RANSAC PnP
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj_pts, img_pts, K, dist,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=float(args.ransac_err_px),
            iterationsCount=int(args.ransac_iter),
            confidence=0.999
        )
        if not ok:
            raise RuntimeError("PnP RANSAC failed. Try lower snr_pct, larger az_gate_deg, or more pairs.")

        # refine on inliers
        rvec, tvec = cv2.solvePnPRefineLM(obj_pts[inl], img_pts[inl], K, dist, rvec, tvec)
        proj, _ = cv2.projectPoints(obj_pts[inl], rvec, tvec, K, dist)
        err = np.linalg.norm(proj.reshape(-1,2) - img_pts[inl].reshape(-1,2), axis=1)
        rmse = float(np.sqrt((err**2).mean()))
        print(f"[moving_single] inliers: {len(inl)}/{len(pairs)}, RMSE: {rmse:.2f} px")

        # Save extrinsic
        R, _ = cv2.Rodrigues(rvec)
        import yaml
        yaml.safe_dump({
            "R": R.tolist(),
            "t": tvec.reshape(-1).tolist(),
            "mapping": M.tolist(),
            "rmse_px": rmse,
            "mode": "moving_single",
            "dict": used_dict,
            "pairs": int(len(pairs)),
            "inliers": int(len(inl))
        }, open(out_dir / "extrinsic.yaml","w"))
        print("Saved extrinsic.yaml")

        # Overlay video with highlights
        print("[overlay] rendering overlay video")
        cap = cv2.VideoCapture(str(args.video))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_out = cap.get(cv2.CAP_PROP_FPS)
        outv = cv2.VideoWriter(str(out_dir / "overlay.mp4"), fourcc, fps_out, (w,h))

        # index inlier frames
        inlier_set = set(int(pairs[k][2]) for k in inl.ravel().tolist())

        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            ridx = vid_to_rad[i] if i < len(vid_to_rad) else -1
            if 0 <= ridx < len(radar_keys):
                key = radar_keys[ridx]
                P = pts_by_frame[key]
                XYZm_all = (M @ P[:,:3].T).T
                proj_all, _ = cv2.projectPoints(XYZm_all.reshape(-1,1,3), rvec, tvec, K, dist)
                for (u,v) in proj_all.reshape(-1,2).astype(int):
                    if 0 <= u < w and 0 <= v < h:
                        cv2.circle(fr, (u,v), 2, (0,255,0), -1)
            # draw harvested pair on this frame
            for (uv, xyz, vid_i, key, _, _, _) in pairs:
                if vid_i == i:
                    proj_one, _ = cv2.projectPoints(np.asarray(xyz, float).reshape(1,1,3), rvec, tvec, K, dist)
                    u1, v1 = proj_one.reshape(2).astype(int)
                    col = (255,0,255) if vid_i in inlier_set else (0,165,255)  # magenta for inlier, orange for outlier
                    if 0 <= u1 < w and 0 <= v1 < h:
                        cv2.circle(fr, (u1, v1), 10, col, 2)
                    u2, v2 = np.asarray(uv, float).astype(int)
                    cv2.circle(fr, (u2, v2), 6, (0,0,255), 2)
            outv.write(fr)
            i += 1
        cap.release()
        outv.release()
        print("Overlay written.")
        return

    # --------------------
    # Multi tag in a single frame (previous mode)
    # --------------------
    # pick a dense radar frame near a frame with tags and solve as before
    best_frame = max(pts_by_frame.keys(), key=lambda k: len(pts_by_frame[k]))
    ridx = radar_keys.index(best_frame)
    vid_idx = vid_to_rad.index(ridx) if ridx in vid_to_rad else 0

    choose_idx, frame, centers, used_dict = find_frame_with_tags(Path(args.video), vid_idx, args.search_frames, args.tag_count, dict_hint=args.aruco_dict)
    if frame is None or len(centers) == 0:
        cv2.imwrite(str(out_dir / "debug_no_tags.jpg"), frame if frame is not None else np.zeros((480,640,3), np.uint8))
        raise RuntimeError("Could not find a frame with detectable tags. Wrote debug_no_tags.jpg")
    print(f"[detect] used dict: {used_dict}, centers: {len(centers)}, at video frame {choose_idx}")

    # Parse windows
    def parse_pair(items):
        out = {}
        for s in items:
            k, rng = s.split(":")
            a, b = rng.split(",")
            out[int(k)] = (float(a), float(b))
        return out
    angle_wins = parse_pair(args.expected_angles)
    range_wins = parse_pair(args.expected_ranges)

    rkey = radar_keys[vid_to_rad[choose_idx]]
    P = pts_by_frame[rkey]
    XYZm = (M @ P[:,:3].T).T
    img_pts, obj_pts = associate_by_angle_and_range(centers, XYZm, P[:,3], angle_wins, range_wins, args.min_snr)

    n = len(img_pts)
    if n < 3:
        cv2.imwrite(str(out_dir / "debug_frame.jpg"), frame)
        with open(out_dir / "debug_points.txt", "w") as f:
            for row in P[:,:5]:
                f.write(",".join(map(str, row.tolist())) + "\n")
        raise RuntimeError(f"Need at least 3 correspondences, got {n}. Saved debug_frame.jpg and debug_points.txt")

    img = np.asarray(img_pts, float).reshape(-1,1,2)
    obj = np.asarray(obj_pts, float).reshape(-1,1,3)

    if n >= 4:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj, img, K, dist,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=float(args.ransac_err_px),
            iterationsCount=int(args.ransac_iter),
            confidence=0.999
        )
        if not ok:
            raise RuntimeError("PnP RANSAC failed with 4 or more points. Try wider gates.")
    else:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            rvec = np.zeros((3,1), np.float64)
            tvec = np.zeros((3,1), np.float64)
            ok, rvec, tvec = cv2.solvePnP(
                obj, img, K, dist, rvec, tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                raise RuntimeError("PnP failed with 3 points. Try wider gates or add a 4th correspondence.")

    try:
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    except Exception as e:
        print("[warn] solvePnPRefineLM failed, continuing without refine:", e)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1,2) - img.reshape(-1,2), axis=1)
    rmse = float(np.sqrt((err**2).mean()))
    print(f"[calibration] RMSE {rmse:.2f} px")

    # Save extrinsic
    R, _ = cv2.Rodrigues(rvec)
    import yaml
    yaml.safe_dump({
        "R": R.tolist(),
        "t": tvec.reshape(-1).tolist(),
        "mapping": M.tolist(),
        "rmse_px": rmse,
        "mode": "multi_tag",
        "dict": used_dict,
        "frame": int(choose_idx)
    }, open(out_dir / "extrinsic.yaml","w"))
    print("Saved extrinsic.yaml")

    # Overlay
    print("[overlay] rendering overlay video")
    cap = cv2.VideoCapture(str(args.video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps_out = cap.get(cv2.CAP_PROP_FPS)
    outv = cv2.VideoWriter(str(out_dir / "overlay.mp4"), fourcc, fps_out, (w,h))
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        ridx = vid_to_rad[i] if i < len(vid_to_rad) else -1
        if 0 <= ridx < len(radar_keys):
            key = radar_keys[ridx]
            P = pts_by_frame[key]
            XYZm = (M @ P[:,:3].T).T
            proj_all, _ = cv2.projectPoints(XYZm.reshape(-1,1,3), rvec, tvec, K, dist)
            for (u,v) in proj_all.reshape(-1,2).astype(int):
                if 0 <= u < w and 0 <= v < h:
                    cv2.circle(fr, (u,v), 2, (0,255,0), -1)
        outv.write(fr)
        i += 1
    cap.release()
    outv.release()
    print("Overlay written.")

if __name__ == "__main__":
    main()
