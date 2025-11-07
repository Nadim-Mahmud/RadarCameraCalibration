#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
radar_cam_pipeline.py
End-to-end pipeline for AWR1843BOOST + Raspberry Pi Camera Module 3

Features
- Parse TI mmWave UART .bin (xwr18xx_mmw_demo TLVs 1 & 7) with per-1024B timestamps
- Detect ArUco markers in video frames
- Synchronize radar frames to video frames using metadata JSON
- Associate ArUco centers with radar points (left→right vs azimuth)
- Solve radar→camera extrinsics via PnP (RANSAC + LM refinement)
- Render overlay video of projected radar points

Usage example:
python3 radar_cam_pipeline.py   --bin radar.bin --video camera.mp4 --meta meta.json --out_dir ./out   --fx 1426.15252568 --fy 1429.83151074 --cx 734.34393901 --cy 326.75890302   --dist 0.16448811 -0.26391235 -0.0043684 0.02723059 0.45555558   --aruco_dict 4X4_50 --min_snr 0 --frameskip 1
"""
import argparse, json, csv, math, os, struct, sys
from pathlib import Path
import numpy as np
import cv2


def filter_by_azimuth(radar_points, az_min=85, az_max=120):
    """
    Filters radar points based on azimuth (in degrees).
    """
    return [p for p in radar_points if az_min <= float(p[6]) <= az_max]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_metadata(meta_path: Path):
    with open(meta_path, "r") as f:
        return json.load(f)

def timestamps_video_frames(meta: dict):
    v = meta["video"]
    start = int(v["start_ms"])  # ms
    end   = int(v["end_ms"])
    fps   = float(v["frame_rate"])
    nframes = max(1, int(round((end - start) * fps / 1000.0)))
    ms = np.linspace(start, end, nframes, endpoint=False)
    return ms.astype(np.int64)  # milliseconds

def sync_frames(video_ts_ms, radar_ts_ms, max_gap_ms=80):
    max_gap = max_gap_ms
    radar = np.array(radar_ts_ms, dtype=np.int64)
    out = np.full(len(video_ts_ms), -1, dtype=np.int32)
    if len(radar) == 0:
        return out
    for i, t in enumerate(video_ts_ms):
        j = int(np.searchsorted(radar, t))
        best = None
        if j < len(radar): best = (abs(radar[j] - t), j)
        if j > 0:
            d2 = abs(radar[j-1] - t)
            if best is None or d2 < best[0]:
                best = (d2, j-1)
        if best and best[0] <= max_gap:
            out[i] = best[1]
    return out


MAGIC = b'\x02\x01\x04\x03\x06\x05\x08\x07'
HDR_FMT = "<8sIIIIIIII"
HDR_SZ  = struct.calcsize(HDR_FMT)
TLV_HDR_FMT = "<II"
TLV_HDR_SZ  = struct.calcsize(TLV_HDR_FMT)
TLV_POINTS    = 1
TLV_SIDE_INFO = 7

def iter_timestamped_chunks(bin_path: Path):
    with open(bin_path, "rb") as f:
        while True:
            ts = f.read(8)
            if len(ts) < 8:
                break
            chunk = f.read(1024)
            if not chunk:
                break
            yield struct.unpack("<Q", ts)[0], chunk

def parse_mmwave_to_csv(bin_path: Path, out_csv: Path, max_frames=None, debug_first_n=0):
    stream = bytearray()
    ts_at_offset = {}
    for ts, chunk in iter_timestamped_chunks(bin_path):
        ts_at_offset[len(stream)] = ts
        stream.extend(chunk)

    i = 0
    frames = rows = 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_us","frame_num","x","y","z","v","snr","noise","range_m","doppler_mps","azimuth_deg","power_snr"])

        while True:
            j = stream.find(MAGIC, i)
            if j < 0 or j + HDR_SZ > len(stream):
                break

            magic, version, total_len, platform, frame_num, cpu, num_det, num_tlvs, sub =                 struct.unpack_from(HDR_FMT, stream, j)
            if total_len <= 0 or j + total_len > len(stream):
                break

            stamp_offsets = [off for off in ts_at_offset if off <= j]
            ts = ts_at_offset[max(stamp_offsets)] if stamp_offsets else 0

            tlv_off = j + HDR_SZ
            points = []
            side   = []

            tlv_debug = []
            for _ in range(num_tlvs):
                if tlv_off + TLV_HDR_SZ > j + total_len:
                    break
                tlv_type, tlv_len = struct.unpack_from(TLV_HDR_FMT, stream, tlv_off)
                payload_off = tlv_off + TLV_HDR_SZ
                payload_end = payload_off + tlv_len
                if payload_end > j + total_len:
                    break
                payload = stream[payload_off:payload_end]
                tlv_debug.append((tlv_type, tlv_len))

                if tlv_type == TLV_POINTS:
                    n_pts = min(num_det, tlv_len // 16)
                    for k in range(n_pts):
                        x, y, z, v = struct.unpack_from("<ffff", payload, k*16)
                        points.append((x, y, z, v))
                elif tlv_type == TLV_SIDE_INFO:
                    n_info = min(num_det, tlv_len // 4)
                    for k in range(n_info):
                        snr, noise = struct.unpack_from("<HH", payload, k*4)
                        side.append((snr, noise))

                tlv_off = payload_end

            for idx, (x, y, z, v) in enumerate(points):
                snr, noise = ("", "")
                if idx < len(side): snr, noise = side[idx]
                rng = math.sqrt(x*x + y*y + z*z)
                az  = math.degrees(math.atan2(y, x))
                w.writerow([ts, frame_num, x, y, z, v, snr, noise, rng, v, az, snr])
                rows += 1

            frames += 1
            i = j + total_len
            if max_frames and frames >= max_frames:
                break
    return frames, rows

ARUCO_DICTS = {
    "4X4_50":  cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_50":  cv2.aruco.DICT_5X5_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "6X6_50":  cv2.aruco.DICT_6X_50 if hasattr(cv2.aruco, "DICT_6X_50") else cv2.aruco.DICT_6X6_50,
    "APRILTAG_36H11": cv2.aruco.DICT_APRILTAG_36h11 if hasattr(cv2.aruco, "DICT_APRILTAG_36h11") else None,
}

def detect_aruco_centers(frame_bgr, aruco_dict_name):
    adict_id = ARUCO_DICTS.get(aruco_dict_name, cv2.aruco.DICT_4X4_50)
    dictionary = cv2.aruco.getPredefinedDictionary(adict_id)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    centers = []
    if ids is None:
        return centers
    ids = ids.flatten().tolist()
    for i, c in enumerate(corners):
        pts = c[0]
        u = float(pts[:,0].mean())
        v = float(pts[:,1].mean())
        centers.append((int(ids[i]), u, v))
    centers.sort(key=lambda x: x[1])
    return centers

def associate_left_to_right(tag_centers, radar_points, min_snr=0):
    if not tag_centers or not radar_points:
        return []
    tags_sorted = sorted(tag_centers, key=lambda t: t[1])
    filt = []
    for rp in radar_points:
        snr = rp[4]
        if snr == "" or float(snr) >= float(min_snr):
            filt.append(rp)
    r_sorted = sorted(filt, key=lambda r: r[6])
    n = min(len(tags_sorted), len(r_sorted))
    pairs = []
    for i in range(n):
        _, u, v = tags_sorted[i]
        x,y,z,vel,snr,noise,az = r_sorted[i][:8]
        pairs.append((u, v, x, y, z))
    return pairs


def associate_by_id0(tag_centers, radar_points, valid_ids=None, min_snr=10, expected_ranges=None):
    """
    expected_ranges: dict {id: (rmin, rmax)} in meters
    """
    pairs = []
    for tid, u, v in tag_centers:
        if valid_ids and tid not in valid_ids:
            continue
        # filter radar points
        rp = [p for p in radar_points if (p[4]=="" or float(p[4])>=min_snr)]
        if expected_ranges and tid in expected_ranges:
            rmin, rmax = expected_ranges[tid]
            rp = [p for p in rp if rmin <= np.linalg.norm([p[0],p[1],p[2]]) <= rmax]
        if not rp:
            continue
        # pick the highest SNR remaining
        best = max(rp, key=lambda p: float(p[4]) if p[4] != "" else 0)
        x,y,z,vel,snr,noise,az = best[:8]
        pairs.append((tid,u,v,x,y,z))
    return pairs

# ---------------- Association function ----------------
def associate_by_id(tag_centers, radar_points,
                    valid_ids=None, min_snr=10, expected_ranges=None, expected_angles=None):
    pairs = []
    for tid, u, v in tag_centers:
        if valid_ids and tid not in valid_ids:
            continue
        rp = [p for p in radar_points if (p[4] == "" or float(p[4]) >= min_snr)]
        if expected_ranges and tid in expected_ranges:
            rmin, rmax = expected_ranges[tid]
            rp = [p for p in rp if rmin <= np.linalg.norm([p[0],p[1],p[2]]) <= rmax]

        if expected_angles and tid in expected_angles:
            amin, amax = expected_angles[tid]
            # rp = [(i, p) for i, p in rp if amin <= float(p[6]) <= amax]  # p[6] = azimuth_deg
            rp = [p for p in rp if amin <= float(p[6]) <= amax]

        if not rp:
            continue
        best = max(rp, key=lambda p: float(p[4]) if p[4] != "" else 0)
        x, y, z, vel, snr, noise, az = best[:8]
        pairs.append((tid, u, v, x, y, z))
    return pairs

def associate_hybrid_confident(tag_centers, radar_points, valid_ids=None,
                               min_snr=50, expected_ranges=None, min_range_gap=1.0):
    if not tag_centers or not radar_points:
        return []

    # Filter radar points by SNR
    radar_filtered = [p for p in radar_points if p[4] == "" or float(p[4]) >= min_snr]
    if len(tag_centers) >= 3 and len(radar_filtered) >= 3:
        # Ordered left-to-right tags
        tag_centers_sorted = sorted(tag_centers, key=lambda t: t[1])

        # Nearest 3 radar points
        radar_sorted = sorted(radar_filtered, key=lambda p: (p[0]**2 + p[1]**2 + p[2]**2)**0.5)
        radar_top3 = radar_sorted[:3]
        ranges = [np.linalg.norm([r[0], r[1], r[2]]) for r in radar_top3]

        # Check if ranges are well-separated
        if (ranges[1] - ranges[0] >= min_range_gap) and (ranges[2] - ranges[1] >= min_range_gap):
            pairs = []
            for tag, radar in zip(tag_centers_sorted, radar_top3):
                tid, u, v = tag
                x, y, z, *_ = radar
                pairs.append((tid, u, v, x, y, z))
            return pairs  # Confident ordered match

    # Fallback to associate_by_id
    return associate_by_id(tag_centers, radar_points,
                           valid_ids=valid_ids,
                           min_snr=min_snr,
                           expected_ranges=expected_ranges)


def associate_by_ordered_range(tag_centers, radar_points, min_snr=0):
    if len(tag_centers) < 3 or len(radar_points) < 3:
        return []

    # Sort tag centers by u (horizontal position in image)
    tag_centers_sorted = sorted(tag_centers, key=lambda t: t[1])  # (id, u, v)

    # Filter radar points by SNR
    radar_filtered = [p for p in radar_points if p[4] == "" or float(p[4]) >= min_snr]
    if len(radar_filtered) < 3:
        return []

    # Sort radar points by increasing range
    radar_sorted = sorted(radar_filtered, key=lambda p: (p[0]**2 + p[1]**2 + p[2]**2)**0.5)

    pairs = []
    for tag, radar in zip(tag_centers_sorted, radar_sorted[:3]):
        tid, u, v = tag
        x, y, z, *_ = radar
        pairs.append((tid, u, v, x, y, z))
    return pairs


def associate_by_id_unique(tag_centers, radar_points,
                           valid_ids=None, min_snr=10, expected_ranges=None, expected_angles=None):
    """
    Associates radar points to tags, ensuring no radar point is reused across tags.
    """
    used_points = set()
    pairs = []

    for tid, u, v in tag_centers:
        if valid_ids and tid not in valid_ids:
            continue

        # Filter radar points by SNR and range
        rp = [
            (i, p) for i, p in enumerate(radar_points)
            if (p[4] == "" or float(p[4]) >= min_snr)
            and i not in used_points
        ]
        if expected_ranges and tid in expected_ranges:
            rmin, rmax = expected_ranges[tid]
            rp = [(i, p) for i, p in rp if rmin <= np.linalg.norm([p[0], p[1], p[2]]) <= rmax]

        if expected_angles and tid in expected_angles:
            amin, amax = expected_angles[tid]
            rp = [(i, p) for i, p in rp if amin <= float(p[6]) <= amax]  # p[6] = azimuth_deg

        if not rp:
            continue

        # Pick the highest SNR point
        best_i, best = max(rp, key=lambda x: float(x[1][4]) if x[1][4] != "" else 0)
        used_points.add(best_i)
        x, y, z, *_ = best
        pairs.append((tid, u, v, x, y, z))

    return pairs


def solve_extrinsics_pnp(all_pairs, K, dist):
    if len(all_pairs) < 6:
        raise RuntimeError("Not enough correspondences; need at least 6.")
    pts2d = np.array([[u, v] for (u,v,_,_,_) in all_pairs], dtype=np.float64)
    pts3d = np.array([[x, y, z] for (_,_,x,y,z) in all_pairs], dtype=np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts2d, K, dist,
        iterationsCount=300, reprojectionError=3.0, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnPRansac failed")
    rvec, tvec = cv2.solvePnPRefineLM(pts3d, pts2d, K, dist, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    proj = proj.reshape(-1,2)
    errs = np.linalg.norm(proj - pts2d, axis=1)
    stats = {
        "n_corr": int(len(pts3d)),
        "n_inliers": int(len(inliers)) if inliers is not None else None,
        "err_mean_px": float(errs.mean()),
        "err_median_px": float(np.median(errs)),
        "err_p95_px": float(np.percentile(errs, 95.0)),
    }
    return R, rvec.reshape(-1), tvec.reshape(-1), stats

def overlay_video(video_path: Path, out_path: Path, K, dist, rvec, tvec, radar_by_frame, video_to_radar_idx, radar_frame_keys, frameskip=1):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps/max(1,frameskip), (w,h))
    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if fidx % max(1,frameskip) != 0:
            fidx += 1
            continue
        ridx = video_to_radar_idx[fidx] if fidx < len(video_to_radar_idx) else -1
        if ridx >= 0:
            rfkey = radar_frame_keys[ridx]
            pts = radar_by_frame.get(rfkey, [])
            #Filter Radar Points to Expected Tag Range
            #pts = [p for p in radar_by_frame.get(rfkey, []) if 3.0 <= (p[0]**2 + p[1]**2 + p[2]**2)**0.5 <= 10.5]
            #pts = [p for p in radar_by_frame.get(rfkey, []) if 3.0 <= (p[0]**2 + p[1]**2 + p[2]**2)**4.0 <= 10.5]


            # pts = filter_by_azimuth(radar_by_frame.get(rfkey, []))

            if pts:
                Pr = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float64)
                uv, _ = cv2.projectPoints(Pr, rvec, tvec, K, dist)
                uv = uv.reshape(-1,2)
                for (u,v) in uv:
                    u_i, v_i = int(round(u)), int(round(v))
                    if 0 <= u_i < w and 0 <= v_i < h:
                        cv2.circle(frame, (u_i, v_i), 2, (0,255,0), -1)
        cv2.putText(frame, f"frame {fidx}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        out.write(frame)
        fidx += 1
    cap.release()
    out.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--fx", type=float, required=True)
    ap.add_argument("--fy", type=float, required=True)
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--cy", type=float, required=True)
    ap.add_argument("--dist", type=float, nargs="+", required=True, help="k1 k2 p1 p2 k3")
    ap.add_argument("--aruco_dict", default="4X4_50", choices=list(ARUCO_DICTS.keys()))
    ap.add_argument("--min_snr", type=float, default=0.0)
    ap.add_argument("--frameskip", type=int, default=1)
    ap.add_argument("--max_frames_parse", type=int, default=None)
    ap.add_argument("--debug_first_n", type=int, default=0)
    ap.add_argument("--tag_ids", type=int, nargs="+", default=None,
                help="Explicit list of ArUco IDs (left-to-right order).")
    ap.add_argument(
        "--expected_ranges",
        type=str, nargs="+", default=None,
        help="Expected ranges for each tag ID, format: id:rmin,rmax (meters). Example: --expected_ranges 0:3.5,4.5 1:6.5,7.5"
    )
    ap.add_argument('--expected_angles', nargs='+', default=[], help='Expected azimuth angle ranges for each tag ID, e.g., 0:-25,-15 1:15,25')

    args = ap.parse_args()

    # Parse expected ranges into dict
    expected_ranges = None
    if args.expected_ranges:
        expected_ranges = {}
        for spec in args.expected_ranges:
            tid, rng = spec.split(":")
            rmin, rmax = map(float, rng.split(","))
            expected_ranges[int(tid)] = (rmin, rmax)

    expected_angles = None
    if args.expected_angles:
        expected_angles = {}
        for spec in args.expected_angles:
            tid, rng = spec.split(":")
            amin, amax = map(float, rng.split(","))
            expected_angles[int(tid)] = (amin, amax)


    ensure_dir(args.out_dir)
    out_csv = args.out_dir / "parsed_radar.csv"
    print("[1/6] Parsing radar .bin -> CSV ...")
    frames_parsed, rows = parse_mmwave_to_csv(args.bin, out_csv, max_frames=args.max_frames_parse, debug_first_n=args.debug_first_n)
    print(f"    frames={frames_parsed}, points={rows}")

    print("[2/6] Loading metadata ...")
    meta = load_metadata(args.meta)
    video_ts_us = timestamps_video_frames(meta)

    print("[3/6] Group radar detections by frame ...")
    radar_by_frame = {}
    radar_frame_keys = []
    radar_ts_us = []
    seen = set()
    with open(out_csv, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            fnum = int(row["frame_num"])
            tsus = int(row["timestamp_us"])
            x = float(row["x"]); y = float(row["y"]); z = float(row["z"]); v = float(row["v"])
            snr = row["snr"]; noise = row["noise"]
            az = float(row["azimuth_deg"])
            if fnum not in radar_by_frame:
                radar_by_frame[fnum] = []
            radar_by_frame[fnum].append((x,y,z,v,snr,noise,az))
            if fnum not in seen:
                radar_frame_keys.append(fnum)
                radar_ts_us.append(tsus)
                seen.add(fnum)

    print("[4/6] Synchronizing frames ...")
    video_to_radar_idx = sync_frames(video_ts_us, radar_ts_us, max_gap_ms=80)

    print("[5/6] Collecting 2D↔3D correspondences ...")
    K = np.array([[args.fx, 0, args.cx],[0, args.fy, args.cy],[0,0,1]], dtype=np.float64)
    dist = np.array(args.dist, dtype=np.float64)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    all_pairs = []
    matches_debug = [["video_frame_idx","radar_frame_key","tag_id","u","v","x","y","z"]]

    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        ridx = video_to_radar_idx[fidx] if fidx < len(video_to_radar_idx) else -1
        if ridx >= 0:
            rfkey = radar_frame_keys[ridx]
            tag_centers = detect_aruco_centers(frame, args.aruco_dict)

            # if tag_centers:
            #     ids = [tid for tid,_,_ in tag_centers]
            #     # Optional: include x position to see ordering
            #     ids_u = [(tid, round(u,1)) for tid,u,_ in tag_centers]
            #     print(f"frame {fidx} ids(u): {ids_u}")

            if fidx % 30 == 0 and tag_centers:
                #print("frame", fidx, "ids:", [tid for tid,_,_ in tag_centers])
                print("frame", fidx, "ids(u):", [(tid, round(u,1)) for tid,u,_ in tag_centers])
            #print('tag_centers',len(tag_centers))
            radar_pts   = radar_by_frame.get(rfkey, [])
            # radar_pts = filter_by_azimuth(radar_by_frame.get(rfkey, []))
            # pairs = associate_left_to_right(tag_centers, radar_pts, min_snr=args.min_snr)

            # for (u, v, x, y, z) in pairs:
            #     all_pairs.append((u, v, x, y, z))
            #     matches_debug.append([fidx, rfkey, u, v, x, y, z])

            pairs = associate_by_id(tag_centers, radar_pts,
                                    valid_ids=args.tag_ids,
                                    min_snr=args.min_snr,expected_ranges=expected_ranges, expected_angles=expected_angles)

            # pairs = associate_by_ordered_range(tag_centers, radar_pts, min_snr=args.min_snr)
            # pairs = associate_hybrid_confident(tag_centers, radar_pts,
            #     valid_ids=args.tag_ids,fv
            #     min_snr=args.min_snr,
            #     expected_ranges=expected_ranges)

            # pairs = associate_by_id_unique(tag_centers, radar_pts,
            #                         valid_ids=args.tag_ids,
            #                         min_snr=args.min_snr,expected_ranges=expected_ranges,expected_angles=expected_angles)
            
            for (tid, u, v, x, y, z) in pairs:
                all_pairs.append((u, v, x, y, z))
                #matches_debug.append([fidx, rfkey, u, v, x, y, z])
                matches_debug.append([fidx, rfkey, tid, u, v, x, y, z])

                # # Group matched radar points per video frame
                # overlay_matches = {}
                # for row in matches_debug[1:]:  # skip header
                #     frame_idx = int(row[0])
                #     x, y, z = float(row[4]), float(row[5]), float(row[6])
                #     overlay_matches.setdefault(frame_idx, []).append((x, y, z))

        fidx += 1
    cap.release()

    if len(all_pairs) < 6:
        raise RuntimeError(f"Not enough matches collected: {len(all_pairs)}. Ensure clear ArUco tags and trihedral returns.")

    with open(args.out_dir / "matches_debug.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerows(matches_debug)

    print(f"    Solving extrinsics with {len(all_pairs)} pairs ...")
    R, rvec, tvec, stats = solve_extrinsics_pnp(all_pairs, K, dist)

    from math import atan2, asin, degrees
    yaw = degrees(atan2(R[1,0], R[0,0]))
    pitch = degrees(asin(-R[2,0]))
    roll = degrees(atan2(R[2,1], R[2,2]))
    report_lines = [
        "=== RADAR→CAMERA EXTRINSIC CALIBRATION REPORT ===",
        f"Pairs used: {stats['n_corr']}, Inliers: {stats['n_inliers']}",
        f"Reprojection error (px): mean={stats['err_mean_px']:.2f}  med={stats['err_median_px']:.2f}  p95={stats['err_p95_px']:.2f}",
        "",
        "Rotation R:",
        np.array2string(R, precision=6),
        "",
        f"rvec: {rvec}",
        f"tvec (meters): {tvec}",
        f"Euler (deg): Yaw={yaw:.2f}, Pitch={pitch:.2f}, Roll={roll:.2f}",
        ""
    ]
    with open(args.out_dir / "calib_report.txt", "w") as f:
        f.write("\n".join(report_lines))
    print("\n".join(report_lines))

    print("[overlay] Rendering overlay video ...")
    overlay_video(args.video, args.out_dir / "overlay.mp4", K, dist, rvec, tvec, radar_by_frame, video_to_radar_idx, radar_frame_keys, frameskip=args.frameskip)
    print("Done. Outputs in:", args.out_dir)

if __name__ == "__main__":
    main()
