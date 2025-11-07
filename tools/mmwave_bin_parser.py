#!/usr/bin/env python3
import struct, csv, math
from pathlib import Path

MAGIC = b'\x02\x01\x04\x03\x06\x05\x08\x07'
HDR_FMT = "<8sIIIIIIII"                 # magic, version, totalLen, platform, frameNum, cpuTime, numDet, numTLVs, subFrame
HDR_SZ  = struct.calcsize(HDR_FMT)
TLV_HDR_FMT = "<II"
TLV_HDR_SZ  = struct.calcsize(TLV_HDR_FMT)

TLV_POINTS    = 1   # (x,y,z,v) float32
TLV_SIDE_INFO = 7   # (snr,noise) uint16

def iter_timestamped_chunks(bin_path: Path):
    with open(bin_path, "rb") as f:
        while True:
            ts = f.read(8)
            if len(ts) < 8: break
            chunk = f.read(1024)
            if not chunk: break
            yield struct.unpack("<Q", ts)[0], chunk

def parse(bin_path: Path, out_csv: Path, max_frames=None, debug_first_n=0):
    # Reassemble UART stream and keep a coarse timestamp map
    stream = bytearray()
    ts_at_offset = {}
    for ts, chunk in iter_timestamped_chunks(bin_path):
        ts_at_offset[len(stream)] = ts
        stream.extend(chunk)

    i = 0
    frames = rows = 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_us","frame_num","x","y","z","v","snr","noise",
                    "range_m","doppler_mps","azimuth_deg","power_snr"])

        while True:
            j = stream.find(MAGIC, i)
            if j < 0 or j + HDR_SZ > len(stream): break

            # frame header
            magic, version, total_len, platform, frame_num, cpu, num_det, num_tlvs, sub = \
                struct.unpack_from(HDR_FMT, stream, j)
            if total_len <= 0 or j + total_len > len(stream): break

            # timestamp for this frame (nearest chunk start at/before j)
            stamp_offsets = [off for off in ts_at_offset if off <= j]
            ts = ts_at_offset[max(stamp_offsets)] if stamp_offsets else 0

            # walk TLVs — IMPORTANT: tlv_len is PAYLOAD length, so advance by (8 + tlv_len)
            tlv_off = j + HDR_SZ
            points = []
            side   = []

            tlv_debug = []  # optional: record tlv types/lengths
            for _ in range(num_tlvs):
                if tlv_off + TLV_HDR_SZ > j + total_len: break
                tlv_type, tlv_len = struct.unpack_from(TLV_HDR_FMT, stream, tlv_off)
                payload_off = tlv_off + TLV_HDR_SZ
                payload_end = payload_off + tlv_len  # payload length only

                if payload_end > j + total_len: break
                payload = stream[payload_off:payload_end]
                tlv_debug.append((tlv_type, tlv_len))

                if tlv_type == TLV_POINTS:
                    # expect float32 x,y,z,v; be lenient if header num_det mismatches payload
                    n_pts = min(num_det, tlv_len // 16)
                    for k in range(n_pts):
                        x, y, z, v = struct.unpack_from("<ffff", payload, k*16)
                        points.append((x, y, z, v))

                elif tlv_type == TLV_SIDE_INFO:
                    n_info = min(num_det, tlv_len // 4)
                    for k in range(n_info):
                        snr, noise = struct.unpack_from("<HH", payload, k*4)
                        side.append((snr, noise))

                # advance to next TLV: header (8) + payload (tlv_len)
                tlv_off = payload_end

            # optional debug of first few frames
            if debug_first_n and frames < debug_first_n:
                print(f"Frame {frame_num}: num_det={num_det}, num_tlvs={num_tlvs}, TLVs={tlv_debug}")

            # write rows
            for idx, (x, y, z, v) in enumerate(points):
                snr, noise = ("","")
                if idx < len(side): snr, noise = side[idx]
                rng = math.sqrt(x*x + y*y + z*z)
                az  = math.degrees(math.atan2(y, x))
                w.writerow([ts, frame_num, x, y, z, v, snr, noise, rng, v, az, snr])
                rows += 1

            frames += 1
            i = j + total_len
            if max_frames and frames >= max_frames: break

    print(f"Done. Frames parsed: {frames}, Rows written: {rows}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.bin> <output.csv> [max_frames] [debug_first_n]")
        sys.exit(1)
    inp  = Path(sys.argv[1])
    outp = Path(sys.argv[2])
    maxf = int(sys.argv[3]) if len(sys.argv) > 3 else None
    dbg  = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    parse(inp, outp, max_frames=maxf, debug_first_n=dbg)
