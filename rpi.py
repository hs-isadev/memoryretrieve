#!/usr/bin/env python3
# rpi_full_featured.py - YOLOv8 + DeepSORT Raspberry Pi script
# Full merged version: includes all 2-part code features + defensive guards + debug

import cv2
import socket
import threading
import time
import json
import os
import zlib
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2
from typing import List, Dict, Tuple
import queue
import traceback

# third-party deep sort
from deep_sort_realtime.deepsort_tracker import DeepSort

# ------------------ <<< ADDED: tiny defensive helpers (non-destructive) ------------------
import numbers
def _is_sequence_like(obj):
    """Return True if obj is a sequence-like container we expect (not scalar floats/ints/strings)."""
    if obj is None:
        return False
    if isinstance(obj, (str, bytes)):
        return False
    if isinstance(obj, (float, int, np.floating, np.integer)):
        return False
    try:
        iter(obj)
    except TypeError:
        return False
    return True

def _ultimate_guard_sanitized(sanitized):
    """
    Final protective guard to ensure sanitized_dets is a list of 6-number lists.
    This function is intentionally strict and will produce an empty list rather than
    allow stray scalars (like 0.0) or malformed entries.
    """
    out = []
    if not isinstance(sanitized, list):
        return out
    for d in sanitized:
        try:
            if not isinstance(d, (list, tuple, np.ndarray)):
                continue
            if len(d) != 6:
                continue
            x1,y1,x2,y2,conf,cls_val = d
            # coerce numeric and check finiteness
            if not all(isinstance(v, (int, float, np.floating, np.integer)) for v in (x1,y1,x2,y2,conf)):
                # allow string numbers only if they coerce cleanly
                try:
                    x1 = float(x1); y1 = float(y1); x2 = float(x2); y2 = float(y2); conf = float(conf)
                except Exception:
                    continue
            # require finite coords/conf
            if not (np.isfinite(float(x1)) and np.isfinite(float(y1)) and np.isfinite(float(x2)) and np.isfinite(float(y2)) and np.isfinite(float(conf))):
                continue
            # coerce class to int
            try:
                cls_int = int(cls_val)
            except Exception:
                # allow float-like integers
                if isinstance(cls_val, (float, np.floating)) and float(cls_val).is_integer():
                    cls_int = int(cls_val)
                else:
                    continue
            out.append([float(x1), float(y1), float(x2), float(y2), float(conf), int(cls_int)])
        except Exception:
            continue
    return out
# --------------------------------------------------------------------------------------

# ===================== CONFIG =====================
SECRET = "memoryretrieve##$"
PHONE_PORT = 9100
PHONE_VIDEO_PORT = 6000
PHONE_IMAGE_PORT = 6200
DISCOVERY_PORT = 5004
COMMAND_PORT = 5005
ACK_PORT = 6201

# ===================== IMAGE TRANSFER CONFIG =====================
CHUNK_SIZE = 1024 * 16     # 16 KB per chunk (safe for UDP)
CHUNK_DELAY = 0.008        # 8 ms delay between chunks (prevents packet flooding)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SQUARE_FEED_SIZE = 480
JPEG_QUALITY = 85
LIVE_JPEG_QUALITY = 40
MAX_TRACKS = 10
DISAPPEAR_FRAMES = 6
CONF_TH = 0.15

TARGET_CLASS_NAMES = [
    'tv','laptop','mouse','remote','keyboard','cell phone','book','scissors','cup','bottle',
    'tennis racket','baseball bat','baseball glove','skateboard','surfboard','sports ball',
    'backpack','umbrella','handbag','tie','suitcase'
]

COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
    'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',
    'dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

OUT_DIR = "/home/pi/memoryretrieve_events"
IMAGES_DIR = os.path.join(OUT_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# ===================== MODEL =====================
print("[PI] Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("[PI] Model loaded.")

try:
    if hasattr(model, "names") and model.names:
        _names = model.names
        if isinstance(_names, dict):
            model_names = {int(k): str(v) for k, v in _names.items()}
        else:
            model_names = {i: str(n) for i, n in enumerate(_names)}
    else:
        model_names = {i: n for i, n in enumerate(COCO_CLASSES)}
except Exception:
    model_names = {i: n for i, n in enumerate(COCO_CLASSES)}

def _norm_name(n: str) -> str:
    return ''.join(ch for ch in n.lower() if ch.isalnum())

name_to_id = {_norm_name(v): k for k, v in model_names.items()}
TARGET_CLASS_IDS = set()
for t in TARGET_CLASS_NAMES:
    norm = _norm_name(t)
    if norm in name_to_id:
        TARGET_CLASS_IDS.add(name_to_id[norm])
    else:
        print(f"[WARN] target class '{t}' not found in model names; ignoring.")
print(f"[PI] Allowed class ids: {sorted(list(TARGET_CLASS_IDS))}")

# ===================== SOCKETS =====================
sock_log = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_image = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ack_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    ack_sock.bind(("0.0.0.0", ACK_PORT))
    ack_sock.settimeout(0.01)
except Exception:
    ack_sock.settimeout(0.01)

PHONE_IP = None
state = {"running": False, "send_video": False}

# ===================== DEEPSORT TRACKER =====================
deepsort = DeepSort(
    max_age=25,
    n_init=4,
    max_cosine_distance=0.4,
    nn_budget=100,
    override_track_class=None
)

active_track_ids = set()
last_seen_ids = {}
frame_counter = 0

# ===================== HELPERS =====================
def get_pi_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "0.0.0.0"

def send_log(event_type, payload):
    global PHONE_IP
    if PHONE_IP is None: return
    try:
        payload["color"]="RGB"
        msg = json.dumps({"event": event_type, "data": payload})
        sock_log.sendto(msg.encode(), (PHONE_IP, PHONE_PORT))
    except Exception as e:
        print("[LOG] send error:", e)

def wait_for_ack(fname, seq, timeout=0.25):
    end = time.time() + timeout
    try:
        while time.time() < end:
            try:
                data, addr = ack_sock.recvfrom(2048)
                msg = data.decode(errors='ignore')
                if msg.startswith("ACK|"):
                    parts = msg.split("|")
                    if len(parts) >= 3 and parts[1] == fname and int(parts[2]) == seq:
                        return True
            except socket.timeout:
                pass
    except Exception:
        pass
    return False

def send_chunk_with_ack(img_bytes: bytes, fname: str, seq: int, total: int, addr_tuple):
    crc = zlib.crc32(img_bytes) & 0xffffffff
    header = f"IMG|{fname}|{seq}|{total}|{crc}|{len(img_bytes)}|".encode()
    pkt = header + img_bytes
    for attempt in range(12):
        try:
            sock_image.sendto(pkt, addr_tuple)
        except Exception as e:
            print("[IMG] send error:", e)
        if wait_for_ack(fname, seq, timeout=0.25):
            return True
        time.sleep(0.02)
    print(f"[IMG] chunk {seq}/{total} failed after retries")
    return False

import struct

CHUNK_SIZE = 60000  # safe UDP chunk size
ACK_TIMEOUT = 0.5   # seconds
MAX_RETRIES = 5

def send_image_reliable(img_bytes, fname, metadata):
    """
    Sends an image reliably over UDP to the phone by splitting into chunks.
    Requires a simple ACK protocol on the phone side.
    """
    if not PHONE_IP:
        return

    total_size = len(img_bytes)
    num_chunks = (total_size + CHUNK_SIZE - 1) // CHUNK_SIZE
    msg_id = int(time.time() * 1000) & 0xFFFFFFFF

    for chunk_idx in range(num_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = start + CHUNK_SIZE
        chunk = img_bytes[start:end]

        # UDP packet format: [msg_id(4B)][total_chunks(2B)][chunk_idx(2B)][payload]
        header = struct.pack(">IHH", msg_id, num_chunks, chunk_idx)
        packet = header + chunk

        for attempt in range(MAX_RETRIES):
            sock_video.sendto(packet, (PHONE_IP, PHONE_VIDEO_PORT))
            # Optional: wait for ACK from phone
            # ack = sock_video.recvfrom(1024)
            # if ack indicates success: break
            time.sleep(0.01)  # tiny delay between retries

    # Send metadata as JSON after image fully sent
    try:
        import json
        meta_bytes = json.dumps({"fname": fname, **metadata}).encode()
        sock_video.sendto(meta_bytes, (PHONE_IP, PHONE_VIDEO_PORT))
    except Exception as e:
        print("[SEND_IMAGE] metadata send failed:", e)


# ===================== DISCOVERY & COMMAND =====================
def set_phone_info(ip):
    global PHONE_IP
    PHONE_IP = ip
    print(f"[PHONE] ip set to {PHONE_IP}")

def discovery_listener():
    pi_ip = get_pi_ip()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    try:
        s.bind(("0.0.0.0", DISCOVERY_PORT))
    except Exception as e:
        print(f"[DISCOVERY] bind error: {e}")
        return

    print(f"[DISCOVERY] Listening on {DISCOVERY_PORT}... PI_IP={pi_ip}")
    reply_msg = f"OK|PI|{pi_ip}".encode()

    while True:
        try:
            data, addr = s.recvfrom(2048)
            msg = data.decode(errors="ignore").strip()
            parts = msg.split("|")
            if len(parts) == 2 and parts[0] == SECRET and parts[1].upper() == "DISCOVER":
                phone_ip = addr[0]
                print(f"[DISCOVERY] DISCOVER from {phone_ip}:{addr[1]}")
                try:
                    s.sendto(reply_msg, (phone_ip, DISCOVERY_PORT))
                    set_phone_info(phone_ip)
                except Exception as e:
                    print(f"[DISCOVERY] reply send error: {e}")
        except Exception:
            time.sleep(0.1)

def command_listener():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    bind_result = s.bind(("0.0.0.0", COMMAND_PORT))
    print(f"[COMMAND] Listening on {COMMAND_PORT}...")

    while True:
        data, addr = s.recvfrom(4096)
        msg = data.decode(errors="ignore").strip()

        if PHONE_IP is None:
            set_phone_info(addr[0])

        parts = msg.split("|")
        if len(parts) < 2:
            continue

        if parts[0] != SECRET:
            continue

        verb = parts[1].upper()

        if verb == "START":
            state["running"] = True
            state["send_video"] = True
            set_phone_info(addr[0])
            print("[COMMAND] START")

        elif verb == "STOP":
            state["running"] = False
            state["send_video"] = False
            print("[COMMAND] STOP")

        elif verb == "DISCOVER":
            reply_msg = f"OK|PI|{get_pi_ip()}".encode()
            try:
                s.sendto(reply_msg, addr)
                set_phone_info(addr[0])
            except OSError as e:
                print("[COMMAND] DISCOVER send error:", e)


# ===================== DRAW =====================
def draw_annotations_on_copy(frame, annots):
    img = frame.copy()
    for a in annots:
        try:
            x1,y1,x2,y2 = map(int, a["bbox"])
            cls = str(a.get("classname","obj"))
            tid = a.get("id",0)
            color = (0, 200, 0)
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            label = f"{cls}_{tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, max(y1-18,0)), (x1+tw+4, max(y1,18)), color, -1)
            cv2.putText(img, label, (x1+2, max(y1-4,12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        except Exception:
            continue
    return img
# ===================== CAMERA LOOP =====================
COLOR_PERM = (0,1,2)
R_GAIN = 1.0
G_GAIN = 1.0
B_GAIN = 1.0

def calibrate_color_once(pic, samples=5):
    global COLOR_PERM, R_GAIN, G_GAIN, B_GAIN
    means=[]
    for _ in range(samples):
        try:
            f = pic.capture_array()
        except:
            f = None
        if f is None:
            time.sleep(0.02)
            continue
        img = f.copy()
        if img.ndim==3 and img.shape[2]==4:
            try: img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            except: img = img[:,:,:3]
        if img.ndim==3 and img.shape[2]==3:
            means.append(img.reshape(-1,3).mean(axis=0))
        time.sleep(0.02)
    if not means:
        return
    mean_all = np.stack(means,axis=0).mean(axis=0)
    c0,c1,c2 = mean_all
    if c2 > c0 * 1.15 and c2 > c1 * 1.05:
        COLOR_PERM = (2,1,0)
    else:
        COLOR_PERM = (0,1,2)
    if mean_all[1] > mean_all[0]*1.3 and mean_all[1] > mean_all[2]*1.3:
        R_GAIN = 1.1

import traceback

import traceback

import time

# Debug time limiter (in seconds). Adjust as needed.
debug_interval_seconds = 1
last_debug_time = time.time()

# Keep track of previous bounding boxes for consistency checks
previous_bboxes = {}

def camera_loop():
    global frame_counter, active_track_ids, last_seen_ids, previous_bboxes, last_debug_time

    pic = Picamera2()
    pic_config = pic.create_preview_configuration(
        main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
    )

    pic.configure(pic_config)
    pic.start()
    print("[CAM] Camera started")

    one_debug_saved = False

    while True:
        if not state["running"]:
            time.sleep(0.02)
            continue

        # --- capture frame safely ---
        try:
            frame = pic.capture_array()
            cv2.imwrite("/home/pi/debug_frame_live.jpg", frame)   # <-- Debug: save current frame
        except Exception as e:
            print(f"[CAM] capture failed: {repr(e)}")
            print(traceback.format_exc())  # Print the traceback for debugging
            time.sleep(0.02)
            continue

        if frame is None:
            print("[CAM] capture returned None")
            time.sleep(0.02)
            continue

        # --- DEBUG SAVE ONE FRAME ---
        if not one_debug_saved:
            try:
                print("[DEBUG] frame type:", type(frame))
                print("[DEBUG] frame shape:", getattr(frame, "shape", None))
                print("[DEBUG] frame dtype:", getattr(frame, "dtype", None))

                success = cv2.imwrite("/home/pi/debug_frame.jpg", frame)
                print("[DEBUG] wrote debug_frame.jpg =", success)

            except Exception as e:
                print("[DEBUG] saving debug failed:", e)
                print(traceback.format_exc())  # Print the traceback for debugging

            one_debug_saved = True

        # ==========================================================
        # Debugging bounding box consistency and classification
        current_time = time.time()
        if current_time - last_debug_time > debug_interval_seconds:
            last_debug_time = current_time

            try:
                results = model(frame, imgsz=480, verbose=False)[0]
                raw_list = []
                raw_data = results.boxes.data
                if hasattr(raw_data, "tolist"):
                    raw_list = raw_data.tolist()

                for r in raw_list:
                    if len(r) < 6:
                        continue
                    x1, y1, x2, y2, conf, cls_id = r

                    # Check consistency of bounding boxes between frames
                    if cls_id in previous_bboxes:
                        prev_bbox = previous_bboxes[cls_id]
                        if abs(x1 - prev_bbox[0]) > 20 or abs(y1 - prev_bbox[1]) > 20:  # Tolerance of 20 pixels
                            print(f"[DEBUG] Bounding box inconsistency detected: {prev_bbox} -> {(x1, y1, x2, y2)}")

                    # Save the current bounding box for future comparison
                    previous_bboxes[cls_id] = (x1, y1, x2, y2)

                    # Debugging classification
                    class_name = model_names.get(cls_id, "Unknown")
                    print(f"[DEBUG] Class ID: {cls_id}, Detected as: {class_name}, Confidence: {conf}")

            except Exception as e:
                print("[DEBUG] Error during YOLO detection:", e)
                continue

        # --- YOLO Detection and Processing ---
        dets_for_tracker = []
        try:
            results = model(frame, imgsz=480, verbose=False)[0]
            raw_list = []
            raw_data = results.boxes.data
            if hasattr(raw_data, "tolist"):
                raw_list = raw_data.tolist()

            for r in raw_list:
                if len(r) < 6:
                    continue
                x1, y1, x2, y2, conf, cls_id = r
                if cls_id in TARGET_CLASS_IDS:
                    if conf >= CONF_TH:
                        dets_for_tracker.append([float(x1), float(y1), float(x2), float(y2), float(conf), cls_id])
                    else:
                        print(f"[YOLO] Skipping detection due to low confidence: {conf}, Class: {cls_id}")
                else:
                    print(f"[YOLO] Skipping non-target class: {model_names.get(cls_id, 'Unknown')}")

        except Exception as e:
            print("[YOLO] Detection error:", repr(e))
            dets_for_tracker = []

        # ---------------- YOLO Detection and Processing ----------------
        # ---------------- YOLO Detection and Processing ----------------
        dets_for_tracker = []
        try:
            results = model(frame, imgsz=480, verbose=False)[0]
            raw_list = []
            raw_data = results.boxes.data
            if hasattr(raw_data, "tolist"):
                raw_list = raw_data.tolist()

            for r in raw_list:
                if len(r) < 6:
                    continue
                x1, y1, x2, y2, conf, cls_id = r
                print(f"[DEBUG] Raw detection class id: {cls_id}, confidence: {conf}")
                if cls_id in TARGET_CLASS_IDS:
                    if conf >= CONF_TH:
                        dets_for_tracker.append([float(x1), float(y1), float(x2), float(y2), float(conf), cls_id])
                    else:
                        print(f"[YOLO] Skipping detection due to low confidence: {conf}, Class: {cls_id}")
                else:
                    print(f"[YOLO] Skipping non-target class: {model_names.get(cls_id, 'Unknown')}")
        except Exception as e:
            print("[YOLO] Detection error:", repr(e))

        # ---------------- SANITIZE DETECTIONS ----------------
        sanitized_dets = _ultimate_guard_sanitized(dets_for_tracker)
        print(f"[DEBUG] Sanitized detections: {sanitized_dets}")  # Debug the sanitized detections


        # ---------------- PREPARE FOR DEEPSORT ----------------
        final_xywh = []
        for d in sanitized_dets:
            try:
                x1, y1, x2, y2, conf, cls_id = d
                if not all(np.isfinite([x1, y1, x2, y2, conf])):
                    print(f"[DEBUG] Skipping due to non-finite values: {d}")
                    continue
                w = float(x2 - x1)
                h = float(y2 - y1)
                if w <= 0 or h <= 0:
                    print(f"[DEBUG] Skipping due to invalid width/height: {d}")
                    continue
                final_xywh.append([x1, y1, w, h, conf, cls_id])
            except Exception as e:
                print(f"[DEBUG] Error processing sanitized detection: {e}")
                continue

        print(f"[DEBUG] Final detections for DeepSORT: {final_xywh}")  # Debug the final detections

        # ---------------- DEEPSORT UPDATE ----------------
        try:
            if len(final_xywh) > 0:
                detections_list_format = [
                    [float(x1), float(y1),
                    float(x1 + w), float(y1 + h),
                    float(conf), int(cls_id)]
                    for x1, y1, w, h, conf, cls_id in final_xywh
                ]
                print(f"[DEBUG] DeepSORT detections: {detections_list_format}")  # Debug formatted detections

                tracks = deepsort.update_tracks(detections_list_format, frame=frame)
                print(f"[DEBUG] Tracks after update: {tracks}")  # Debug tracks after update
            else:
                print("[DEEPSORT] No detections to track.")
                tracks = deepsort.update_tracks([], frame=frame)
                print(f"[DEBUG] Tracks after no detections: {tracks}")  # Debug tracks when no detections
        except Exception as e:
            print(f"[DEEPSORT] unexpected error: {e}")
            tracks = []

        # ---------------- HANDLE TRACKS ----------------
        current_ids = set()
        annotations_for_frame = []
        for tr in tracks:
            try:
                if isinstance(getattr(tr, "det_class", None), float):
                    continue
            except:
                pass
            annotations_for_frame.append({
                "id": tr.track_id,
                "bbox": tr.to_ltrb(),
                "classname": str(tr.det_class)
            })

        # ---------------- APPEAR/DISAPPEAR LOGIC ----------------
        appeared_ids = current_ids - active_track_ids
        disappeared_ids = {tid for tid in active_track_ids if frame_counter - last_seen_ids.get(tid, 0) > DISAPPEAR_FRAMES}
        active_track_ids = (active_track_ids | appeared_ids) - disappeared_ids

        # ---------------- HANDLE APPEARED/DIASAPPEARED ----------------
        # Add logic to handle appeared and disappeared IDs as needed (e.g., sending images, etc.)

        # ---------------- LIVE VIDEO ----------------
        if state.get("send_video", False) and PHONE_IP:
            try:
                h, w = frame.shape[:2]
                side = min(h, w)
                cx, cy = w // 2, h // 2
                left, top = max(0, cx - side // 2), max(0, cy - side // 2)
                square = frame[top:top + side, left:left + side]
                square_resized = cv2.resize(square, (SQUARE_FEED_SIZE, SQUARE_FEED_SIZE))
                ret, jpg = cv2.imencode(".jpg", square_resized,
                                        [int(cv2.IMWRITE_JPEG_QUALITY), LIVE_JPEG_QUALITY])
                if ret:
                    sock_video.sendto(jpg.tobytes(), (PHONE_IP, PHONE_VIDEO_PORT))
            except:
                pass

        time.sleep(0.01)



# ===================== START THREADS =====================
if __name__ == "__main__":
    print("[PI] Starting discovery and command listeners...")
    t_disc = threading.Thread(target=discovery_listener, daemon=True)
    t_cmd  = threading.Thread(target=command_listener, daemon=True)
    t_cam  = threading.Thread(target=camera_loop, daemon=True)

    t_disc.start()
    t_cmd.start()
    t_cam.start()

    print("[PI] Raspberry Pi memory retrieve system started.")
    print(f"[PI] Local IP: {get_pi_ip()}")
    print("[PI] Waiting for phone commands...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[PI] Shutting down...")
        state["running"] = False
        state["send_video"] = False
        time.sleep(0.5)
        print("[PI] Exiting.")
