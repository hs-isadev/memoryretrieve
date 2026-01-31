#!/usr/bin/env python3
#the previous comments are a load of bs
#THIS FILE WILL ALWAYS BE CALLED rpi.py
import sys
sys.path.insert(0, "/home/pi/deep_sort_realtime")

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
import datetime



# third-party deep sort
from deepsort_wrapper import DeepSort




# ===================== TRACKING STATE =====================
active_track_ids = set()
last_seen_ids = {}
frame_counter = 0
previous_bboxes = {}


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
    
    ack_sock.settimeout(0.01)
except Exception:
    ack_sock.settimeout(0.01)

PHONE_IP = None
state = {"running": False, "send_video": False}

# ===================== DEEPSORT TRACKER =====================
deepsort = DeepSort(
    max_age=6,
    n_init=2
)

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

def send_meta(payload):
    if PHONE_IP is None:
        return
    try:
        msg = json.dumps(payload)
        sock_image.sendto(msg.encode(), (PHONE_IP, ACK_PORT))

    except Exception as e:
        print("[META] send error:", e)


def send_log(event_type, payload):
    global PHONE_IP
    if PHONE_IP is None: return
    try:
        payload["color"]="RGB"
        msg = json.dumps({"event": event_type, "data": payload})
        sock_log.sendto(msg.encode(), (PHONE_IP, PHONE_PORT))
    except Exception as e:
        print("[LOG] send error:", e)



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

def send_image_reliable(img_bytes: bytes, fname: str, metadata: dict):
    """
    Sends image chunks using the EXACT protocol expected by Android:
    IMG|fname|seq|total|crc|len|<binary>
    """
    if not PHONE_IP:
        return

    CHUNK = 1300  # matches Android safe size
    total = (len(img_bytes) + CHUNK - 1) // CHUNK

    for seq in range(total):
        start = seq * CHUNK
        end = start + CHUNK
        chunk = img_bytes[start:end]

        crc = zlib.crc32(chunk) & 0xffffffff
        header = (
            f"IMG|{fname}|{seq}|{total}|{crc}|{len(chunk)}|"
        ).encode("utf-8")

        try:
            sock_image.sendto(
                header + chunk,
                (PHONE_IP, PHONE_IMAGE_PORT)
            )
        except Exception as e:
            print("[IMG] send error:", e)

        time.sleep(0.005)  # small pacing, prevents burst loss


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



# + helper to get confidence for a track
def get_confidence_for_tid(tid, sanitized_list, final_xywh, annotations):
    """
    Return the confidence associated with a given track id.
    """
    for det in final_xywh:
        x1, y1, w, h, conf, cls_id = det
        for ann in annotations:
            if ann["id"] == tid:
                bx1, by1, bx2, by2 = ann["bbox"]
                # + simple IOU check to match bbox
                iou_w = min(bx2, x1+w) - max(bx1, x1)
                iou_h = min(by2, y1+h) - max(by1, y1)
                if iou_w > 0 and iou_h > 0:
                    return conf
    return 0.0


# Debug time limiter (in seconds). Adjust as needed.
debug_interval_seconds = 1
last_debug_time = time.time()

# Keep track of previous bounding boxes for consistency checks
previous_bboxes = {}

def camera_loop():
    """
    Main camera loop:
    - Captures frames from Picamera2
    - Runs YOLO detections
    - Sanitizes detections for DeepSort
    - Updates DeepSort tracker
    - Sends appear/disappear logs and images to phone
    - Sends live video feed if enabled
    """
    global frame_counter, active_track_ids, last_seen_ids
    global previous_bboxes, last_debug_time
    global yolo_seen_once, sanitize_empty_warned, deepsort_empty_warned

    # ---- safety init ----
    if "yolo_seen_once" not in globals():
        yolo_seen_once = False
    if "sanitize_empty_warned" not in globals():
        sanitize_empty_warned = False
    if "deepsort_empty_warned" not in globals():
        deepsort_empty_warned = False

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

        # ---------------- CAPTURE FRAME ----------------
        try:
            frame = pic.capture_array()
        except Exception as e:
            print("[CAM] capture failed:", repr(e))
            time.sleep(0.02)
            continue
        if frame is None:
            time.sleep(0.02)
            continue

        # ---------------- SAVE ONE DEBUG FRAME ----------------
        if not one_debug_saved:
            try:
                cv2.imwrite("/home/pi/debug_frame.jpg", frame)
                print("[DEBUG] saved debug_frame.jpg")
            except Exception as e:
                print("[DEBUG] frame save failed:", e)
            one_debug_saved = True

        # ---------------- YOLO INFERENCE ----------------
        try:
            results = model(frame, imgsz=640, verbose=False)[0]

            raw_data = getattr(results.boxes, "data", [])
            raw_list = raw_data.tolist() if hasattr(raw_data, "tolist") else []
        except Exception as e:
            print("[YOLO] inference failed:", e)
            raw_list = []

        if not yolo_seen_once:
            print("[YOLO] model active, class names:", model_names)
            yolo_seen_once = True

        print(f"[YOLO] raw detections: {len(raw_list)}")

        # ---------------- FILTER BY CLASS / CONF ----------------
        dets_for_tracker = []
        for r in raw_list:
            if len(r) < 6:
                continue
            x1, y1, x2, y2, conf, cls_id = r
            if cls_id in TARGET_CLASS_IDS and conf >= CONF_TH:
                dets_for_tracker.append([x1, y1, x2, y2, conf, cls_id])

        if len(dets_for_tracker) == 0:
            print("[FILTER] 0 detections after class/conf filter")

        # ---------------- SANITIZE ----------------
        sanitized_dets = _ultimate_guard_sanitized(dets_for_tracker)

        if dets_for_tracker and not sanitized_dets and not sanitize_empty_warned:
            print("[SANITIZE] WARNING: sanitizer removed ALL detections")
            print("[SANITIZE] input:", dets_for_tracker)
            sanitize_empty_warned = True

        final_xywh = [[x1, y1, x2 - x1, y2 - y1, conf, cls_id] for x1, y1, x2, y2, conf, cls_id in sanitized_dets]

        # ---------------- PREPARE DEEPSORT INPUT ----------------
        ds_inputs = [
            (
                [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                float(conf),
                int(cls_id)   # pass integer class ID
            )
            for x1, y1, x2, y2, conf, cls_id in sanitized_dets
            if x2 - x1 > 0 and y2 - y1 > 0
        ]


        # ---------------- UPDATE DEEPSORT ----------------
        try:
            tracks = deepsort.update(ds_inputs, frame)

        except Exception as e:
            print("[DEEPSORT] update failed:", e)
            tracks = []

        print(f"[DEEPSORT] tracks returned: {len(tracks)}")

        if final_xywh and not tracks and not deepsort_empty_warned:
            print("[DEEPSORT] WARNING: detections sent but no tracks produced")
            print("[DEEPSORT] input:", final_xywh)
            deepsort_empty_warned = True

        current_ids = set()
        annotations_for_frame = []

        # ---------------- PROCESS TRACKS ----------------
        for tr in tracks:
            tid = tr["track_id"]
            x1, y1, x2, y2 = tr["bbox"]
            current_ids.add(tid)

            # ---------- INCONSISTENT BBOX DEBUG ----------
            if tid in previous_bboxes:
                px1, py1, px2, py2 = previous_bboxes[tid]
                dx = abs(x1 - px1)
                dy = abs(y1 - py1)
                dw = abs((x2 - x1) - (px2 - px1))
                dh = abs((y2 - y1) - (py2 - py1))
                if dx > 80 or dy > 80:
                    print(f"[BBOX-JUMP] id={tid} moved dx={dx} dy={dy}")
                if dw > 100 or dh > 100:
                    print(f"[BBOX-SIZE] id={tid} size jump dw={dw} dh={dh}")
                if x2 <= x1 or y2 <= y1:
                    print(f"[BBOX-INVALID] id={tid} bbox={x1,y1,x2,y2}")
            previous_bboxes[tid] = (x1, y1, x2, y2)

            # ---- match confidence via IoU ----
            best_area = 0
            conf_match = 0
            for dx1, dy1, dx2, dy2, dconf, _ in sanitized_dets:
                iw = max(0, min(dx2, x2) - max(dx1, x1))
                ih = max(0, min(dy2, y2) - max(dy1, y1))
                area = iw * ih
                if area > best_area:
                    best_area = area
                    conf_match = dconf

            annotations_for_frame.append({
                "id": tid,
                "bbox": [x1, y1, x2, y2],
                "classname": model_names.get(int(tr["class_id"]), "obj"),
                "confidence": float(conf_match)
            })

        # ---------------- DEBUG PRINT ----------------
        now = time.time()
        if now - last_debug_time > debug_interval_seconds:
            last_debug_time = now
            for ann in annotations_for_frame:
                print(f"[TRACK] id={ann['id']} "
                      f"bbox={ann['bbox']} "
                      f"conf={ann['confidence']:.2f}")

        # ---------------- APPEAR / DISAPPEAR ----------------
        appeared_ids = current_ids - active_track_ids
        disappeared_ids = {tid for tid in active_track_ids
                           if frame_counter - last_seen_ids.get(tid, 0) > DISAPPEAR_FRAMES}

        active_track_ids = (active_track_ids | current_ids) - disappeared_ids
        for tid in current_ids:
            last_seen_ids[tid] = frame_counter
        frame_counter += 1

        ts = time.time()

        # ---------------- APPEARED ----------------
        for tid in appeared_ids:
            ann = next((a for a in annotations_for_frame if a["id"] == tid), None)
            if ann is None:
                print(f"[WARN] appeared id {tid} but no annotation")
                continue
            send_log("appear", {
                "id": tid,
                "classname": ann["classname"],
                "bbox": ann["bbox"],
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            })

        # ---------------- DISAPPEARED ----------------
        for tid in disappeared_ids:
            send_log("disappear", {
                "id": tid,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            })

        # ---------------- HEARTBEAT ----------------
        if frame_counter % 30 == 0:
            print(f"[HEARTBEAT] frame={frame_counter} "
                  f"raw={len(raw_list)} "
                  f"filtered={len(dets_for_tracker)} "
                  f"sanitized={len(sanitized_dets)} "
                  f"tracks={len(tracks)}")

        # ---------------- LIVE VIDEO ----------------
        if state.get("send_video") and PHONE_IP:
            try:
                h, w = frame.shape[:2]
                side = min(h, w)
                y0 = (h - side) // 2
                x0 = (w - side) // 2
                
                square_crop = frame[y0:y0+side, x0:x0+side]
                square = cv2.resize(square_crop, (SQUARE_FEED_SIZE, SQUARE_FEED_SIZE))

                ok, jpg = cv2.imencode(
                    ".jpg", square,
                    [int(cv2.IMWRITE_JPEG_QUALITY), LIVE_JPEG_QUALITY]
                )
                if ok:
                    sock_video.sendto(jpg.tobytes(),
                                      (PHONE_IP, PHONE_VIDEO_PORT))
            except Exception as e:
                print("[VIDEO] send failed:", e)

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
