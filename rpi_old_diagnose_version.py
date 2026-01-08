#!/usr/bin/env python3
"""
rpi_auto_yolo_tracker.py

Features implemented per request:
- UDP auto-discovery / control (SECRET|DISCOVER replies)
- YOLOv8 inference in a separate process (robust to hangs)
- Lightweight tracker (stable IDs, appears/disappears) — behaves like DeepSORT for appeared/disappeared events
- Sends raw UDP video feed (JPEG frames) to phone while connected
- Sends only Appeared / Disappeared events to phone as JSON; when Appeared, attaches a gallery-openable JPEG image (saved to disk and chunked-sent over UDP)
- Robust non-blocking queues, watchdogs and diagnostics to avoid freezes

Drop this single file onto your Pi (replace your main). It is fully paste-ready.
"""

import os
import time
import socket
import json
import threading
import traceback
import zlib
import queue
import multiprocessing as mp
from typing import List, Dict, Tuple

import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# ============== CONFIG ==============
SECRET = "memoryretrieve##$"
PHONE_PORT = 9100
PHONE_VIDEO_PORT = 6000
PHONE_IMAGE_PORT = 6200
CONTROL_PORT = 5004
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 85
MAX_TRACKS = 20
DISAPPEAR_FRAMES = 3
CONF_TH = 0.40
CHUNK_SIZE = 4096
CHUNK_DELAY = 0.0008
FORCE_RGB = True  # camera gives RGB; convert where needed
OUT_DIR = "/home/pi/memoryretrieve_events"
IMAGES_DIR = os.path.join(OUT_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

TARGET_CLASS_NAMES = [
    'tv','laptop','mouse','remote','keyboard','cell phone','book','scissors','cup','bottle',
    'tennis racket','baseball bat','baseball glove','skateboard','surfboard','sports ball',
    'backpack','umbrella','handbag','tie','suitcase'
]

# Inference/process tuning
INF_QUEUE_MAX = 4
OUT_QUEUE_MAX = 8
INFERENCE_MAX_AGE_S = 10.0

# Minimal COCO fallback
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
]

# ============== sockets ==============
sock_log = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_image = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
for s in (sock_log, sock_video, sock_image):
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock_video.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
    sock_image.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
except Exception:
    pass

PHONE_IP = None
state = {"running": False, "send_video": False}
last_heartbeat = time.time()
phone_connected = False

# ============== tracker (lightweight) ==============
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    inter = interW * interH
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    union = boxAArea + boxBArea - inter if (boxAArea+boxBArea-inter) > 0 else 1
    return inter / union if union > 0 else 0.0

class SimpleTracker:
    """Assigns small stable integer IDs, reports appeared/disappeared events only."""
    def __init__(self, max_tracks=MAX_TRACKS, disappear_thresh=DISAPPEAR_FRAMES, iou_threshold=0.3):
        self.max_tracks = max_tracks
        self.disappear_thresh = disappear_thresh
        self.iou_threshold = iou_threshold
        self.tracks = {}  # id -> {bbox, classname, conf, missed, last_seen}
        self.frame_idx = 0

    def _new_id(self):
        for i in range(1, self.max_tracks + 1):
            if i not in self.tracks:
                return i
        return None

    def update(self, detections: List[Dict]) -> List[Dict]:
        """Return list of events: {id, bbox, classname, conf, appeared|disappeared}"""
        self.frame_idx += 1
        events = []
        if len(self.tracks) == 0:
            for det in detections[:self.max_tracks]:
                nid = self._new_id()
                if nid is None: break
                self.tracks[nid] = {"bbox": det["bbox"], "classname": det.get("classname","unknown"), "conf": det.get("conf",0.0), "missed":0, "last_seen":self.frame_idx}
                events.append({"id":nid, "bbox":det["bbox"], "classname":det.get("classname","unknown"), "conf":det.get("conf",0.0), "appeared":True})
            return events
        # compute IoU matrix
        track_ids = list(self.tracks.keys())
        assigned_tracks = set(); assigned_dets = set()
        iou_mat = [[iou(self.tracks[tid]["bbox"], det["bbox"]) for det in detections] for tid in track_ids]
        # greedy match
        while True:
            best = 0; bt=-1; bd=-1
            for ti,row in enumerate(iou_mat):
                for di,val in enumerate(row):
                    if val>best and val>=self.iou_threshold and track_ids[ti] not in assigned_tracks and di not in assigned_dets:
                        best=val; bt=ti; bd=di
            if bt==-1: break
            tid=track_ids[bt]; det=detections[bd]
            self.tracks[tid]["bbox"] = det["bbox"]
            self.tracks[tid]["classname"] = det.get("classname", self.tracks[tid].get("classname","unknown"))
            self.tracks[tid]["conf"] = det.get("conf", 0.0)
            self.tracks[tid]["missed"] = 0
            self.tracks[tid]["last_seen"] = self.frame_idx
            assigned_tracks.add(tid); assigned_dets.add(bd)
        # unassigned detections -> new tracks
        for di,det in enumerate(detections):
            if di in assigned_dets: continue
            nid = self._new_id()
            if nid is None: continue
            self.tracks[nid] = {"bbox":det["bbox"], "classname":det.get("classname","unknown"), "conf":det.get("conf",0.0), "missed":0, "last_seen":self.frame_idx}
            assigned_tracks.add(nid)
            events.append({"id":nid, "bbox":det["bbox"], "classname":det.get("classname","unknown"), "conf":det.get("conf",0.0), "appeared":True})
        # increment missed for unassigned tracks
        for tid in list(self.tracks.keys()):
            if tid in assigned_tracks: continue
            self.tracks[tid]["missed"] += 1
            if self.tracks[tid]["missed"] > self.disappear_thresh:
                events.append({"id":tid, "bbox":self.tracks[tid]["bbox"], "classname":self.tracks[tid].get("classname","unknown"), "conf":self.tracks[tid].get("conf",0.0), "disappeared":True})
                del self.tracks[tid]
        return events

tracker = SimpleTracker()

# ============== helpers ==============
def get_pi_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "0.0.0.0"

def send_log(event_type: str, payload: dict):
    global PHONE_IP
    if PHONE_IP is None:
        return
    try:
        payload.setdefault("classname", payload.get("label"))
        payload.setdefault("label", payload.get("classname"))
        payload["color"] = "RGB"
        msg = json.dumps({"event": event_type, "data": payload})
        try:
            sock_log.sendto(msg.encode(), (PHONE_IP, PHONE_PORT))
        except Exception:
            try:
                sock_log.sendto(msg.encode(), (PHONE_IP, PHONE_PORT))
            except Exception:
                pass
    except Exception as e:
        print("[LOG] send error:", e)

def save_image_file(img_bytes: bytes, fname: str) -> str:
    path = os.path.join(IMAGES_DIR, fname)
    with open(path, "wb") as f:
        f.write(img_bytes)
    return path

def set_phone_info(ip: str):
    global PHONE_IP, phone_connected, last_heartbeat
    PHONE_IP = ip
    phone_connected = True
    last_heartbeat = time.time()
    state["running"] = True
    state["send_video"] = True
    print(f"[PHONE] set ip={PHONE_IP}")

# ============== image sender (chunked) ==============
image_send_queue = queue.Queue(maxsize=64)

def image_sender_thread():
    while True:
        try:
            item = image_send_queue.get()
            if item is None:
                break
            img_bytes, fname = item
            total = len(img_bytes)
            num = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
            announce = {"filename": fname, "chunks": num, "size": total, "color":"RGB"}
            send_log("image_transfer", announce)
            for seq in range(num):
                start = seq * CHUNK_SIZE
                end = min(start + CHUNK_SIZE, total)
                chunk = img_bytes[start:end]
                crc = zlib.crc32(chunk) & 0xffffffff
                header = f"IMG|{fname}|{seq}|{num}|{crc}|{len(chunk)}|".encode()
                try:
                    if PHONE_IP:
                        sock_image.sendto(header + chunk, (PHONE_IP, PHONE_IMAGE_PORT))
                except Exception as e:
                    print("[IMG] chunk send error, continuing:", e)
                time.sleep(CHUNK_DELAY)
        except Exception as e:
            print("[IMG-SENDER] thread error:", e)
            try: traceback.print_exc()
            except: pass
            time.sleep(0.05)

t_image_sender = threading.Thread(target=image_sender_thread, daemon=True)
t_image_sender.start()

def queue_image_for_send(img_bytes: bytes, fname: str):
    try:
        image_send_queue.put_nowait((img_bytes, fname))
    except queue.Full:
        print("[IMG] send queue full, dropping image", fname)

# ============== control listener ==============
def control_listener():
    global PHONE_IP, state, last_heartbeat, phone_connected
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    try:
        s.bind(("0.0.0.0", CONTROL_PORT))
    except Exception as e:
        print(f"[CONTROL] bind error: {e}")
        return
    print(f"[CONTROL] Listening on {CONTROL_PORT} for DISCOVER/COMMANDS...")
    pi_ip = get_pi_ip()
    while True:
        try:
            data, addr = s.recvfrom(4096)
            msg = data.decode(errors="ignore").strip()
            print(f"[CONTROL] recv from {addr[0]}:{addr[1]} -> '{msg[:200]}'")
            if msg == f"{SECRET}|DISCOVER" or msg.startswith(f"{SECRET}|DD|DISCOVER") or (("DISCOVER" in msg) and (SECRET.split('|')[0] in msg)):
                reply_msg = f"DISCOVER_REPLY|{pi_ip}".encode()
                try:
                    s.sendto(reply_msg, addr)
                    print(f"[DISCOVERY] Reply sent to {addr[0]}:{addr[1]} -> {reply_msg.decode()}")
                except Exception as e:
                    print("[DISCOVERY] reply-to-src err:", e)
                try:
                    s.sendto(reply_msg, (addr[0], CONTROL_PORT))
                except Exception:
                    pass
                set_phone_info(addr[0])
                continue
            parts = msg.split("|")
            if len(parts) >= 2 and parts[0] == SECRET:
                verb = parts[1]
                if verb == "START":
                    state["running"] = True
                    state["send_video"] = True
                    set_phone_info(addr[0])
                    print("[CONTROL] START OK")
                elif verb == "STOP":
                    state["running"] = False
                    state["send_video"] = False
                    print("[CONTROL] STOP OK")
                elif verb == "HB":
                    last_heartbeat = time.time()
                    phone_connected = True
                else:
                    print("[CONTROL] Unknown verb:", verb)
        except Exception as e:
            print("[CONTROL] recv error:", e)
            time.sleep(0.05)

# ============== inference process ==============
# process will load a YOLO model and process frames from multiprocessing.Queue

def serialize_frame(frame: np.ndarray):
    return {"shape": frame.shape, "dtype": str(frame.dtype), "bytes": frame.tobytes()}

def deserialize_frame(obj):
    arr = np.frombuffer(obj["bytes"], dtype=np.dtype(obj["dtype"]))
    return arr.reshape(obj["shape"])

def _norm_name(n: str) -> str:
    return ''.join(ch for ch in n.lower() if ch.isalnum())

def inference_process_main(in_q: mp.Queue, out_q: mp.Queue):
    try:
        print("[INFER_PROC] starting, loading model...")
        model = YOLO("yolov8n.pt")
        if hasattr(model, "names") and model.names:
            _names = model.names
            if isinstance(_names, dict):
                model_names = {int(k): str(v) for k, v in _names.items()}
            else:
                model_names = {i: str(n) for i, n in enumerate(_names)}
        else:
            model_names = {i: n for i, n in enumerate(COCO_CLASSES)}
        name_to_id = {_norm_name(v): k for k, v in model_names.items()}
        TARGET_CLASS_IDS_LOCAL = set()
        for t in TARGET_CLASS_NAMES:
            norm = _norm_name(t)
            if norm in name_to_id:
                TARGET_CLASS_IDS_LOCAL.add(name_to_id[norm])
        if not TARGET_CLASS_IDS_LOCAL:
            TARGET_CLASS_IDS_LOCAL = set(model_names.keys())
        print("[INFER_PROC] model loaded, allowed classes:", sorted(list(TARGET_CLASS_IDS_LOCAL)))
    except Exception as e:
        print("[INFER_PROC] model failed to load:", e)
        return

    while True:
        try:
            item = None
            try:
                item = in_q.get(timeout=1.0)
            except Exception:
                continue
            if item is None:
                break
            frame_ts = item.get("ts")
            frame_obj = item.get("frame")
            bgr_obj = item.get("bgr")
            try:
                rgb_frame = deserialize_frame(frame_obj)
            except Exception as e:
                print("[INFER_PROC] failed to deserialize frame:", e)
                continue
            try:
                results = model.predict(rgb_frame, imgsz=480, verbose=False)[0]
                dets = []
                for box in getattr(results, "boxes", []):
                    try:
                        cls = int(box.cls[0]); conf = float(box.conf[0])
                    except Exception:
                        continue
                    if conf < CONF_TH: continue
                    if cls not in TARGET_CLASS_IDS_LOCAL: continue
                    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                    x1=max(0,x1); y1=max(0,y1); x2=max(0,x2); y2=max(0,y2)
                    classname = model_names.get(cls, str(cls))
                    dets.append({"bbox": [x1,y1,x2,y2], "conf": conf, "classname": classname})
                annotated_bytes = None
                if dets and bgr_obj is not None:
                    try:
                        bgr_frame = deserialize_frame(bgr_obj)
                        ann = bgr_frame.copy()
                        for d in dets:
                            x1,y1,x2,y2 = d["bbox"]
                            cv2.rectangle(ann, (x1,y1),(x2,y2),(0,255,0),2)
                            text = f"{d['classname']} {int(d['conf']*100)}%"
                            cv2.putText(ann, text, (max(2,x1), max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)
                        ret,j = cv2.imencode('.jpg', ann, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                        if ret:
                            annotated_bytes = j.tobytes()
                    except Exception as e:
                        print("[INFER_PROC] annotate err:", e)
                try:
                    out_q.put_nowait((frame_ts, dets, annotated_bytes))
                except Exception:
                    pass
            except Exception as e:
                print("[INFER_PROC] model.predict err:", e)
                try: traceback.print_exc()
                except: pass
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("[INFER_PROC] loop err:", e)
            try: traceback.print_exc()
            except: pass
    print("[INFER_PROC] exiting cleanly.")

# ============== Inference manager ==============
class InferenceManager:
    def __init__(self):
        self.in_q = mp.Queue(maxsize=INF_QUEUE_MAX)
        self.out_q = mp.Queue(maxsize=OUT_QUEUE_MAX)
        self.proc = None
        self.last_result_time = time.time()
        self.lock = threading.Lock()
        self.start_process()

    def start_process(self):
        with self.lock:
            if self.proc is not None and self.proc.is_alive():
                return
            try:
                try:
                    self._close_queues()
                except Exception:
                    pass
                self.in_q = mp.Queue(maxsize=INF_QUEUE_MAX)
                self.out_q = mp.Queue(maxsize=OUT_QUEUE_MAX)
                self.proc = mp.Process(target=inference_process_main, args=(self.in_q, self.out_q), daemon=True)
                self.proc.start()
                self.last_result_time = time.time()
                print(f"[INFERENCE-MGR] Started inference process pid={self.proc.pid}")
            except Exception as e:
                print("[INFERENCE-MGR] failed to start process:", e)

    def _close_queues(self):
        try:
            if self.in_q is not None:
                while not self.in_q.empty():
                    try: self.in_q.get_nowait()
                    except: break
                self.in_q.close()
        except Exception: pass
        try:
            if self.out_q is not None:
                while not self.out_q.empty():
                    try: self.out_q.get_nowait()
                    except: break
                self.out_q.close()
        except Exception: pass

    def stop_process(self):
        with self.lock:
            try:
                if self.in_q is not None:
                    try: self.in_q.put_nowait(None)
                    except Exception: pass
            except Exception:
                pass
            try:
                if self.proc is not None and self.proc.is_alive():
                    self.proc.terminate()
                    self.proc.join(timeout=2.0)
                print("[INFERENCE-MGR] stopped process")
            except Exception as e:
                print("[INFERENCE-MGR] stop err:", e)
            finally:
                self.proc = None
                self._close_queues()

    def enqueue_frame(self, ts_ms, rgb_frame, bgr_frame=None):
        item = {"ts": ts_ms, "frame": serialize_frame(rgb_frame)}
        if bgr_frame is not None:
            item["bgr"] = serialize_frame(bgr_frame)
        try:
            self.in_q.put_nowait(item)
            return True
        except Exception:
            return False

    def try_get_result(self):
        try:
            res = self.out_q.get_nowait()
            self.last_result_time = time.time()
            return res
        except Exception:
            return None

    def check_and_restart_if_needed(self):
        with self.lock:
            if self.proc is None or not self.proc.is_alive():
                print("[INFERENCE-MGR] process dead, restarting...")
                self.start_process()
                return
            age = time.time() - self.last_result_time
            if age > INFERENCE_MAX_AGE_S:
                print(f"[INFERENCE-MGR] no results for {age:.1f}s -> restarting inference process")
                try:
                    self.proc.terminate(); self.proc.join(timeout=2.0)
                except Exception:
                    pass
                self.proc = None
                self.start_process()

inference_mgr = InferenceManager()

# monitor
def inference_monitor_thread():
    while True:
        try:
            inference_mgr.check_and_restart_if_needed()
            time.sleep(2.0)
        except Exception as e:
            print("[INFER-MON] err:", e)
            time.sleep(1.0)

threading.Thread(target=inference_monitor_thread, daemon=True).start()

# ============== camera loop ==============
last_camera_activity = time.time()

def ensure_bgr_for_cv(frame: np.ndarray) -> np.ndarray:
    if frame is None: return frame
    if frame.ndim != 3 or frame.shape[2] < 3: return frame
    if FORCE_RGB:
        return frame[..., ::-1].copy()
    return frame

def _safe_encode_jpeg_for_udp(bgr_img, encode_params, max_bytes=200_000):
    if bgr_img is None or bgr_img.size==0: return None
    cur=bgr_img; attempt=0
    while True:
        ret,j=cv2.imencode('.jpg',cur,encode_params)
        if not ret: return None
        b=j.tobytes()
        if len(b) <= max_bytes or (cur.shape[1]<80 or cur.shape[0]<60):
            return b
        attempt+=1
        scale=0.85
        new_w=max(80,int(cur.shape[1]*scale)); new_h=max(60,int(cur.shape[0]*scale))
        cur=cv2.resize(cur,(new_w,new_h),interpolation=cv2.INTER_AREA)
        if attempt>6: return b


def camera_loop():
    global last_camera_activity, last_heartbeat, phone_connected
    try:
        pic = Picamera2()
        config = pic.create_preview_configuration(main={"size":(FRAME_WIDTH,FRAME_HEIGHT)})
        pic.configure(config); pic.start()
        print("[CAM] camera started")
    except Exception as e:
        print("[CAMERA] failed to start camera:", e)
        time.sleep(1.0); return

    encode_params=[int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    last_summary_time = 0
    while True:
        try:
            frame = pic.capture_array()
            last_camera_activity = time.time()
            if frame is None:
                time.sleep(0.01); continue
            if frame.ndim==3 and frame.shape[2]==4:
                frame = frame[:,:,:3]
            bgr_frame = ensure_bgr_for_cv(frame)
            # stream small video frames
            if state.get("send_video") and PHONE_IP:
                try:
                    ff = _safe_encode_jpeg_for_udp(bgr_frame, encode_params, max_bytes=60000)
                    if ff:
                        try: sock_video.sendto(ff, (PHONE_IP, PHONE_VIDEO_PORT))
                        except Exception as e: print("[VIDEO] send err:", e)
                except Exception as e:
                    print("[VIDEO] encode/send err:", e)
            if not state.get("running"):
                if time.time() - last_heartbeat > 20:
                    phone_connected = False; state["send_video"] = False
                time.sleep(0.02); continue
            # prepare rgb for model and enqueue
            try:
                rgb_for_model = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            except Exception:
                rgb_for_model = bgr_frame[..., ::-1].copy()
            ts = int(time.time()*1000)
            inference_mgr.enqueue_frame(ts, rgb_for_model, bgr_frame.copy())
            # process results
            for _ in range(3):
                res = inference_mgr.try_get_result()
                if res is None: break
                res_ts, dets, annotated_bytes = res
                if not dets:
                    if time.time() - last_summary_time > 0.6:
                        send_log("detections_summary", {"frame_ts": res_ts, "detections": [], "note":"no_detections"})
                        last_summary_time = time.time()
                    continue
                # update tracker -> events
                events = tracker.update(dets)
                for ev in events:
                    if ev.get("appeared"):
                        eid = f"track_{ev['id']}_{res_ts}"
                        payload = {"id": eid, "classname": ev.get("classname","unknown"), "label": ev.get("classname","unknown"), "event": "Appeared", "timestamp": res_ts, "track_id": ev['id'], "conf": round(float(ev.get("conf",0.0)),3)}
                        if annotated_bytes:
                            fname = f"frame_{res_ts}.jpg"
                            try:
                                save_image_file(annotated_bytes, fname)
                                payload["image_filename"] = fname
                                queue_image_for_send(annotated_bytes, fname)
                            except Exception as e:
                                print("[IMG] save err", e)
                        send_log("detection", payload)
                    elif ev.get("disappeared"):
                        eid = f"track_{ev['id']}_{res_ts}"
                        payload = {"id": eid, "classname": ev.get("classname","unknown"), "label": ev.get("classname","unknown"), "event": "Disappeared", "timestamp": res_ts, "track_id": ev['id'], "conf": round(float(ev.get("conf",0.0)),3)}
                        send_log("detection", payload)
            if time.time() - last_heartbeat > 20:
                phone_connected = False; state["send_video"] = False
            time.sleep(0.005)
        except Exception as e:
            print("[CAMERA] loop error:", e)
            try: traceback.print_exc()
            except: pass
            time.sleep(0.05)

# watchdog

def camera_watchdog(start_fn, max_idle=10.0, check_interval=3.0):
    global last_camera_activity
    while True:
        try:
            now = time.time(); idle = now - last_camera_activity
            if idle > max_idle:
                print(f"[WATCHDOG] camera idle {idle:.1f}s > {max_idle}s, restarting camera thread...")
                try:
                    t = threading.Thread(target=start_fn, daemon=True)
                    t.start()
                    last_camera_activity = time.time()
                except Exception as e:
                    print("[WATCHDOG] failed to restart camera thread:", e)
            time.sleep(check_interval)
        except Exception as e:
            print("[WATCHDOG] err:", e); time.sleep(1.0)

# heartbeat

def heartbeat_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    global last_heartbeat
    while True:
        try:
            now_ms = int(time.time()*1000)
            hb = {"event":"heartbeat","data":{"ts": now_ms, "color":"RGB"}}
            msg = json.dumps(hb)
            if PHONE_IP:
                try: sock.sendto(msg.encode(), (PHONE_IP, 6001))
                except Exception: pass
            time.sleep(1.0)
        except Exception as e:
            print("[HEART] err", e); time.sleep(1.0)

# ============== main ==============
if __name__ == '__main__':
    print("🔥 rpi_auto_yolo_tracker.py started")
    t_ctrl = threading.Thread(target=control_listener, daemon=True)
    t_ctrl.start()
    t_cam = threading.Thread(target=camera_loop, daemon=True)
    t_cam.start()
    t_watch = threading.Thread(target=camera_watchdog, args=(lambda: threading.Thread(target=camera_loop, daemon=True).start(),12.0,3.0), daemon=True)
    t_watch.start()
    t_img = threading.Thread(target=image_sender_thread, daemon=True)
    t_img.start()
    t_hb = threading.Thread(target=heartbeat_loop, daemon=True)
    t_hb.start()

    try:
        while True:
            try:
                alive = inference_mgr.proc.is_alive() if inference_mgr.proc else False
                print(f"[MAIN-DIAG] phone={PHONE_IP} running={state.get('running')} send_video={state.get('send_video')} inf_proc_alive={alive}")
            except Exception:
                pass
            if time.time() - last_heartbeat > 40:
                PHONE_IP = None; phone_connected = False; state["send_video"] = False
            time.sleep(4.0)
    except KeyboardInterrupt:
        print("Exiting.")
    finally:
        try: image_send_queue.put_nowait(None)
        except: pass
        try: inference_mgr.stop_process()
        except: pass