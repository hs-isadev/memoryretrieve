import json
import os
import time
from deep_sort_realtime.deepsort_tracker import DeepSort as _DeepSortRT

class DeepSort:
    """
    Fixed DeepSort wrapper for deep_sort_realtime.

    Input:
      detections = [
        ([x, y, w, h], conf, class_id),
        ...
      ]

    Output:
      [
        {
          "track_id": int,
          "bbox": [x1, y1, x2, y2],
          "class_id": int,
          "confirmed": True,
          "time_since_update": int
        }
      ]
    """

    def __init__(
        self,
        max_age=6,
        n_init=2,
        max_iou_distance=0.7,
        max_cosine_distance=0.4,
        nn_budget=100,
        max_tracks=10,
        max_ids=50,
        memory_file="memory.json",
    ):
        self.max_tracks = max_tracks
        self.max_ids = max_ids
        self.memory_file = memory_file
        self._load_memory()

        self.tracker = _DeepSortRT(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
        )

    # ---------------- MEMORY ----------------
    def _load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    self.memory = json.load(f)
            except Exception:
                self.memory = {}
        else:
            self.memory = {}

    def _save_memory(self):
        try:
            with open(self.memory_file, "w") as f:
                json.dump(self.memory, f)
        except Exception:
            pass

    # ---------------- UPDATE ----------------
    def update(self, detections, frame):
        """
        detections:
            [([x,y,w,h], conf, class_id), ...]
        frame:
            RGB image (numpy array)
        """
        if not detections:
            tracks = self.tracker.update_tracks([], frame=frame)
            self._update_memory(tracks)
            return []

        try:
            tracks = self.tracker.update_tracks(detections, frame=frame)
        except Exception as e:
            print("[DEEPSORT] update_tracks failed:", e)
            return []

        self._update_memory(tracks)
        return self._format_tracks(tracks)

    # ---------------- MEMORY UPDATE ----------------
    def _update_memory(self, tracks):
        now = time.time()
        active_ids = set()

        for t in tracks:
            if not t.is_confirmed():
                continue
            active_ids.add(t.track_id)
            if len(self.memory) < self.max_ids or str(t.track_id) in self.memory:
                self.memory[str(t.track_id)] = {
                    "class_id": int(t.det_class) if t.det_class is not None else -1,
                    "last_seen": now,
                }

        # expire old
        for tid in list(self.memory.keys()):
            if int(tid) not in active_ids:
                if now - self.memory[tid]["last_seen"] > 3600:
                    del self.memory[tid]

        self._save_memory()

    # ---------------- FORMAT ----------------
    def _format_tracks(self, tracks):
        output = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            try:
                x1, y1, x2, y2 = map(int, t.to_ltrb())
            except Exception:
                continue

            output.append({
                "track_id": int(t.track_id),
                "bbox": [x1, y1, x2, y2],
                "class_id": int(t.det_class) if t.det_class is not None else -1,
                "confirmed": True,
                "time_since_update": int(t.time_since_update),
            })

        return output[: self.max_tracks]

    # ---------------- RESET ----------------
    def delete_all_tracks(self):
        self.tracker.delete_all_tracks()
        self.memory = {}
        self._save_memory()
