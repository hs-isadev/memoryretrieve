# deep_sort_realtime/deepsort_tracker.py
import json
import os
import time
from collections.abc import Iterable

import numpy as np
from deep_sort_realtime.deep_sort import nn_matching
from deep_sort_realtime.deep_sort.detection import Detection
from deep_sort_realtime.deep_sort.tracker import Tracker
from deep_sort_realtime.utils.nms import non_max_suppression

class DeepSortTracker:
    """
    Lightweight DeepSort wrapper for Raspberry Pi.
    Motion-only tracking (no embedder).
    Max ID capped, off-camera memory support, event-based logging.
    """

    def __init__(
        self,
        max_age=6,          # frames object can disappear
        n_init=2,
        iou_distance=0.7,
        max_ids=10,
        memory_file="memory.json",  # long-term object memory
    ):
        self.max_ids = max_ids
        self.memory_file = memory_file
        self._load_memory()

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
        self.tracker = Tracker(
            metric,
            max_iou_distance=iou_distance,
            max_age=max_age,
            n_init=n_init,
        )

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                self.memory = json.load(f)
        else:
            self.memory = {}  # format: {track_id: {"class_id": int, "last_seen": timestamp}}

    def _save_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f)

    def update(self, detections, frame=None):
        """
        detections format:
        [
            [x1, y1, x2, y2, confidence, class_id],
            ...
        ]
        """

        if detections is None or len(detections) == 0:
            tracks = self.tracker.update_tracks([], frame=frame)
            return self._format_tracks(tracks)

        formatted = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            w = x2 - x1
            h = y2 - y1
            formatted.append(([x1, y1, w, h], float(conf), int(cls)))

        tracks = self.tracker.update_tracks(formatted, frame=frame)

        # --- Update memory ---
        self._update_memory(tracks)

        return self._format_tracks(tracks)

    def _update_memory(self, tracks):
        now = time.time()
        # remove old memory if track disappeared long ago
        for tid in list(self.memory.keys()):
            if tid not in [t.track_id for t in tracks]:
                # keep it short term only if object not seen
                if now - self.memory[tid]["last_seen"] > 3600:  # 1 hour
                    del self.memory[tid]

        # add/update current tracks
        for t in tracks:
            if len(self.memory) >= self.max_ids and t.track_id not in self.memory:
                # cap number of stored tracks
                continue
            self.memory[t.track_id] = {
                "class_id": getattr(t, "det_class", None),
                "last_seen": now,
            }

        self._save_memory()

    def _format_tracks(self, tracks):
        """
        Returns clean track objects:
        [
            {
                "track_id": int,
                "bbox": [x1, y1, x2, y2],
                "class_id": int,
                "confirmed": bool,
                "time_since_update": int
            }
        ]
        """
        output = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            l, t, w, h = track.to_ltrb()
            output.append(
                {
                    "track_id": track.track_id,
                    "bbox": [int(l), int(t), int(w), int(h)],
                    "class_id": getattr(track, "det_class", None),
                    "confirmed": track.is_confirmed(),
                    "time_since_update": track.time_since_update,
                }
            )

        return output

    def delete_all_tracks(self):
        self.tracker.delete_all_tracks()
        self.memory = {}
        self._save_memory()
