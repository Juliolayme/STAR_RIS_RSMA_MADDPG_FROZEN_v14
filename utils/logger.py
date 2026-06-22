"""Lightweight logger: console + CSV + optional TensorBoard."""
from __future__ import annotations
import csv
import os
import time
from collections import defaultdict
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    _HAS_TB = False
    SummaryWriter = None  # type: ignore


class Logger:
    def __init__(self, log_dir: str, run_name: str, use_tb: bool = True):
        self.run_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.csv_path = os.path.join(self.run_dir, "log.csv")
        self._csv_file = None
        self._csv_writer = None
        self._csv_fields: list[str] = []
        self.use_tb = use_tb and _HAS_TB
        self.tb = SummaryWriter(self.run_dir) if self.use_tb else None
        self._start = time.time()
        self._buffers: dict[str, list[float]] = defaultdict(list)

    def log(self, step: int, data: dict[str, Any]):
        row = {"step": int(step), "wall_time": round(time.time() - self._start, 3)}
        for k, v in data.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                continue
        self._write_csv(row)
        if self.tb is not None:
            for k, v in row.items():
                if k in ("step", "wall_time"):
                    continue
                self.tb.add_scalar(k, v, step)

    def buffer(self, key: str, value: float):
        try:
            self._buffers[key].append(float(value))
        except (TypeError, ValueError):
            pass

    def flush_buffers(self, step: int):
        if not self._buffers:
            return
        avg = {k: sum(v) / max(len(v), 1) for k, v in self._buffers.items()}
        self.log(step, avg)
        self._buffers.clear()

    def _write_csv(self, row: dict[str, Any]):
        new_fields = [k for k in row if k not in self._csv_fields]
        if new_fields or self._csv_writer is None:
            if self._csv_file is not None:
                self._csv_file.close()
            self._csv_fields = list(dict.fromkeys(self._csv_fields + list(row.keys())))
            mode = "a" if os.path.exists(self.csv_path) else "w"
            # Recreate with new header if columns expanded
            if mode == "a" and new_fields:
                # Read old rows, then rewrite with expanded header.
                try:
                    with open(self.csv_path, "r", newline="") as f:
                        reader = csv.DictReader(f)
                        old_rows = list(reader)
                except Exception:
                    old_rows = []
                self._csv_file = open(self.csv_path, "w", newline="")
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields)
                self._csv_writer.writeheader()
                for r in old_rows:
                    self._csv_writer.writerow({k: r.get(k, "") for k in self._csv_fields})
            else:
                self._csv_file = open(self.csv_path, "w", newline="")
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields)
                self._csv_writer.writeheader()
        self._csv_writer.writerow({k: row.get(k, "") for k in self._csv_fields})
        self._csv_file.flush()

    def close(self):
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        if self.tb is not None:
            self.tb.close()
            self.tb = None
