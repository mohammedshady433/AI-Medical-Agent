"""Bounded diagnostic timings. Never record bodies, queries, headers or keys."""
from collections import deque
import json
import logging
from logging.handlers import RotatingFileHandler
import re
import time


class Activity:
    def __init__(self, data):
        self.rows = deque(maxlen=200)
        folder = data / "logs"; folder.mkdir(exist_ok=True)
        self.logger = logging.Logger("ems-timings")
        self.handler = RotatingFileHandler(folder / "activity.jsonl", maxBytes=512000, backupCount=2, encoding="utf8")
        self.logger.addHandler(self.handler)
    def add(self, reference, method, route, status, seconds):
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "reference": reference,
               "method": method, "route": re.sub(r"[a-f0-9]{32}", "{id}", route), "status": status,
               "headers_ms": round(seconds * 1000, 1)}
        self.rows.append(row)
        self.logger.warning(json.dumps(row, ensure_ascii=False))
    def close(self): self.handler.close()
