import time
import numpy as np
import json
from pathlib import Path

class LatencyMeter:
    def __init__(self):
        self.timings = {}


    def start(self, name):
        self.timings.setdefault(name, []).append({"start": time.time(), "end": None})

    def stop(self, name):
        self.timings[name][-1]["end"] = time.time()

    def summary(self):
        summary = {}
        for k, entries in self.timings.items():
            durations = [e["end"] - e["start"] for e in entries if e["end"]]
            summary[k] = {
                "mean": float(np.mean(durations)) if durations else 0.0,
                "std": float(np.std(durations)) if durations else 0.0,
                "count": len(durations)
            }
        return summary

    def log(self, output_path: Path):
        summary = self.summary()
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"🕒 Latency metrics saved to {output_path}")
