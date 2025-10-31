import time
import numpy as np
import json
from pathlib import Path

class LatencyMeter:
    """
    Utility class for measuring and summarizing execution latency.
    Tracks start and end times for multiple named stages within the RAG pipeline.
    """

    # Initialize the latency tracker with an empty dictionary of stage timings.
    def __init__(self):
        self.timings = {}


    def start(self, name):
        """
        Start a timer for a given stage name.

        Args:
            name (str): Identifier for the pipeline stage being measured.
        """
        self.timings.setdefault(name, []).append({"start": time.time(), "end": None})

    def stop(self, name):
        """
        Stop the timer for the most recent entry of a given stage.

        Args:
            name (str): Identifier for the pipeline stage whose timer is being stopped.
        """
        self.timings[name][-1]["end"] = time.time()

    def summary(self):
        """
        Compute average, standard deviation, and count of durations for each tracked stage.

        Returns:
            dict: Summary of timing statistics for all stages.
        """
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
        """
        Save the latency summary as a JSON file and print a confirmation message.

        Args:
            output_path (Path): Path to the output JSON file.
        """
        summary = self.summary()
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"🕒 Latency metrics saved to {output_path}")
