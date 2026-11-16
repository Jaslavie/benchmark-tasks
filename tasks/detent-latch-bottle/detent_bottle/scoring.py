"""Numeric scorer shared by native trials and Inspect Robots log scoring."""
import argparse
import json
import math
from pathlib import Path

CONTROL_HZ = 20
MAX_STEPS = 2400
HOLD_TICKS = 8
INSTRUCTION = "Open the bottle and keep it upright on the table."


class Scorer:
    """Score the current outcome; tipping or an aborted trial permanently vetoes credit."""

    def __init__(self):
        self.steps = 0
        self.holds = {0.25: 0, 0.5: 0, 1.0: 0}
        self.overload_ticks = 0
        self.failure = None
        self.score = 0.0

    def update(self, row):
        step = row["step"]
        if type(step) is not int or step != self.steps + 1 or step > MAX_STEPS:
            raise ValueError("Measurements must contain consecutive control ticks 1..2400")
        keys = ("time_s", "lid_deg", "loop_deg", "upright_dot", "bottle_z_m", "wrist_force_n")
        if not all(isinstance(row[k], (int, float)) and not isinstance(row[k], bool) and math.isfinite(row[k]) for k in keys):
            raise ValueError("Measurements must be finite numbers")
        if abs(row["time_s"] - step / CONTROL_HZ) > 1e-7:
            raise ValueError("Measurement time must match the 20 Hz control tick")
        if type(row["numerical_warning"]) is not bool or row["wrist_force_n"] < 0:
            raise ValueError("Invalid warning or force measurement")
        if "bottle_on_table" in row and type(row["bottle_on_table"]) is not bool:
            raise ValueError("Table support measurement must be boolean")
        self.steps = step
        self.overload_ticks = self.overload_ticks + 1 if row["wrist_force_n"] > 100 else 0
        if row["numerical_warning"]:
            self.failure = self.failure or "numerical_instability"
        elif row["upright_dot"] <= 0.90:
            self.failure = self.failure or "bottle_tipped"
        elif row["bottle_z_m"] < 0.79:
            self.failure = self.failure or "bottle_fell"
        elif row.get("bottle_on_table", True) is False:
            self.failure = self.failure or "bottle_left_table"
        elif self.overload_ticks >= 4:
            self.failure = self.failure or "wrist_force_limit"
        conditions = {
            0.25: row["loop_deg"] >= 150,
            0.5: row["lid_deg"] > 15,
            1.0: row["lid_deg"] > 60,
        }
        for value, holds in conditions.items():
            self.holds[value] = self.holds[value] + 1 if holds else 0
        self.score = 0.0 if self.failure else max(
            (value for value, ticks in self.holds.items() if ticks >= HOLD_TICKS), default=0.0
        )
        return self.score

    def result(self):
        return dict(score=self.score, success=self.score == 1.0,
                    failure=self.failure, steps=self.steps,
                    simulated_seconds=self.steps / CONTROL_HZ)


def score_trace(path):
    scorer = Scorer()
    with Path(path).open() as stream:
        for line in stream:
            scorer.update(json.loads(line))
    if not scorer.steps:
        raise ValueError("The measurement trace is empty")
    return scorer.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-score a saved evaluator measurement trace")
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(score_trace(args.trace), indent=2))
