"""Open an upright free-standing bottle with a U safety loop and spring-return rocker.

Fold the safety loop down, then press the lower rocker to release the lid. Keep
the bottle upright on the table. Only the YAM arms may manipulate the bottle.
This PhysX port is experimental; its contact mechanism is being validated.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bottle_runtime import Bottle, install, current_score
from yam_arena.task_file import TABLE_TOP

install()
SCENE = "station"
TOP = TABLE_TOP[SCENE]
OBJECTS = [Bottle(name="bottle", pos=(0.20, 0.0, TOP + 0.0005),
                  size=(0.10, 0.12, 0.282), mass=0.35)]
SUCCESS_HOLD_S = 0.0  # The evaluator already requires eight consecutive ticks.
SCORE_OVER = "end"


def success(objects):
    return current_score() == 1.0


def score(objects):
    return current_score()


def demo():
    """Known-layout robot-only plan; first direct trial reached1.0 in403ticks."""
    from motion import plan
    return plan(TOP)
