"""Score a saved native measurement trace."""
import argparse
import json
from pathlib import Path
from detent_bottle.scoring import Scorer, score_trace

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    print(json.dumps(score_trace(parser.parse_args().trace), indent=2))
