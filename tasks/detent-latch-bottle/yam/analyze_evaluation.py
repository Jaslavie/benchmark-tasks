"""Independently score a completed single-trial Inspect Robots bottle run.

Infrastructure errors retain partial trace diagnostics but receive no valid
benchmark score. This script performs no model calls and never reads credentials.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detent_bottle.scoring import Scorer
from detent_bottle.support import table_support


def analyze(directory):
    logs = [p for p in directory.glob("adhoc_*.json") if not p.name.endswith(".live.json")]
    if len(logs) != 1:
        raise ValueError("Expected one final evaluation JSON; live logs are not completed results")
    log = json.loads(logs[0].read_text())
    samples = log.get("samples", [])
    results = log.get("results", {})
    trace = directory / "measurements.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    scorer = Scorer()
    first_credit = {}
    regraded_support = 0
    for row in rows:
        if "bottle_on_table" not in row:
            row.update(table_support(row["bottle_world_xyz"],row["upright_dot"]))
            regraded_support += 1
        value = scorer.update(row)
        if value > 0:
            first_credit.setdefault(str(value), row["step"])
    counted_steps = sum(event["kind"] == "step" for sample in samples
                        for epoch in sample.get("events", []) for event in epoch)
    errors = [log.get("error"), *(s.get("error") for s in samples)]
    framework_score = results.get("metrics", {}).get("task_score")
    valid = (not any(errors) and results.get("errored_trials") == 0
             and results.get("total_trials") == 1 and bool(rows)
             and counted_steps == len(rows) and framework_score is not None
             and all(row.get("wrist_force_measured") is True for row in rows))
    if valid and abs(float(framework_score) - scorer.score) > 1e-9:
        raise ValueError("Framework score disagrees with the independent physical trace scorer")
    transcripts = [t for s in samples for t in (s.get("policy_transcripts") or [])
                   if isinstance(t, dict)]
    requests = [r for t in transcripts for r in t.get("requests", []) if isinstance(r, dict)]
    text = (directory / "run.log").read_text(errors="replace")
    ids = [r.get("session_id") for r in requests]
    continuous = bool(ids) and all(ids) and len(set(ids)) == 1 and "Dream: expired;" not in text
    gate_file = directory / "continuity.jsonl"
    gate_rows = [json.loads(line) for line in gate_file.read_text().splitlines()] if gate_file.exists() else []
    chunks = [r for r in gate_rows if r.get("kind") == "chunk"]
    if gate_file.exists():
        continuous = continuous and len(chunks) == len(requests) and all(r["continuous"] for r in chunks)
    return {
        "continuous_model_evaluation": bool(valid and continuous),
        "continuous_model_score": float(framework_score) if valid and continuous else None,
        "unique_server_sessions": len(set(i for i in ids if i)),
        "session_transitions": sum(a != b for a,b in zip(ids, ids[1:])),
        "table_support_regraded_samples": regraded_support,
        "all_samples_on_table": all(r.get("bottle_on_table") is True for r in rows),
        "table_support_criterion": "Circular base low edge within10mm of tabletop; base centre x in[-.154,.874], y in[-.614,.614]m",
        "valid_completed_trial": bool(valid),
        "benchmark_score": float(framework_score) if valid else None,
        "framework_score": framework_score,
        "partial_or_complete_trace_result": scorer.result() if rows else None,
        "first_credit_ticks": first_credit,
        "recorded_steps": len(rows),
        "framework_step_events": counted_steps,
        "policy": log.get("eval", {}).get("policy"),
        "model": transcripts[0].get("model") if transcripts else None,
        "termination_reasons": [r for s in samples for r in s.get("termination_reasons", [])],
        "framework_results": results,
        "has_infrastructure_error": bool(any(errors) or results.get("errored_trials", 0)),
        "max_sampled_wrist_force_n": max((r["wrist_force_n"] for r in rows), default=None),
        "max_sampled_tilt_deg": math.degrees(math.acos(max(-1., min(1., min(
            r["upright_dot"] for r in rows))))) if rows else None,
        "max_angles_deg": {key: max((r[key] for r in rows), default=None)
                           for key in ("lid_deg", "loop_deg", "button_deg")},
        "dream_request_count": len(requests),
        "dream_request_attempts": sum(int(r.get("attempts", 1)) for r in requests),
        "expired_retry_messages": text.count("Dream: expired;"),
        "tracking_error_rad": [t.get("tracking_error_rad") for t in transcripts],
        "guard_modifications": sum(event["kind"] == "approval" and bool(event.get("data", {}).get("modified"))
                                   for s in samples for epoch in s.get("events", []) for event in epoch),
        "final_log": logs[0].name,
        "scope": "One fixed-layout trial; no success-rate estimate or robustness claim.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = analyze(args.directory)
    (args.directory / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
