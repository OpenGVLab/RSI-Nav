#!/usr/bin/env python3
"""Summarize RSI-Nav evo timing JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def iter_timing_paths(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if (path / "timings.jsonl").exists():
        yield path / "timings.jsonl"
        return
    for candidate in sorted(path.glob("data/round_*/timings.jsonl")):
        yield candidate


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def seconds(row: dict[str, Any], key: str = "seconds") -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def add_metric(groups, label: str, value: float) -> None:
    groups[label]["seconds"] += value
    groups[label]["events"] += 1


def summarize(path: Path) -> tuple[dict[str, dict[str, float]], int]:
    groups: dict[str, dict[str, float]] = defaultdict(lambda: {"seconds": 0.0, "events": 0})
    rows = 0
    for timing_path in iter_timing_paths(path):
        for row in read_jsonl(timing_path):
            rows += 1
            stage = str(row.get("stage") or "unknown")
            if stage == "rollout_worker_load":
                add_metric(groups, "rollout.data_setup", seconds(row, "data_setup_seconds"))
                add_metric(groups, "rollout.env_setup", seconds(row, "env_setup_seconds"))
                add_metric(groups, "rollout.model_load", seconds(row, "model_load_seconds"))
                add_metric(groups, "rollout.processor_load", seconds(row, "processor_load_seconds"))
            elif stage == "rollout_candidate":
                add_metric(groups, "rollout.instruction_generation", seconds(row, "instruction_generation_seconds"))
                add_metric(groups, "rollout.nav_verification", seconds(row, "nav_verification_seconds"))
                add_metric(groups, "rollout.candidate_total", seconds(row))
            elif stage == "rollout_worker_total":
                add_metric(groups, "rollout.worker_wall_sum", seconds(row))
            elif stage == "rollout_command":
                add_metric(groups, "controller.rollout_command_wall", seconds(row))
            elif stage == "train_command":
                add_metric(groups, "controller.train_command_wall", seconds(row))
            elif stage == "checkpoint_rotate":
                add_metric(groups, "controller.checkpoint_rotate", seconds(row))
            elif stage == "benchmark_eval_command":
                add_metric(groups, "controller.benchmark_eval_command_wall", seconds(row))
            else:
                add_metric(groups, f"other.{stage}", seconds(row))
    return groups, rows


def fmt(seconds_value: float) -> str:
    return f"{seconds_value:10.3f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize evo timing events.")
    parser.add_argument(
        "path",
        type=Path,
        help="Path to evo_runs, data/round_xxxxxx, or a timings.jsonl file.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    groups, rows = summarize(args.path)
    total = sum(v["seconds"] for v in groups.values())
    ordered = sorted(groups.items(), key=lambda item: item[1]["seconds"], reverse=True)

    if args.json:
        print(json.dumps({"rows": rows, "total_seconds": total, "groups": groups}, indent=2, ensure_ascii=False))
        return

    print(f"Timing rows: {rows}")
    print(f"Summed stage seconds: {total:.3f}s")
    print("\nstage                                      seconds      percent   events")
    print("-" * 74)
    for name, values in ordered:
        pct = (values["seconds"] / total * 100.0) if total > 0 else 0.0
        print(f"{name:<38} {fmt(values['seconds'])}  {pct:8.2f}%  {int(values['events']):6d}")

    print("\nNotes:")
    print("- controller.*_wall is wall-clock time for that subprocess stage.")
    print("- rollout.worker_wall_sum sums all worker wall times, so it can exceed controller rollout wall time because workers run in parallel.")
    print("- train save time inside the training script is included in controller.train_command_wall unless the train script emits finer-grained timings.")


if __name__ == "__main__":
    main()
