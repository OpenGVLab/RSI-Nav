#!/usr/bin/env python3
"""Round-level RSI-Nav evo controller.

A round is one complete pass over the current candidate pool. A batch is one
rollout/annotation command at a specific candidate offset. Successful rollout
trajectories are first aggregated into a durable pending queue; every time the
queue reaches ``accepted_target`` evo QA rows, the controller dequeues one
train update chunk and optionally launches training on it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from evo_utils import (
    aggregate_batch_success,
    aggregate_round_summaries,
    append_jsonl,
    append_timing,
    atomic_write_json,
    atomic_write_jsonl,
    batch_done_path,
    build_prefiltered_spatial_qa_pool,
    build_prefiltered_stop_qa_pool,
    build_train_update_chunk,
    count_jsonl,
    next_batch_id,
    read_json,
    read_jsonl,
    resolve_current_ckpt,
    rotate_checkpoints,
    round_dir,
    train_update_dir,
    load_trained_spatial_qa_ids_with_status,
    load_trained_stop_qa_ids_with_status,
    utc_timestamp,
    write_round_success_rate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orchestrate RSI-Nav evo rounds.")
    parser.add_argument("--runs-dir", default=str(Path(__file__).resolve().parents[1] / "evo_runs"))
    parser.add_argument(
        "--checkpoint-dir",
        default="",
        help="Directory for rolling current_ckpt/next_ckpt_tmp weights; defaults to --runs-dir.",
    )
    parser.add_argument("--initial-ckpt", default="")
    parser.add_argument("--round-start", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--accepted-target", type=int, default=192, help="Evo QA rows per train update.")
    parser.add_argument("--spatial-qa-per-update", type=int, default=64)
    parser.add_argument(
        "--spatial-qa-datasets",
        default="",
        help="Comma-separated spatial QA JSONL paths or built-in aliases. Empty uses the default merged VSI, ScanQA/LLaVA-3D, and MindCube pool.",
    )
    parser.add_argument("--spatial-qa-seed", type=int, default=42)
    parser.add_argument("--stop-qa-per-update", type=int, default=0)
    parser.add_argument(
        "--stop-qa-datasets",
        default="",
        help="Comma-separated stop QA JSONL paths or built-in aliases. Empty uses the default R2R/RxR/SRDF stop-QA pool when --stop-qa-per-update > 0.",
    )
    parser.add_argument("--stop-qa-seed", type=int, default=42)
    parser.add_argument(
        "--candidate-stride",
        type=int,
        default=512,
        help="Candidate offset increment per rollout batch, usually world_size * max_candidates_per_rank.",
    )
    parser.add_argument(
        "--total-source-candidates",
        type=int,
        default=int(os.environ.get("RSI_NAV_EVO_TOTAL_SOURCE_CANDIDATES", "2000000")),
        help="Estimated total source trajectories in one round, used only for progress/ETA logging.",
    )
    parser.add_argument(
        "--expected-rollout-ranks",
        type=int,
        default=0,
        help="Expected number of rank shards per rollout batch. 0 disables the completeness check.",
    )
    parser.add_argument(
        "--rollout-done-wait-seconds",
        type=float,
        default=float(os.environ.get("RSI_NAV_EVO_ROLLOUT_DONE_WAIT_SECONDS", "7200")),
        help=(
            "After the rollout command exits, wait up to this many seconds for all expected "
            "rank .done files before aggregating. 0 disables the wait."
        ),
    )
    parser.add_argument("--max-rollout-batches-per-round", type=int, default=1000000)
    parser.add_argument(
        "--train-every-batches",
        type=int,
        default=1,
        help="Deprecated compatibility option. Self-evo now opens a training window after every rollout batch.",
    )
    parser.add_argument(
        "--save-batch",
        type=int,
        default=0,
        help="Archive a persistent checkpoint snapshot every N successful evo train/rotate updates. 0 disables snapshots; rolling current_ckpt still updates after each trained batch.",
    )
    parser.add_argument(
        "--max-train-updates-per-window",
        type=int,
        default=1,
        help="Maximum SFT updates to run in one training window. 1 enforces rollout -> train -> rotate -> rollout alternation; 0 means unlimited.",
    )
    parser.add_argument("--keep-batch-success-shards", action="store_true", default=False)
    parser.add_argument(
        "--source-episodes-jsonl",
        default="",
        help="Original rollout episodes JSONL. When set with --source-trajectory-jsonl, controller prefilters trained VA2L ids once at startup.",
    )
    parser.add_argument(
        "--source-trajectory-jsonl",
        default="",
        help="Original rollout trajectory JSONL. When set with --source-episodes-jsonl, controller writes filtered rollout inputs once at startup.",
    )
    parser.add_argument("--rollout-command", required=True, help="Shell command template for one rollout batch.")
    parser.add_argument("--train-command", default="", help="Optional shell command template for each train update chunk.")
    parser.add_argument("--eval-command", default="", help="Optional shell command template for benchmark evaluation.")
    parser.add_argument("--eval-every-rounds", type=int, default=0)
    parser.add_argument("--eval-every-accepted", type=int, default=0)
    parser.add_argument(
        "--post-update-eval-command",
        default="",
        help=(
            "Optional benchmark eval command template run after train updates selected by "
            "--post-update-eval-odd-update-only."
        ),
    )
    parser.add_argument(
        "--post-update-eval-output-root",
        default="xx/evo-vln/evo_traj/30720cand_211_evo/evo_evals",
        help="Root directory for post-update eval outputs.",
    )
    parser.add_argument(
        "--post-update-eval-odd-update-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run post-update eval only after odd-numbered train update ids, e.g. update001 and update003.",
    )
    parser.add_argument(
        "--post-update-eval-even-rollout-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--post-update-eval-node-wait-seconds",
        type=float,
        default=float(os.environ.get("RSI_NAV_EVO_POST_UPDATE_EVAL_NODE_WAIT_SECONDS", "86400")),
        help="Leader wait timeout for all controller nodes to finish a post-update eval command.",
    )
    parser.add_argument(
        "--log-prefix",
        default=os.environ.get("RSI_NAV_EVO_LOG_PREFIX") or socket.gethostname(),
        help="Prefix used when forwarding child process output, e.g. job name shown before >>.",
    )
    parser.add_argument("--rotate-checkpoint", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def controller_node_rank() -> int:
    for name in ("RSI_NAV_EVO_CONTROLLER_NODE_RANK", "ROLLOUT_NODE_RANK", "NODE_RANK"):
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    return 0


def controller_node_count() -> int:
    for name in ("RSI_NAV_EVO_CONTROLLER_NODE_COUNT", "ROLLOUT_NNODES", "NODE_COUNT"):
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    return 1


def control_run_id() -> str:
    return os.environ.get("RSI_NAV_EVO_RUN_ID") or os.environ.get("JOB_ID") or ""


def control_json_matches_run(data: Dict[str, object]) -> bool:
    run_id = control_run_id()
    return not run_id or data.get("run_id") == run_id


def control_dir(runs_dir: str | Path) -> Path:
    return Path(runs_dir) / "control"


def control_prepared_path(runs_dir: str | Path) -> Path:
    return control_dir(runs_dir) / "prepared.json"


def control_plan_path(runs_dir: str | Path, round_id: int) -> Path:
    return control_dir(runs_dir) / f"round_{int(round_id):06d}_plan.json"


def control_batch_done_path(runs_dir: str | Path, round_id: int, batch_id: int) -> Path:
    return control_dir(runs_dir) / f"round_{int(round_id):06d}_batch_{int(batch_id):06d}_done.json"


def control_train_plan_path(runs_dir: str | Path, round_id: int, batch_id: int) -> Path:
    return control_dir(runs_dir) / f"round_{int(round_id):06d}_batch_{int(batch_id):06d}_train_plan.json"


def control_eval_plan_path(runs_dir: str | Path, round_id: int, batch_id: int) -> Path:
    return control_dir(runs_dir) / f"round_{int(round_id):06d}_batch_{int(batch_id):06d}_eval_plan.json"


def control_eval_node_done_path(
    runs_dir: str | Path,
    round_id: int,
    batch_id: int,
    update_id: int,
    node_rank: int,
) -> Path:
    return (
        control_dir(runs_dir)
        / f"round_{int(round_id):06d}_batch_{int(batch_id):06d}_update_{int(update_id):06d}"
        / f"eval_node_{int(node_rank):03d}_done.json"
    )


def control_round_complete_path(runs_dir: str | Path, round_id: int) -> Path:
    return control_dir(runs_dir) / f"round_{int(round_id):06d}_complete.json"


def wait_for_json(path: str | Path, description: str, poll_seconds: float = 2.0) -> Dict[str, object]:
    path = Path(path)
    while True:
        if path.exists():
            try:
                data = read_json(path)
                if control_json_matches_run(data):
                    return data
            except json.JSONDecodeError:
                pass
        print(f"[evo_controller][worker_wait] waiting_for={description} path={path}", flush=True)
        time.sleep(poll_seconds)


def wait_for_rollout_done_files(
    round_path: str | Path,
    batch_id: int,
    expected_ranks: int,
    timeout_seconds: float,
    poll_seconds: float = 5.0,
) -> List[int]:
    if expected_ranks <= 0 or timeout_seconds <= 0:
        return []

    round_path = Path(round_path)
    deadline = time.monotonic() + float(timeout_seconds)
    last_reported_missing: Optional[List[int]] = None
    last_report_time = 0.0
    while True:
        missing = [
            rank
            for rank in range(int(expected_ranks))
            if not batch_done_path(round_path, batch_id, rank).exists()
        ]
        if not missing:
            print(
                "[evo_controller][rollout_done_wait] "
                f"batch={batch_id} expected_ranks={expected_ranks} missing=0",
                flush=True,
            )
            return []

        now = time.monotonic()
        if now >= deadline:
            return missing

        if missing != last_reported_missing or now - last_report_time >= 60.0:
            missing_text = ",".join(f"{rank:03d}" for rank in missing[:32])
            if len(missing) > 32:
                missing_text += f",...(+{len(missing) - 32})"
            print(
                "[evo_controller][rollout_done_wait] "
                f"batch={batch_id} expected_ranks={expected_ranks} missing={len(missing)} "
                f"missing_ranks={missing_text} remaining_seconds={max(0.0, deadline - now):.1f}",
                flush=True,
            )
            last_reported_missing = missing
            last_report_time = now

        time.sleep(min(float(poll_seconds), max(0.1, deadline - now)))


def wait_for_eval_node_done_files(
    runs_dir: str | Path,
    round_id: int,
    batch_id: int,
    update_id: int,
    expected_nodes: int,
    timeout_seconds: float,
    poll_seconds: float = 5.0,
) -> List[int]:
    if expected_nodes <= 1 or timeout_seconds <= 0:
        return []

    deadline = time.monotonic() + float(timeout_seconds)
    last_reported_missing: Optional[List[int]] = None
    last_report_time = 0.0
    while True:
        missing = [
            node_rank
            for node_rank in range(int(expected_nodes))
            if not control_eval_node_done_path(runs_dir, round_id, batch_id, update_id, node_rank).exists()
        ]
        if not missing:
            print(
                "[evo_controller][post_update_eval_node_wait] "
                f"round={round_id} batch={batch_id} update={update_id} "
                f"expected_nodes={expected_nodes} missing=0",
                flush=True,
            )
            return []

        now = time.monotonic()
        if now >= deadline:
            return missing

        if missing != last_reported_missing or now - last_report_time >= 60.0:
            missing_text = ",".join(f"{rank:03d}" for rank in missing[:32])
            if len(missing) > 32:
                missing_text += f",...(+{len(missing) - 32})"
            print(
                "[evo_controller][post_update_eval_node_wait] "
                f"round={round_id} batch={batch_id} update={update_id} "
                f"expected_nodes={expected_nodes} missing={len(missing)} "
                f"missing_nodes={missing_text} remaining_seconds={max(0.0, deadline - now):.1f}",
                flush=True,
            )
            last_reported_missing = missing
            last_report_time = now

        time.sleep(min(float(poll_seconds), max(0.1, deadline - now)))


def wait_for_next_plan(args: argparse.Namespace, round_id: int, last_batch_id: Optional[int]) -> Optional[Dict[str, object]]:
    plan_path = control_plan_path(args.runs_dir, round_id)
    while True:
        complete_path = control_round_complete_path(args.runs_dir, round_id)
        if complete_path.exists():
            complete = read_json(complete_path)
            if control_json_matches_run(complete):
                return None
        if plan_path.exists():
            try:
                plan = read_json(plan_path)
            except json.JSONDecodeError:
                plan = {}
            try:
                batch_id = int(plan.get("batch_id"))
            except (TypeError, ValueError):
                batch_id = None
            if batch_id is not None and batch_id != last_batch_id and control_json_matches_run(plan):
                done_path = control_batch_done_path(args.runs_dir, round_id, batch_id)
                if not done_path.exists():
                    return plan
                done = read_json(done_path)
                if not control_json_matches_run(done):
                    return plan
                if done.get("round_complete"):
                    return None
                last_batch_id = batch_id
        print(
            f"[evo_controller][worker_wait] waiting_for=rollout_plan round={round_id} "
            f"last_batch={last_batch_id} path={plan_path}",
            flush=True,
        )
        time.sleep(2.0)


def run_worker_controller(args: argparse.Namespace, node_rank: int) -> None:
    runs_dir = Path(args.runs_dir)
    logs_dir = runs_dir / "logs"
    wait_for_json(control_prepared_path(runs_dir), "controller_prepared")
    print(
        f"[evo_controller][worker_start] node_rank={node_rank} node_count={controller_node_count()} "
        f"runs_dir={runs_dir}",
        flush=True,
    )
    for round_id in range(args.round_start, args.round_start + args.max_rounds):
        last_batch_id: Optional[int] = None
        while True:
            plan = wait_for_next_plan(args, round_id, last_batch_id)
            if plan is None:
                print(f"[evo_controller][worker_round_complete] node_rank={node_rank} round={round_id}", flush=True)
                break
            batch_id = int(plan["batch_id"])
            context = dict(plan.get("context") or {})
            print(
                f"[evo_controller][worker_rollout_start] node_rank={node_rank} "
                f"round={round_id} batch={batch_id}",
                flush=True,
            )
            status = run_command(
                args.rollout_command,
                context,
                args.dry_run,
                logs_dir / f"round_{round_id:06d}_rollout_node_{node_rank:03d}.log",
            )
            if status != 0:
                raise RuntimeError(
                    f"worker rollout command failed with status {status} "
                    f"for node_rank={node_rank} round={round_id} batch={batch_id}"
                )
            seen_train_updates: set[int] = set()
            seen_eval_updates: set[int] = set()
            while True:
                done_path = control_batch_done_path(args.runs_dir, round_id, batch_id)
                if done_path.exists():
                    done = read_json(done_path)
                    if control_json_matches_run(done):
                        break
                train_plan_path = control_train_plan_path(args.runs_dir, round_id, batch_id)
                if train_plan_path.exists() and args.train_command:
                    try:
                        train_plan = read_json(train_plan_path)
                        if not control_json_matches_run(train_plan):
                            train_plan = {}
                    except json.JSONDecodeError:
                        train_plan = {}
                    try:
                        update_id = int(train_plan.get("update_id"))
                    except (TypeError, ValueError):
                        update_id = -1
                    if update_id >= 0 and update_id not in seen_train_updates:
                        train_context = dict(train_plan.get("context") or {})
                        print(
                            f"[evo_controller][worker_train_start] node_rank={node_rank} "
                            f"round={round_id} batch={batch_id} update={update_id}",
                            flush=True,
                        )
                        status = run_command(
                            args.train_command,
                            train_context,
                            args.dry_run,
                            logs_dir / f"round_{round_id:06d}_train_node_{node_rank:03d}.log",
                        )
                        if status != 0:
                            raise RuntimeError(
                                f"worker train command failed with status {status} for node_rank={node_rank} "
                                f"round={round_id} batch={batch_id} update={update_id}"
                            )
                        seen_train_updates.add(update_id)
                        print(
                            f"[evo_controller][worker_train_done] node_rank={node_rank} "
                            f"round={round_id} batch={batch_id} update={update_id}",
                            flush=True,
                        )
                        continue
                eval_plan_path = control_eval_plan_path(args.runs_dir, round_id, batch_id)
                if eval_plan_path.exists() and args.post_update_eval_command:
                    try:
                        eval_plan = read_json(eval_plan_path)
                        if not control_json_matches_run(eval_plan):
                            eval_plan = {}
                    except json.JSONDecodeError:
                        eval_plan = {}
                    try:
                        eval_update_id = int(eval_plan.get("update_id"))
                    except (TypeError, ValueError):
                        eval_update_id = -1
                    if eval_update_id >= 0 and eval_update_id not in seen_eval_updates:
                        eval_context = dict(eval_plan.get("context") or {})
                        print(
                            f"[evo_controller][worker_post_update_eval_start] node_rank={node_rank} "
                            f"round={round_id} batch={batch_id} update={eval_update_id} "
                            f"output_dir={eval_context.get('eval_output_dir', '')}",
                            flush=True,
                        )
                        status = run_command(
                            args.post_update_eval_command,
                            eval_context,
                            args.dry_run,
                            logs_dir / f"round_{round_id:06d}_batch_{batch_id:06d}_update_{eval_update_id:06d}_eval_node_{node_rank:03d}.log",
                            timing_stage="post_update_eval_command",
                            prefix_output=True,
                            output_prefix=args.log_prefix,
                        )
                        write_eval_node_done(args, round_id, batch_id, eval_update_id, node_rank, status, eval_context)
                        if status != 0:
                            raise RuntimeError(
                                f"worker post-update eval command failed with status {status} "
                                f"for node_rank={node_rank} round={round_id} batch={batch_id} update={eval_update_id}"
                            )
                        seen_eval_updates.add(eval_update_id)
                        print(
                            f"[evo_controller][worker_post_update_eval_done] node_rank={node_rank} "
                            f"round={round_id} batch={batch_id} update={eval_update_id}",
                            flush=True,
                        )
                        continue
                print(
                    f"[evo_controller][worker_wait] waiting_for=batch_done_or_train_or_eval "
                    f"round={round_id} batch={batch_id} node_rank={node_rank}",
                    flush=True,
                )
                time.sleep(2.0)
            print(
                f"[evo_controller][worker_batch_done] node_rank={node_rank} round={round_id} "
                f"batch={batch_id} round_complete={int(bool(done.get('round_complete')))}",
                flush=True,
            )
            last_batch_id = batch_id
            if done.get("round_complete"):
                print(f"[evo_controller][worker_round_complete] node_rank={node_rank} round={round_id}", flush=True)
                break


def candidate_key(row: Dict[str, object]) -> str:
    for key in ("id", "path_id", "episode_id", "trajectory_id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    raise KeyError(f"Cannot infer candidate id from keys={sorted(row.keys())}")


def va2l_source_id(sample_id: object) -> Optional[str]:
    if sample_id is None:
        return None
    value = str(sample_id)
    suffix = "_va2l"
    return value[: -len(suffix)] if value.endswith(suffix) else value


def trained_va2l_path(runs_dir: str | Path) -> Path:
    return Path(runs_dir) / "data" / "trained_accepted_va2l.jsonl"


def annotated_trajectory_path(runs_dir: str | Path) -> Path:
    return Path(runs_dir) / "data" / "annotated_trajectory_ids.jsonl"


def load_source_ids_from_jsonl_with_status(path: Path, id_mode: str) -> tuple[set[str], Dict[str, object]]:
    status: Dict[str, object] = {
        "path": str(path),
        "parent_exists": path.parent.exists(),
        "exists": path.exists(),
        "skip_reason": "",
    }
    if not path.parent.exists():
        status["skip_reason"] = "missing_parent_dir"
        return set(), status
    if not path.exists():
        status["skip_reason"] = "missing_file"
        return set(), status

    source_ids: set[str] = set()
    for row in read_jsonl(path):
        if id_mode == "va2l":
            source_id = va2l_source_id(row.get("id"))
        else:
            source_id = row.get("id") or row.get("trajectory_id") or row.get("path_id") or row.get("episode_id")
        if source_id:
            source_ids.add(str(source_id))
    status["ids"] = len(source_ids)
    return source_ids, status


def load_trained_va2l_source_ids_with_status(runs_dir: str | Path) -> tuple[set[str], Dict[str, object]]:
    return load_source_ids_from_jsonl_with_status(trained_va2l_path(runs_dir), "va2l")


def load_annotated_trajectory_source_ids_with_status(runs_dir: str | Path) -> tuple[set[str], Dict[str, object]]:
    annotated_ids, annotated_status = load_source_ids_from_jsonl_with_status(
        annotated_trajectory_path(runs_dir), "source"
    )
    # Backward compatibility: exclude trajectories that were already accepted and
    # trained before annotated_trajectory_ids.jsonl existed.
    trained_ids, trained_status = load_trained_va2l_source_ids_with_status(runs_dir)
    source_ids = set(annotated_ids) | set(trained_ids)
    annotated_status["ids"] = len(annotated_ids)
    annotated_status["legacy_trained_path"] = trained_status.get("path")
    annotated_status["legacy_trained_exists"] = trained_status.get("exists")
    annotated_status["legacy_trained_ids"] = len(trained_ids)
    annotated_status["combined_ids"] = len(source_ids)
    return source_ids, annotated_status


def prepare_filtered_rollout_inputs(args: argparse.Namespace, runs_dir: Path) -> None:
    episodes_src = Path(args.source_episodes_jsonl) if args.source_episodes_jsonl else None
    traj_src = Path(args.source_trajectory_jsonl) if args.source_trajectory_jsonl else None
    args.filtered_episodes_jsonl = str(episodes_src or "")
    args.filtered_trajectory_jsonl = str(traj_src or "")
    if not episodes_src or not traj_src:
        return

    processed_source_ids, status = load_annotated_trajectory_source_ids_with_status(runs_dir)
    print(
        "[evo_controller][nav_dedup_source] "
        f"path={status['path']} "
        f"parent_exists={int(bool(status.get('parent_exists')))} "
        f"exists={int(bool(status.get('exists')))} "
        f"skip_reason={status.get('skip_reason') or 'none'} "
        f"annotated_ids={status.get('ids', 0)} "
        f"legacy_trained_va2l_ids={status.get('legacy_trained_ids', 0)} "
        f"combined_source_ids={len(processed_source_ids)}",
        flush=True,
    )

    cache_dir = runs_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    filtered_traj = cache_dir / "rollout_trajectory_prefiltered.jsonl"
    filtered_episodes = cache_dir / "rollout_episodes_prefiltered.jsonl"

    original_traj = 0
    kept_traj_count = 0
    kept_ids: set[str] = set()
    tmp_traj = filtered_traj.with_suffix(filtered_traj.suffix + ".tmp")
    with tmp_traj.open("w", encoding="utf-8") as handle:
        for row in read_jsonl(traj_src):
            original_traj += 1
            key = candidate_key(row)
            if key in processed_source_ids:
                continue
            kept_ids.add(key)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept_traj_count += 1
    tmp_traj.replace(filtered_traj)
    deduped_traj = original_traj - kept_traj_count
    print(
        "[evo_controller][dedup] "
        f"data=evo original={original_traj} deduped={deduped_traj} remaining={kept_traj_count} "
        f"processed_source_ids={len(processed_source_ids)} filtered_trajectory_jsonl={filtered_traj}",
        flush=True,
    )

    original_episodes = 0
    kept_episode_count = 0
    tmp_episodes = filtered_episodes.with_suffix(filtered_episodes.suffix + ".tmp")
    with tmp_episodes.open("w", encoding="utf-8") as handle:
        for row in read_jsonl(episodes_src):
            original_episodes += 1
            if candidate_key(row) in kept_ids:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept_episode_count += 1
    tmp_episodes.replace(filtered_episodes)
    print(
        "[evo_controller][dedup] "
        f"data=episodes original={original_episodes} remaining={kept_episode_count} "
        f"matched_trajectory_ids={len(kept_ids)} filtered_episodes_jsonl={filtered_episodes}",
        flush=True,
    )
    args.filtered_episodes_jsonl = str(filtered_episodes)
    args.filtered_trajectory_jsonl = str(filtered_traj)


def prepare_prefiltered_spatial_pool(args: argparse.Namespace, runs_dir: Path) -> None:
    round_path = round_dir(runs_dir, args.round_start)
    trained_ids, status = load_trained_spatial_qa_ids_with_status(round_path)
    print(
        "[evo_controller][spatial_qa_dedup_source] "
        f"path={status['path']} "
        f"parent_exists={int(bool(status.get('parent_exists')))} "
        f"exists={int(bool(status.get('exists')))} "
        f"skip_reason={status.get('skip_reason') or 'none'} "
        f"trained_spatial_ids={len(trained_ids)}",
        flush=True,
    )
    pool, meta = build_prefiltered_spatial_qa_pool(
        Path(__file__).resolve().parents[1],
        split_csv(args.spatial_qa_datasets),
        args.spatial_qa_seed,
        processed_ids=trained_ids,
    )
    args.prefiltered_spatial_pool = pool
    print(
        "[evo_controller][dedup] "
        f"data=spatial original={meta['original']} used_id_deduped={meta['used_id_deduped']} "
        f"duplicate_pool_ids={meta['duplicate_pool_ids']} remaining={meta['remaining']} "
        f"trained_spatial_ids={meta['trained_spatial_ids']}",
        flush=True,
    )


def prepare_prefiltered_stop_pool(args: argparse.Namespace, runs_dir: Path) -> None:
    if int(args.stop_qa_per_update or 0) <= 0:
        args.prefiltered_stop_pool = []
        print(
            "[evo_controller][dedup] data=stop disabled=1 stop_qa_per_update=0",
            flush=True,
        )
        return

    round_path = round_dir(runs_dir, args.round_start)
    trained_ids, status = load_trained_stop_qa_ids_with_status(round_path)
    print(
        "[evo_controller][stop_qa_dedup_source] "
        f"path={status['path']} "
        f"parent_exists={int(bool(status.get('parent_exists')))} "
        f"exists={int(bool(status.get('exists')))} "
        f"skip_reason={status.get('skip_reason') or 'none'} "
        f"trained_stop_ids={len(trained_ids)}",
        flush=True,
    )
    pool, meta = build_prefiltered_stop_qa_pool(
        Path(__file__).resolve().parents[1],
        split_csv(args.stop_qa_datasets),
        args.stop_qa_seed,
        processed_ids=trained_ids,
    )
    args.prefiltered_stop_pool = pool
    print(
        "[evo_controller][dedup] "
        f"data=stop original={meta['original']} used_id_deduped={meta['used_id_deduped']} "
        f"duplicate_pool_ids={meta['duplicate_pool_ids']} remaining={meta['remaining']} "
        f"trained_stop_ids={meta['trained_stop_ids']}",
        flush=True,
    )


def checkpoint_dir(args: argparse.Namespace) -> Path:
    return Path(args.checkpoint_dir) if args.checkpoint_dir else Path(args.runs_dir)


def current_checkpoint_state(args: argparse.Namespace) -> Dict[str, object]:
    state = read_json(checkpoint_dir(args) / "current_ckpt.json")
    if not state:
        state = {"generation": 0, "source": "initial", "path": resolve_current_ckpt(checkpoint_dir(args), fallback=args.initial_ckpt or None)}
    state.setdefault("generation", 0)
    return state


def next_checkpoint_generation(args: argparse.Namespace) -> int:
    try:
        return int(current_checkpoint_state(args).get("generation", 0) or 0) + 1
    except (TypeError, ValueError):
        return 1


def candidate_offset_for_batch(args: argparse.Namespace, round_id: int, batch_id: Optional[int]) -> int:
    if batch_id is None:
        return 0
    offset_batch_id = int(batch_id)
    if (
        bool(getattr(args, "filtered_candidate_offset_enabled", False))
        and int(round_id) == int(getattr(args, "filtered_candidate_offset_round", -1))
    ):
        base_batch_id = int(getattr(args, "filtered_candidate_offset_batch_base", 0) or 0)
        offset_batch_id = max(0, offset_batch_id - base_batch_id)
    return offset_batch_id * int(args.candidate_stride)


def format_context(
    args: argparse.Namespace,
    round_id: int,
    batch_id: Optional[int] = None,
    update_id: Optional[int] = None,
    update_dir: Optional[Path] = None,
) -> Dict[str, str]:
    runs_dir = Path(args.runs_dir)
    rdir = round_dir(runs_dir, round_id)
    ckpt_dir = checkpoint_dir(args)
    current_ckpt = resolve_current_ckpt(ckpt_dir, fallback=args.initial_ckpt or None)
    ckpt_state = current_checkpoint_state(args)
    prev_failed = round_dir(runs_dir, round_id - 1) / "failed.jsonl"
    retry_jsonl = str(prev_failed) if round_id > 0 and prev_failed.exists() else ""
    if update_dir is None and update_id is not None:
        update_dir = train_update_dir(rdir, update_id)
    update_dir = update_dir or rdir
    accepted_vl2a = update_dir / "accepted_vl2a.jsonl"
    accepted_va2l = update_dir / "accepted_va2l.jsonl"
    spatial_qa = update_dir / "spatial_qa.jsonl"
    stop_qa = update_dir / "stop_qa.jsonl"
    failed = rdir / "failed.jsonl"
    eval_name = f"round_{round_id:06d}"
    if batch_id is not None:
        eval_name += f"_batch_{int(batch_id):06d}"
    if update_id is not None:
        eval_name += f"_update_{int(update_id):06d}"
    eval_output_root = Path(getattr(args, "post_update_eval_output_root", "") or "")
    eval_output_dir = eval_output_root / eval_name if str(eval_output_root) else Path(eval_name)
    context = {
        "runs_dir": str(runs_dir),
        "round_dir": str(rdir),
        "round_id": str(round_id),
        "batch_id": "" if batch_id is None else str(batch_id),
        "update_id": "" if update_id is None else str(update_id),
        "update_dir": str(update_dir),
        "candidate_offset": str(candidate_offset_for_batch(args, round_id, batch_id)),
        "filtered_episodes_jsonl": str(getattr(args, "filtered_episodes_jsonl", "") or getattr(args, "source_episodes_jsonl", "")),
        "filtered_trajectory_jsonl": str(getattr(args, "filtered_trajectory_jsonl", "") or getattr(args, "source_trajectory_jsonl", "")),
        "current_ckpt": current_ckpt,
        "current_ckpt_generation": str(ckpt_state.get("generation", 0)),
        "current_ckpt_updated_at": str(ckpt_state.get("updated_at", "")),
        "next_ckpt_tmp": str(ckpt_dir / "next_ckpt_tmp"),
        "accepted_nav_jsonl": str(accepted_vl2a),
        "accepted_vl2a_jsonl": str(accepted_vl2a),
        "accepted_va2l_jsonl": str(accepted_va2l),
        "accepted_records_jsonl": str(accepted_vl2a),
        "spatial_qa_jsonl": str(spatial_qa),
        "stop_qa_jsonl": str(stop_qa),
        "failed_jsonl": str(failed),
        "failed_retry_jsonl": str(failed),
        "retry_jsonl": retry_jsonl,
        "trained_accepted_nav_jsonl": str(runs_dir / "data" / "trained_accepted_vl2a.jsonl"),
        "trained_accepted_vl2a_jsonl": str(runs_dir / "data" / "trained_accepted_vl2a.jsonl"),
        "trained_accepted_va2l_jsonl": str(runs_dir / "data" / "trained_accepted_va2l.jsonl"),
        "annotated_trajectory_ids_jsonl": str(runs_dir / "data" / "annotated_trajectory_ids.jsonl"),
        "trained_spatial_qa_jsonl": str(runs_dir / "data" / "trained_spatial_qa.jsonl"),
        "trained_stop_qa_jsonl": str(runs_dir / "data" / "trained_stop_qa.jsonl"),
        "post_update_eval_output_root": str(eval_output_root),
        "eval_output_root": str(eval_output_root),
        "post_update_eval_output_dir": str(eval_output_dir),
        "eval_output_dir": str(eval_output_dir),
    }
    return context


def local_timestamp() -> str:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if len(stamp) >= 5:
        return f"{stamp[:-2]}:{stamp[-2:]}"
    return stamp


def run_command(
    template: str,
    context: Dict[str, str],
    dry_run: bool,
    log_path: Path,
    timing_stage: str = "command",
    prefix_output: bool = False,
    output_prefix: str = "",
    timing_totals: Optional[Dict[str, float]] = None,
) -> int:
    command = template.format(**context)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status = 0
    with log_path.open("a", encoding="utf-8") as log:
        header = f"\n[{utc_timestamp()}] $ {command}\n"
        print(header, end="", flush=True)
        log.write(header)
        log.flush()
        if dry_run:
            print(f"[dry-run] {command}")
        else:
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("PYTHONIOENCODING", "utf-8")
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert process.stdout is not None
            for line in process.stdout:
                if prefix_output:
                    prefix = output_prefix or timing_stage
                    line = f"[{local_timestamp()}] {prefix} >> {line}"
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
    elapsed = time.monotonic() - started
    if timing_totals is not None:
        timing_totals[timing_stage] = float(timing_totals.get(timing_stage, 0.0)) + elapsed
    print(
        "[evo_controller][timing] "
        f"stage={timing_stage} round={context.get('round_id', '')} "
        f"batch={context.get('batch_id', '')} update={context.get('update_id', '')} "
        f"seconds={elapsed:.3f} status={status}",
        flush=True,
    )
    round_path = context.get("round_dir")
    if round_path:
        append_timing(
            round_path,
            {
                "source": "controller",
                "stage": timing_stage,
                "round_id": context.get("round_id"),
                "batch_id": context.get("batch_id"),
                "update_id": context.get("update_id"),
                "seconds": elapsed,
                "status": status,
                "dry_run": dry_run,
                "log_path": str(log_path),
            },
        )
    return status


def print_checkpoint_output_summary(output_dir: str | Path, label: str) -> None:
    path = Path(output_dir)
    if not path.exists():
        print(f"[evo_controller][checkpoint_output] {label} path={path} exists=0", flush=True)
        return

    checkpoint_dirs = sorted(p for p in path.glob("checkpoint-*") if p.is_dir())
    weight_files = sorted(
        list(path.glob("*.safetensors"))
        + list(path.glob("*.bin"))
        + list(path.glob("*.pt"))
        + list(path.glob("*.pth"))
    )
    important_files = [
        name
        for name in (
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer_config.json",
            "trainer_state.json",
            "training_args.bin",
        )
        if (path / name).exists()
    ]
    latest_checkpoints = ",".join(p.name for p in checkpoint_dirs[-4:]) or "none"
    print(
        "[evo_controller][checkpoint_output] "
        f"{label} path={path} exists=1 "
        f"checkpoint_dirs={len(checkpoint_dirs)} latest={latest_checkpoints} "
        f"top_level_weight_files={len(weight_files)} metadata_files={','.join(important_files) or 'none'}",
        flush=True,
    )


def should_eval(args: argparse.Namespace, round_id: int, total_accepted: int) -> bool:
    if not args.eval_command:
        return False
    if args.eval_every_rounds and (round_id + 1) % args.eval_every_rounds == 0:
        return True
    if args.eval_every_accepted and total_accepted and total_accepted % args.eval_every_accepted == 0:
        return True
    return False


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def percent(part: int, total: int) -> float:
    return (float(part) / float(total) * 100.0) if total > 0 else 0.0


def print_batch_runtime_summary(
    start_time: float,
    round_id: int,
    batch_id: int,
    summary: Dict[str, object],
    completed_round_candidates: int,
    completed_round_accepted: int,
    completed_round_failed: int,
    timing_totals: Dict[str, float],
) -> None:
    elapsed_seconds = max(0.001, time.monotonic() - start_time)
    round_candidates = int(summary.get("candidates", 0))
    round_accepted = int(summary.get("accepted", 0))
    round_failed = int(summary.get("failed", 0))
    processed = completed_round_candidates + round_candidates
    accepted = completed_round_accepted + round_accepted
    failed = completed_round_failed + round_failed
    eval_seconds = (
        float(timing_totals.get("rollout_command", 0.0))
        + float(timing_totals.get("benchmark_eval_command", 0.0))
        + float(timing_totals.get("post_update_eval_command", 0.0))
    )
    train_seconds = float(timing_totals.get("train_command", 0.0))
    print(
        "[evo_controller][batch_runtime_summary] "
        f"elapsed={format_elapsed(elapsed_seconds)} elapsed_seconds={elapsed_seconds:.3f} "
        f"round={round_id} batch={batch_id} "
        f"processed_trajectories={processed} accepted={accepted} failed={failed} "
        f"accepted_pct={percent(accepted, processed):.2f} failed_pct={percent(failed, processed):.2f} "
        f"eval_seconds={eval_seconds:.3f} eval_elapsed={format_elapsed(eval_seconds)} "
        f"train_seconds={train_seconds:.3f} train_elapsed={format_elapsed(train_seconds)} "
        f"rollout_seconds={float(timing_totals.get('rollout_command', 0.0)):.3f} "
        f"benchmark_eval_seconds={float(timing_totals.get('benchmark_eval_command', 0.0)):.3f} "
        f"post_update_eval_seconds={float(timing_totals.get('post_update_eval_command', 0.0)):.3f} "
        f"checkpoint_snapshot_seconds={float(timing_totals.get('checkpoint_snapshot', 0.0)):.3f}",
        flush=True,
    )


def print_round_eta(
    start_time: float,
    round_id: int,
    batch_id: int,
    summary: Dict[str, object],
    completed_round_candidates: int,
    total_source_candidates: int,
) -> None:
    total = int(total_source_candidates or 0)
    if total <= 0:
        return
    elapsed_seconds = max(0.001, time.monotonic() - start_time)
    processed = completed_round_candidates + int(summary.get("candidates", 0) or 0)
    processed = max(0, processed)
    remaining = max(0, total - processed)
    rate = processed / elapsed_seconds if elapsed_seconds > 0 else 0.0
    eta_seconds = (remaining / rate) if rate > 0 else 0.0
    print(
        "[evo_controller][round_eta] "
        f"elapsed={format_elapsed(elapsed_seconds)} elapsed_seconds={elapsed_seconds:.3f} "
        f"round={round_id} batch={batch_id} "
        f"processed_source={processed} total_source={total} remaining_source={remaining} "
        f"round_progress_pct={percent(processed, total):.6f} "
        f"source_per_hour={rate * 3600.0:.2f} "
        f"eta_seconds={eta_seconds:.3f} eta_elapsed={format_elapsed(eta_seconds)}",
        flush=True,
    )


def print_run_progress(
    start_time: float,
    round_id: int,
    batch_id: int,
    summary: Dict[str, object],
    completed_round_accepted: int,
    completed_round_train_updates: int,
) -> None:
    elapsed_seconds = max(0.001, time.monotonic() - start_time)
    accepted_total = completed_round_accepted + int(summary.get("accepted", 0))
    train_updates_total = completed_round_train_updates + int(summary.get("train_updates", 0))
    success_per_hour = accepted_total / elapsed_seconds * 3600.0
    updates_per_hour = train_updates_total / elapsed_seconds * 3600.0
    print(
        "[evo_controller][progress] "
        f"elapsed={format_elapsed(elapsed_seconds)} "
        f"round={round_id} batch={batch_id} "
        f"successful_trajectories={accepted_total} "
        f"train_updates={train_updates_total} "
        f"pending_queue={summary.get('pending_queue_size', 0)} "
        f"success_rate={float(summary.get('success_rate', 0.0)):.6f} "
        f"success_per_hour={success_per_hour:.2f} "
        f"train_updates_per_hour={updates_per_hour:.2f}",
        flush=True,
    )


def print_round_jsonl_status(round_id: int, rdir: Path, summary: Dict[str, object]) -> None:
    failed = rdir / "failed.jsonl"
    print(
        "[evo_controller][jsonl] "
        f"round={round_id} successful_trajectories={summary.get('accepted', 0)} "
        f"queue_head={summary.get('pending_queue_head', 0)} "
        f"queue_size={summary.get('pending_queue_size', 0)} "
        f"train_updates={summary.get('train_updates', 0)} "
        f"failed_jsonl={failed} rows={count_jsonl(failed)} "
        f"success_rate={float(summary.get('success_rate', 0.0)):.6f}",
        flush=True,
    )


def print_eval_context(args: argparse.Namespace, round_id: int, total_accepted: int, context: Dict[str, str]) -> None:
    print(
        "[evo_controller][eval] "
        f"round={round_id} total_accepted={total_accepted} current_ckpt={context['current_ckpt']} "
        f"failed_jsonl={context['failed_jsonl']}:{count_jsonl(context['failed_jsonl'])}",
        flush=True,
    )
    print(f"[evo_controller][eval] command={args.eval_command.format(**context)}", flush=True)


def should_run_post_update_eval(args: argparse.Namespace, update_id: Optional[int]) -> bool:
    if not args.post_update_eval_command:
        return False
    if update_id is None:
        return False
    if args.post_update_eval_odd_update_only and int(update_id) % 2 != 1:
        return False
    return True


def write_eval_node_done(
    args: argparse.Namespace,
    round_id: int,
    batch_id: int,
    update_id: int,
    node_rank: int,
    status: int,
    context: Dict[str, str],
) -> None:
    atomic_write_json(
        control_eval_node_done_path(args.runs_dir, round_id, batch_id, update_id, node_rank),
        {
            "round_id": round_id,
            "batch_id": batch_id,
            "update_id": update_id,
            "node_rank": node_rank,
            "status": int(status),
            "eval_output_dir": context.get("eval_output_dir", ""),
            "run_id": control_run_id(),
            "created_at": utc_timestamp(),
        },
    )


def run_post_update_eval(
    args: argparse.Namespace,
    round_id: int,
    batch_id: int,
    update_id: int,
    update_dir: Path,
    logs_dir: Path,
    timing_totals: Optional[Dict[str, float]] = None,
) -> None:
    if not should_run_post_update_eval(args, update_id):
        return

    context = format_context(args, round_id, batch_id=batch_id, update_id=update_id, update_dir=update_dir)
    eval_output_dir = Path(context["eval_output_dir"])
    if not args.dry_run:
        eval_output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        control_eval_plan_path(args.runs_dir, round_id, batch_id),
        {
            "round_id": round_id,
            "batch_id": batch_id,
            "update_id": update_id,
            "context": context,
            "run_id": control_run_id(),
            "created_at": utc_timestamp(),
        },
    )
    print(
        "[evo_controller][post_update_eval_start] "
        f"round={round_id} batch={batch_id} update={update_id} "
        f"checkpoint={context['current_ckpt']} output_dir={eval_output_dir}",
        flush=True,
    )
    status = run_command(
        args.post_update_eval_command,
        context,
        args.dry_run,
        logs_dir / f"round_{round_id:06d}_batch_{batch_id:06d}_update_{update_id:06d}_eval.log",
        timing_stage="post_update_eval_command",
        prefix_output=True,
        output_prefix=args.log_prefix,
        timing_totals=timing_totals,
    )
    write_eval_node_done(args, round_id, batch_id, update_id, 0, status, context)
    if status != 0:
        raise RuntimeError(
            f"post-update eval command failed with status {status} "
            f"for round={round_id} batch={batch_id} update={update_id}"
        )
    missing = wait_for_eval_node_done_files(
        args.runs_dir,
        round_id,
        batch_id,
        update_id,
        expected_nodes=controller_node_count(),
        timeout_seconds=args.post_update_eval_node_wait_seconds,
    )
    if missing:
        raise TimeoutError(
            f"Timed out waiting for post-update eval nodes for round={round_id} "
            f"batch={batch_id} update={update_id}: missing={missing}"
        )
    print(
        "[evo_controller][post_update_eval_done] "
        f"round={round_id} batch={batch_id} update={update_id} output_dir={eval_output_dir}",
        flush=True,
    )


def append_jsonl_file(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0
    count = 0
    for row in read_jsonl(src):
        append_jsonl(dst, row)
        count += 1
    return count


def archive_trained_update(round_id: int, update_id: int, context: Dict[str, str]) -> None:
    mappings = [
        (Path(context["accepted_vl2a_jsonl"]), Path(context["trained_accepted_vl2a_jsonl"]), "vl2a"),
        (Path(context["accepted_va2l_jsonl"]), Path(context["trained_accepted_va2l_jsonl"]), "va2l"),
        (Path(context["spatial_qa_jsonl"]), Path(context["trained_spatial_qa_jsonl"]), "spatial"),
        (Path(context["stop_qa_jsonl"]), Path(context["trained_stop_qa_jsonl"]), "stop"),
    ]
    parts = []
    for src, dst, name in mappings:
        appended = append_jsonl_file(src, dst)
        parts.append(f"{name}={appended}->{dst}")
    print(f"[evo_controller][trained_update_archive] round={round_id} update={update_id} " + " ".join(parts), flush=True)


def write_initial_pointer(args: argparse.Namespace) -> None:
    ckpt_dir = checkpoint_dir(args)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pointer = ckpt_dir / "current_ckpt.txt"
    current = ckpt_dir / "current_ckpt"
    state_path = ckpt_dir / "current_ckpt.json"
    if not current.exists() and not pointer.exists() and args.initial_ckpt:
        pointer.write_text(args.initial_ckpt.strip() + "\n", encoding="utf-8")
    if not state_path.exists():
        current_path = str(current) if current.exists() else (args.initial_ckpt.strip() if args.initial_ckpt else "")
        if current_path:
            atomic_write_json(
                state_path,
                {
                    "path": current_path,
                    "generation": 0,
                    "source": "initial",
                    "updated_at": utc_timestamp(),
                },
            )


def should_process_train_updates(args: argparse.Namespace, batch_id: int, round_complete: bool = False) -> bool:
    return True


def should_save_checkpoint_snapshot(args: argparse.Namespace, generation: int) -> bool:
    interval = int(args.save_batch or 0)
    generation = int(generation or 0)
    return interval > 0 and generation > 0 and (generation % interval == 0)


def save_checkpoint_snapshot(
    args: argparse.Namespace,
    round_id: int,
    batch_id: int,
    update_id: int,
    timing_totals: Optional[Dict[str, float]] = None,
    reason: str = "save_batch",
    dst_name: Optional[str] = None,
) -> None:
    source = checkpoint_dir(args) / "current_ckpt"
    if not source.exists():
        print(f"[evo_controller][checkpoint_snapshot] skipped missing_current_ckpt path={source} reason={reason}", flush=True)
        return
    dst_root = checkpoint_dir(args) / "saved_ckpts"
    dst = dst_root / (dst_name or f"round_{round_id:06d}_batch_{int(batch_id):06d}_update_{int(update_id):06d}")
    started = time.monotonic()
    dst_root.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if source.is_dir() and not source.is_symlink():
        shutil.copytree(source, dst)
    else:
        dst.mkdir(parents=True, exist_ok=True)
        resolved = source.resolve() if source.is_symlink() else source
        if resolved.is_dir():
            shutil.copytree(resolved, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(resolved, dst / resolved.name)
    seconds = time.monotonic() - started
    if timing_totals is not None:
        timing_totals["checkpoint_snapshot"] = float(timing_totals.get("checkpoint_snapshot", 0.0)) + seconds
    print(
        "[evo_controller][checkpoint_snapshot] "
        f"round={round_id} batch={batch_id} update={update_id} save_batch={args.save_batch} "
        f"reason={reason} path={dst} seconds={seconds:.3f}",
        flush=True,
    )


def save_round_final_checkpoint(
    args: argparse.Namespace,
    round_id: int,
    batch_id: int,
    timing_totals: Optional[Dict[str, float]] = None,
) -> None:
    if args.dry_run:
        print(f"[evo_controller][round_final_checkpoint] skipped dry_run round={round_id} batch={batch_id}", flush=True)
        return
    state = current_checkpoint_state(args)
    generation = int(state.get("generation", 0) or 0)
    save_checkpoint_snapshot(
        args,
        round_id,
        batch_id,
        generation,
        timing_totals=timing_totals,
        reason="round_complete",
        dst_name=f"round_{round_id:06d}_final",
    )


def print_no_train_update(args: argparse.Namespace, round_id: int, batch_id: int, summary: Dict[str, object]) -> None:
    print(
        "[evo_controller][train_update_skipped] "
        f"round={round_id} batch={batch_id} reason=pending_success_below_target "
        f"pending_queue={summary.get('pending_queue_size', 0)} accepted_target={args.accepted_target}",
        flush=True,
    )


def clean_next_checkpoint_tmp(path: str | Path, dry_run: bool) -> None:
    next_path = Path(path)
    if dry_run or not next_path.exists():
        return
    started = time.monotonic()
    if next_path.is_dir() and not next_path.is_symlink():
        shutil.rmtree(next_path)
    else:
        next_path.unlink()
    print(
        "[evo_controller][checkpoint_prepare] "
        f"removed_stale_next_ckpt_tmp={next_path} seconds={time.monotonic() - started:.3f}",
        flush=True,
    )


def parse_train_timings(log_path: str | Path) -> Dict[str, float]:
    path = Path(log_path)
    timings: Dict[str, float] = {}
    if not path.exists():
        return timings
    pattern = re.compile(r"stage=([^\s]+)\s+seconds=([0-9.]+)")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "[evo_train][timing]" not in line:
                continue
            match = pattern.search(line)
            if not match:
                continue
            try:
                timings[match.group(1)] = float(match.group(2))
            except ValueError:
                continue
    return timings


def summarize_rollout_rank_timings(batch_summary: Dict[str, object]) -> Dict[str, float]:
    summaries = batch_summary.get("rank_summaries") or []
    result = {
        "data_setup_seconds": 0.0,
        "habitat_init_seconds": 0.0,
        "model_load_seconds": 0.0,
        "processor_load_seconds": 0.0,
        "rollout_worker_seconds": 0.0,
    }
    if not isinstance(summaries, list):
        return result
    fields = {
        "data_setup_seconds": "data_setup_seconds",
        "habitat_init_seconds": "env_setup_seconds",
        "model_load_seconds": "model_load_seconds",
        "processor_load_seconds": "processor_load_seconds",
        "rollout_worker_seconds": "run_seconds",
    }
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        timing = summary.get("timing") if isinstance(summary.get("timing"), dict) else {}
        for out_name, source_name in fields.items():
            try:
                value = float(timing.get(source_name, 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            result[out_name] = max(result[out_name], value)
    return result


def print_evo_cycle_timing(
    round_id: int,
    batch_id: int,
    rollout_command_seconds: float,
    batch_summary: Dict[str, object],
    train_update: Dict[str, object],
    cycle_total_seconds: float,
) -> None:
    rollout_timing = summarize_rollout_rank_timings(batch_summary)
    train_timings = train_update.get("train_timings") if isinstance(train_update.get("train_timings"), dict) else {}
    train_save_seconds = float(train_timings.get("save_total", 0.0) or 0.0)
    print(
        "[evo_controller][evo_cycle_timing] "
        f"round={round_id} batch={batch_id} update={train_update.get('update_id')} "
        f"checkpoint_used={train_update.get('checkpoint_used')} "
        f"checkpoint_generation_used={train_update.get('checkpoint_generation_used')} "
        f"checkpoint_generation_after={train_update.get('checkpoint_generation_after')} "
        f"total_seconds={cycle_total_seconds:.3f} total_elapsed={format_elapsed(cycle_total_seconds)} "
        f"habitat_init_seconds={rollout_timing['habitat_init_seconds']:.3f} "
        f"rollout_model_load_seconds={rollout_timing['model_load_seconds']:.3f} "
        f"rollout_processor_load_seconds={rollout_timing['processor_load_seconds']:.3f} "
        f"rollout_command_seconds={rollout_command_seconds:.3f} "
        f"train_startup_seconds={float(train_timings.get('train_startup', 0.0) or 0.0):.3f} "
        f"train_loop_seconds={float(train_timings.get('train_loop', 0.0) or 0.0):.3f} "
        f"train_command_seconds={float(train_update.get('train_command_seconds', 0.0) or 0.0):.3f} "
        f"save_weights_seconds={train_save_seconds:.3f} "
        f"save_state_seconds={float(train_timings.get('save_state', 0.0) or 0.0):.3f} "
        f"save_model_seconds={float(train_timings.get('save_model', 0.0) or 0.0):.3f} "
        f"processor_save_seconds={float(train_timings.get('processor_save', 0.0) or 0.0):.3f} "
        f"checkpoint_rotate_seconds={float(train_update.get('checkpoint_rotate_seconds', 0.0) or 0.0):.3f} "
        f"checkpoint_snapshot_seconds={float(train_update.get('checkpoint_snapshot_seconds', 0.0) or 0.0):.3f}",
        flush=True,
    )


def process_ready_train_updates(
    args: argparse.Namespace,
    round_id: int,
    rdir: Path,
    logs_dir: Path,
    timing_totals: Optional[Dict[str, float]] = None,
    batch_id: Optional[int] = None,
) -> List[Dict[str, object]]:
    train_updates: List[Dict[str, object]] = []
    max_updates = int(args.max_train_updates_per_window or 0)
    while True:
        if max_updates > 0 and len(train_updates) >= max_updates:
            return train_updates
        current_ckpt = resolve_current_ckpt(checkpoint_dir(args), fallback=args.initial_ckpt or None)
        checkpoint_state_before = current_checkpoint_state(args)
        update = build_train_update_chunk(
            rdir,
            accepted_target=args.accepted_target,
            spatial_qa_count=args.spatial_qa_per_update,
            spatial_qa_datasets=split_csv(args.spatial_qa_datasets),
            seed=args.spatial_qa_seed,
            repo_root=Path(__file__).resolve().parents[1],
            checkpoint=current_ckpt,
            checkpoint_version=current_ckpt,
            allow_partial=False,
            prefiltered_spatial_pool=getattr(args, "prefiltered_spatial_pool", None),
            stop_qa_count=args.stop_qa_per_update,
            stop_qa_datasets=split_csv(args.stop_qa_datasets),
            stop_qa_seed=args.stop_qa_seed,
            prefiltered_stop_pool=getattr(args, "prefiltered_stop_pool", None),
        )
        if update is None:
            return train_updates
        manifest = update["manifest"]
        update_id = int(manifest["update_id"])
        update_dir = Path(update["update_dir"])
        metric: Dict[str, object] = {
            "update_id": update_id,
            "checkpoint_used": current_ckpt,
            "checkpoint_generation_used": checkpoint_state_before.get("generation", 0),
            "checkpoint_generation_after": checkpoint_state_before.get("generation", 0),
            "train_command_seconds": 0.0,
            "checkpoint_rotate_seconds": 0.0,
            "checkpoint_snapshot_seconds": 0.0,
            "train_timings": {},
        }
        print(
            "[evo_controller][train_update_ready] "
            f"round={round_id} update={update_id} dir={update_dir} "
            f"checkpoint={current_ckpt} checkpoint_generation={checkpoint_state_before.get('generation', 0)} "
            f"evo_qa={manifest.get('accepted_evo_qa', 0)} "
            f"trajectories={manifest['accepted_trajectories']} "
            f"vl2a={manifest['accepted_vl2a']} va2l={manifest['accepted_va2l']} "
            f"spatial_qa={manifest['spatial_qa']} stop_qa={manifest.get('stop_qa', 0)}",
            flush=True,
        )
        train_updates.append(metric)
        if not args.train_command:
            continue
        context = format_context(args, round_id, update_id=update_id, update_dir=update_dir)
        clean_next_checkpoint_tmp(context["next_ckpt_tmp"], dry_run=args.dry_run)
        if batch_id is not None:
            atomic_write_json(
                control_train_plan_path(args.runs_dir, round_id, int(batch_id)),
                {
                    "round_id": round_id,
                    "batch_id": int(batch_id),
                    "update_id": update_id,
                    "context": context,
                    "run_id": control_run_id(),
                    "created_at": utc_timestamp(),
                },
            )
        print_checkpoint_output_summary(context["next_ckpt_tmp"], "before_train")
        train_log_path = logs_dir / f"round_{round_id:06d}_update_{update_id:06d}_train.log"
        train_started = time.monotonic()
        status = run_command(
            args.train_command,
            context,
            args.dry_run,
            train_log_path,
            timing_stage="train_command",
            prefix_output=True,
            output_prefix=args.log_prefix,
            timing_totals=timing_totals,
        )
        metric["train_command_seconds"] = time.monotonic() - train_started
        metric["train_timings"] = parse_train_timings(train_log_path)
        if status != 0:
            raise RuntimeError(f"train command failed with status {status} for round={round_id} update={update_id}")
        print_checkpoint_output_summary(context["next_ckpt_tmp"], "after_train")
        if not args.dry_run:
            archive_trained_update(round_id, update_id, context)
        if args.rotate_checkpoint and not args.dry_run:
            rotate_start = time.monotonic()
            generation_after = next_checkpoint_generation(args)
            rotate_metadata = {
                "generation": generation_after,
                "source": "train_update",
                "round_id": round_id,
                "batch_id": batch_id,
                "update_id": update_id,
                "trained_from": current_ckpt,
                "train_update_dir": str(update_dir),
            }
            rotate_checkpoints(checkpoint_dir(args), next_ckpt=context["next_ckpt_tmp"], metadata=rotate_metadata)
            print_checkpoint_output_summary(checkpoint_dir(args) / "current_ckpt", "current_after_rotate")
            rotate_seconds = time.monotonic() - rotate_start
            metric["checkpoint_rotate_seconds"] = rotate_seconds
            metric["checkpoint_generation_after"] = generation_after
            print(
                "[evo_controller][timing] "
                f"stage=checkpoint_rotate round={round_id} batch={batch_id} update={update_id} "
                f"generation={generation_after} seconds={rotate_seconds:.3f}",
                flush=True,
            )
            if timing_totals is not None:
                timing_totals["checkpoint_rotate"] = float(timing_totals.get("checkpoint_rotate", 0.0)) + rotate_seconds
            snapshot_before = float(timing_totals.get("checkpoint_snapshot", 0.0)) if timing_totals is not None else 0.0
            if batch_id is not None and should_save_checkpoint_snapshot(args, generation_after):
                save_checkpoint_snapshot(args, round_id, int(batch_id), update_id, timing_totals=timing_totals)
            snapshot_after = float(timing_totals.get("checkpoint_snapshot", 0.0)) if timing_totals is not None else snapshot_before
            metric["checkpoint_snapshot_seconds"] = max(0.0, snapshot_after - snapshot_before)
            append_timing(
                rdir,
                {
                    "source": "controller",
                    "stage": "checkpoint_rotate",
                    "round_id": round_id,
                    "batch_id": batch_id,
                    "update_id": update_id,
                    "generation": generation_after,
                    "seconds": rotate_seconds,
                    "next_ckpt_tmp": context["next_ckpt_tmp"],
                    "checkpoint_dir": str(checkpoint_dir(args)),
                },
            )
        if batch_id is not None:
            run_post_update_eval(
                args,
                round_id,
                int(batch_id),
                update_id,
                update_dir,
                logs_dir,
                timing_totals=timing_totals,
            )
    return train_updates


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    node_rank = controller_node_rank()
    if node_rank != 0:
        run_worker_controller(args, node_rank)
        return

    control_dir(runs_dir).mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        control_prepared_path(runs_dir),
        {
            "ready": False,
            "status": "preparing_inputs",
            "node_count": controller_node_count(),
            "run_id": control_run_id(),
            "created_at": utc_timestamp(),
        },
    )
    write_initial_pointer(args)
    prepare_filtered_rollout_inputs(args, runs_dir)
    filtered_traj = str(getattr(args, "filtered_trajectory_jsonl", "") or "")
    source_traj = str(getattr(args, "source_trajectory_jsonl", "") or "")
    args.filtered_candidate_offset_enabled = bool(filtered_traj and filtered_traj != source_traj)
    args.filtered_candidate_offset_round = int(args.round_start)
    args.filtered_candidate_offset_batch_base = (
        next_batch_id(round_dir(runs_dir, args.round_start))
        if args.filtered_candidate_offset_enabled
        else 0
    )
    if args.filtered_candidate_offset_enabled:
        print(
            "[evo_controller][resume_filtered_offset] "
            f"round={args.filtered_candidate_offset_round} "
            f"batch_base={args.filtered_candidate_offset_batch_base} "
            f"candidate_stride={args.candidate_stride} "
            f"filtered_trajectory_jsonl={filtered_traj}",
            flush=True,
        )
    prepare_prefiltered_spatial_pool(args, runs_dir)
    prepare_prefiltered_stop_pool(args, runs_dir)
    atomic_write_json(
        control_prepared_path(runs_dir),
        {
            "ready": True,
            "status": "ready",
            "node_count": controller_node_count(),
            "run_id": control_run_id(),
            "created_at": utc_timestamp(),
            "filtered_episodes_jsonl": str(getattr(args, "filtered_episodes_jsonl", "")),
            "filtered_trajectory_jsonl": str(getattr(args, "filtered_trajectory_jsonl", "")),
        },
    )
    logs_dir = runs_dir / "logs"
    run_start_time = time.monotonic()
    total_accepted_all_rounds = 0
    completed_round_accepted = 0
    completed_round_failed = 0
    completed_round_candidates = 0
    completed_round_train_updates = 0
    timing_totals: Dict[str, float] = {}

    for round_id in range(args.round_start, args.round_start + args.max_rounds):
        rdir = round_dir(runs_dir, round_id)
        rdir.mkdir(parents=True, exist_ok=True)
        rollout_batches = 0

        while True:
            before = aggregate_round_summaries(rdir)
            before_candidates = int(before.get("candidates", 0))
            if rollout_batches >= args.max_rollout_batches_per_round:
                raise RuntimeError(
                    f"round {round_id} stopped after {rollout_batches} batches without exhausting candidates "
                    f"processed={before_candidates}"
                )

            batch_id = next_batch_id(rdir)
            context = format_context(args, round_id, batch_id=batch_id)
            atomic_write_json(
                control_plan_path(runs_dir, round_id),
                {
                    "round_id": round_id,
                    "batch_id": batch_id,
                    "context": context,
                    "run_id": control_run_id(),
                    "created_at": utc_timestamp(),
                },
            )
            cycle_started = time.monotonic()
            print(
                "[evo_controller][checkpoint_use] "
                f"round={round_id} batch={batch_id} checkpoint={context['current_ckpt']} "
                f"generation={context['current_ckpt_generation']} updated_at={context['current_ckpt_updated_at']} "
                f"candidate_offset={context['candidate_offset']} "
                f"filtered_offset_base_batch={getattr(args, 'filtered_candidate_offset_batch_base', 0)}",
                flush=True,
            )
            rollout_started = time.monotonic()
            status = run_command(
                args.rollout_command,
                context,
                args.dry_run,
                logs_dir / f"round_{round_id:06d}_rollout.log",
                timing_stage="rollout_command",
                timing_totals=timing_totals,
            )
            rollout_command_seconds = time.monotonic() - rollout_started
            if status != 0:
                raise RuntimeError(f"rollout command failed with status {status} for round={round_id} batch={batch_id}")
            rollout_batches += 1
            if args.expected_rollout_ranks > 0 and args.rollout_done_wait_seconds > 0:
                wait_started = time.monotonic()
                missing_done = wait_for_rollout_done_files(
                    rdir,
                    batch_id,
                    expected_ranks=args.expected_rollout_ranks,
                    timeout_seconds=args.rollout_done_wait_seconds,
                )
                wait_seconds = time.monotonic() - wait_started
                timing_totals["rollout_done_wait"] = float(timing_totals.get("rollout_done_wait", 0.0)) + wait_seconds
                append_timing(
                    rdir,
                    {
                        "source": "controller",
                        "stage": "rollout_done_wait",
                        "round_id": round_id,
                        "batch_id": batch_id,
                        "seconds": wait_seconds,
                        "expected_ranks": args.expected_rollout_ranks,
                        "missing_ranks": missing_done,
                        "timeout_seconds": args.rollout_done_wait_seconds,
                    },
                )

            batch_summary = aggregate_batch_success(
                rdir,
                batch_id,
                keep_shards=args.keep_batch_success_shards,
                expected_ranks=args.expected_rollout_ranks,
            )
            print(
                "[evo_controller][batch_success_aggregate] "
                f"round={round_id} batch={batch_id} shards={batch_summary.get('success_shards', 0)} "
                f"success_rows={batch_summary.get('success_rows', 0)} "
                f"vl2a_samples={batch_summary.get('accepted_vl2a_samples', 0)} "
                f"va2l_samples={batch_summary.get('accepted_va2l_samples', 0)} "
                f"appended={batch_summary.get('appended_to_pending', 0)}",
                flush=True,
            )
            after = write_round_success_rate(rdir, round_id, last_batch_id=batch_id)
            after_candidates = int(after.get("candidates", 0))
            round_complete = after_candidates == before_candidates
            train_updates: List[Dict[str, object]] = []
            if should_process_train_updates(args, batch_id, round_complete=round_complete):
                train_updates = process_ready_train_updates(
                    args,
                    round_id,
                    rdir,
                    logs_dir,
                    timing_totals=timing_totals,
                    batch_id=batch_id,
                )
            if not train_updates:
                print_no_train_update(args, round_id, batch_id, after)
            else:
                cycle_total_seconds = time.monotonic() - cycle_started
                for train_update in train_updates:
                    print_evo_cycle_timing(
                        round_id,
                        batch_id,
                        rollout_command_seconds,
                        batch_summary,
                        train_update,
                        cycle_total_seconds,
                    )
            after = write_round_success_rate(rdir, round_id, last_batch_id=batch_id)
            print_round_jsonl_status(round_id, rdir, after)
            print_run_progress(
                run_start_time,
                round_id,
                batch_id,
                after,
                completed_round_accepted,
                completed_round_train_updates,
            )
            print_batch_runtime_summary(
                run_start_time,
                round_id,
                batch_id,
                after,
                completed_round_candidates,
                completed_round_accepted,
                completed_round_failed,
                timing_totals,
            )
            print_round_eta(
                run_start_time,
                round_id,
                batch_id,
                after,
                completed_round_candidates,
                args.total_source_candidates,
            )
            after_candidates = int(after.get("candidates", 0))
            if round_complete:
                save_round_final_checkpoint(args, round_id, batch_id, timing_totals=timing_totals)
                atomic_write_json(
                    control_batch_done_path(runs_dir, round_id, batch_id),
                    {
                        "round_id": round_id,
                        "batch_id": batch_id,
                        "round_complete": True,
                        "run_id": control_run_id(),
                        "created_at": utc_timestamp(),
                    },
                )
                atomic_write_json(
                    control_round_complete_path(runs_dir, round_id),
                    {"round_id": round_id, "batch_id": batch_id, "run_id": control_run_id(), "created_at": utc_timestamp()},
                )
                print(
                    "[evo_controller][round_complete] "
                    f"round={round_id} batch={batch_id} processed={after_candidates} "
                    f"before_candidates={before_candidates} after_candidates={after_candidates} "
                    "reason=no_new_source_candidates_candidate_pool_exhausted",
                    flush=True,
                )
                break
            atomic_write_json(
                control_batch_done_path(runs_dir, round_id, batch_id),
                {
                    "round_id": round_id,
                    "batch_id": batch_id,
                    "round_complete": False,
                    "run_id": control_run_id(),
                    "created_at": utc_timestamp(),
                },
            )

        round_summary = write_round_success_rate(rdir, round_id)
        round_candidates = int(round_summary.get("candidates", 0))
        round_accepted = int(round_summary.get("accepted", 0))
        round_failed = int(round_summary.get("failed", 0))
        round_train_updates = int(round_summary.get("train_updates", 0))
        total_accepted_all_rounds += round_accepted
        completed_round_candidates += round_candidates
        completed_round_accepted += round_accepted
        completed_round_failed += round_failed
        completed_round_train_updates += round_train_updates

        if should_eval(args, round_id, total_accepted_all_rounds):
            context = format_context(args, round_id)
            print_eval_context(args, round_id, total_accepted_all_rounds, context)
            status = run_command(
                args.eval_command,
                context,
                args.dry_run,
                logs_dir / f"round_{round_id:06d}_eval.log",
                timing_stage="benchmark_eval_command",
                timing_totals=timing_totals,
            )
            print(f"[evo_controller][eval] round={round_id} exit_status={status}", flush=True)
            if status != 0:
                raise RuntimeError(f"eval command failed with status {status} for round={round_id}")


if __name__ == "__main__":
    main()
