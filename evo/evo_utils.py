"""Utilities for RSI-Nav evo rollout/training orchestration.

All files created by these helpers are expected to live under the caller-provided
``runs_dir``. The module intentionally has no dependency on the existing qwen3vl
training/eval code so it can be used by rollout, controller, and tests.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence


JsonDict = Dict[str, Any]
DEFAULT_SPATIAL_QA_JSONLS = [
    "xxx/RSI-Nav/data/datasets/vsi_590k/vsi_590k_convert_image32.jsonl",
    "xxx/RSI-Nav/data/datasets/vsi_590k/evohalf.jsonl",
    "xxx/RSI-Nav/data/datasets/scanqa/LLaVA-3D-546K-scannet-only_image32.jsonl",
    "xxx/RSI-Nav/data/datasets/mindcube/MindCube.jsonl",
    "xxx/RSI-Nav/data/datasets/va2l/r2rva2l5x.jsonl",
]
DEFAULT_SPATIAL_QA_DATA_PATHS = {
    DEFAULT_SPATIAL_QA_JSONLS[0]: "xxx/RSI-Nav/data/datasets/vsi_590k/VSI-590K/",
    DEFAULT_SPATIAL_QA_JSONLS[1]: "xxx/RSI-Nav/data/datasets/vsi_590k/VSI-590K/",
    DEFAULT_SPATIAL_QA_JSONLS[2]: "xxx/RSI-Nav/data/datasets/vsi_590k/VSI-590K/",
    DEFAULT_SPATIAL_QA_JSONLS[3]: "xxx/RSI-Nav/data/datasets/mindcube",
    DEFAULT_SPATIAL_QA_JSONLS[4]: "xxx/RSI-Nav/data/trajectory_data/R2R/",
}
SPATIAL_QA_JSONL_ALIASES = {
    "vsi_590k": DEFAULT_SPATIAL_QA_JSONLS[0],
    "evohalf": DEFAULT_SPATIAL_QA_JSONLS[1],
    "scanqa": DEFAULT_SPATIAL_QA_JSONLS[2],
    "mindcube": DEFAULT_SPATIAL_QA_JSONLS[3],
    "r2r_va2l": DEFAULT_SPATIAL_QA_JSONLS[4],
}
_SPATIAL_QA_CACHE: Dict[tuple[str, ...], List[tuple[JsonDict, Path, str]]] = {}
_SPATIAL_QA_SHUFFLED_CACHE: Dict[tuple[str, ...], List[tuple[JsonDict, Path, str]]] = {}

DEFAULT_STOP_QA_JSONLS = [
    "xxx/RSI-Nav/data/datasets/stop/r2r_qwen_32step_episodes_single_stop.jsonl",
    "xxx/RSI-Nav/data/datasets/stop/rxr_qwen_32step_episodes_single_stop.jsonl",
    "xxx/RSI-Nav/data/datasets/stop/srdf_qwen_32step_episodes_single_stop.jsonl",
]
DEFAULT_STOP_QA_DATA_PATHS = {
    DEFAULT_STOP_QA_JSONLS[0]: "xxx/RSI-Nav/data/trajectory_data/R2R/",
    DEFAULT_STOP_QA_JSONLS[1]: "xxx/RSI-Nav/data/trajectory_data/RxR/",
    DEFAULT_STOP_QA_JSONLS[2]: "xxx/RSI-Nav/data/trajectory_data/srdf/images/",
}
STOP_QA_JSONL_ALIASES = {
    "r2r_stop": DEFAULT_STOP_QA_JSONLS[0],
    "stop_r2r": DEFAULT_STOP_QA_JSONLS[0],
    "rxr_stop": DEFAULT_STOP_QA_JSONLS[1],
    "stop_rxr": DEFAULT_STOP_QA_JSONLS[1],
    "srdf_stop": DEFAULT_STOP_QA_JSONLS[2],
    "stop_srdf": DEFAULT_STOP_QA_JSONLS[2],
}
_STOP_QA_CACHE: Dict[tuple[str, ...], List[tuple[JsonDict, Path, str]]] = {}
_STOP_QA_SHUFFLED_CACHE: Dict[tuple[str, ...], List[tuple[JsonDict, Path, str]]] = {}


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def timing_path(round_path: str | Path) -> Path:
    return Path(round_path) / "timings.jsonl"


def append_timing(round_path: str | Path, row: JsonDict) -> None:
    event = dict(row)
    event.setdefault("created_at", utc_timestamp())
    append_jsonl(timing_path(round_path), event)


def read_jsonl(path: str | Path) -> Iterator[JsonDict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def write_jsonl(path: str | Path, rows: Iterable[JsonDict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: str | Path, row: JsonDict) -> None:
    append_jsonl_many(path, [row])


def append_jsonl_many(path: str | Path, rows: Iterable[JsonDict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return count


def atomic_write_json(path: str | Path, data: JsonDict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def write_json_locked(path: str | Path, data: JsonDict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0)
        f.truncate()
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def atomic_write_jsonl(path: str | Path, rows: Iterable[JsonDict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = write_jsonl(tmp, rows)
    tmp.replace(path)
    return count


def count_jsonl(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def ensure_empty_or_create(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def round_dir(runs_dir: str | Path, round_id: int) -> Path:
    return Path(runs_dir) / "data" / f"round_{round_id:06d}"


def batch_dir(runs_dir: str | Path, round_id: int, batch_id: int) -> Path:
    return round_dir(runs_dir, round_id) / f"rollout_batch_{batch_id:06d}"


def train_update_dir(round_path: str | Path, update_id: int) -> Path:
    return Path(round_path) / f"train_update_{update_id:06d}"


def pending_success_path(round_path: str | Path) -> Path:
    return Path(round_path) / "pending_success.jsonl"


def pending_success_state_path(round_path: str | Path) -> Path:
    return Path(round_path) / "pending_success_state.json"


def failed_path(round_path: str | Path) -> Path:
    return Path(round_path) / "failed.jsonl"


def success_rate_path(round_path: str | Path) -> Path:
    return Path(round_path) / "success_rate.json"


def batch_success_path(round_path: str | Path, batch_id: int, rank: int) -> Path:
    return Path(round_path) / f"rollout_batch_{batch_id:06d}" / f"rank_{int(rank):03d}.success.jsonl"


def batch_summary_path(round_path: str | Path, batch_id: int, rank: int) -> Path:
    return Path(round_path) / f"rollout_batch_{batch_id:06d}" / f"rank_{int(rank):03d}.summary.json"


def batch_done_path(round_path: str | Path, batch_id: int, rank: int) -> Path:
    return Path(round_path) / f"rollout_batch_{batch_id:06d}" / f"rank_{int(rank):03d}.done"


def batch_aggregate_path(round_path: str | Path, batch_id: int) -> Path:
    return Path(round_path) / f"rollout_batch_{batch_id:06d}" / "aggregate_success.json"


def live_batch_accepted_evo_qa_path(round_path: str | Path, batch_id: int) -> Path:
    return Path(round_path) / f"rollout_batch_{batch_id:06d}" / "live_accepted_evo_qa.jsonl"


def live_batch_accepted_trajectories_path(round_path: str | Path, batch_id: int) -> Path:
    return Path(round_path) / f"rollout_batch_{batch_id:06d}" / "live_accepted_trajectories.jsonl"


def live_batch_accepted_vl2a_path(round_path: str | Path, batch_id: int) -> Path:
    return Path(round_path) / f"rollout_batch_{batch_id:06d}" / "live_accepted_vl2a.jsonl"


def live_batch_accepted_va2l_path(round_path: str | Path, batch_id: int) -> Path:
    return Path(round_path) / f"rollout_batch_{batch_id:06d}" / "live_accepted_va2l.jsonl"


def accepted_vl2a_path(round_path: str | Path) -> Path:
    return Path(round_path) / "accepted_vl2a.jsonl"


def accepted_va2l_path(round_path: str | Path) -> Path:
    return Path(round_path) / "accepted_va2l.jsonl"


def read_json(path: str | Path) -> JsonDict:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_queue_state(round_path: str | Path) -> JsonDict:
    round_path = Path(round_path)
    state = read_json(pending_success_state_path(round_path))
    total = count_jsonl(pending_success_path(round_path))
    head = int(state.get("queue_head", 0) or 0)
    head = max(0, min(head, total))
    next_update_id = int(state.get("next_update_id", 0) or 0)
    state.update(
        {
            "queue_head": head,
            "queue_total": total,
            "queue_size": total - head,
            "next_update_id": next_update_id,
        }
    )
    return state


def write_queue_state(round_path: str | Path, state: JsonDict) -> JsonDict:
    round_path = Path(round_path)
    total = count_jsonl(pending_success_path(round_path))
    head = int(state.get("queue_head", 0) or 0)
    state = dict(state)
    state.update(
        {
            "queue_head": max(0, min(head, total)),
            "queue_total": total,
            "queue_size": total - max(0, min(head, total)),
            "updated_at": utc_timestamp(),
        }
    )
    write_json_locked(pending_success_state_path(round_path), state)
    return state


def print_queue_state(stage: str, round_path: str | Path, state: Optional[JsonDict] = None, **extra: Any) -> JsonDict:
    state = read_queue_state(round_path) if state is None else dict(state)
    extra_text = " ".join(f"{key}={value}" for key, value in extra.items())
    if extra_text:
        extra_text = " " + extra_text
    print(
        "[evo_utils][queue] "
        f"stage={stage} pool_size={state.get('queue_size', 0)} "
        f"total={state.get('queue_total', 0)} head={state.get('queue_head', 0)} "
        f"next_update={state.get('next_update_id', 0)}"
        f"{extra_text}",
        flush=True,
    )
    return state


def print_queue_pool_change(stage: str, before: JsonDict, after: JsonDict, **extra: Any) -> None:
    before_size = int(before.get("queue_size", 0) or 0)
    after_size = int(after.get("queue_size", 0) or 0)
    extra_text = " ".join(f"{key}={value}" for key, value in extra.items())
    if extra_text:
        extra_text = " " + extra_text
    print(
        "[evo_utils][queue_pool] "
        f"stage={stage} pool_size_before={before_size} "
        f"pool_size_after={after_size} pool_size_delta={after_size - before_size} "
        f"total_before={before.get('queue_total', 0)} total_after={after.get('queue_total', 0)} "
        f"head_before={before.get('queue_head', 0)} head_after={after.get('queue_head', 0)}"
        f"{extra_text}",
        flush=True,
    )


def read_jsonl_slice(path: str | Path, start: int, count: int) -> List[JsonDict]:
    rows: List[JsonDict] = []
    if count <= 0:
        return rows
    for idx, row in enumerate(read_jsonl(path)):
        if idx < start:
            continue
        if len(rows) >= count:
            break
        rows.append(row)
    return rows


def write_rank_success_shard(round_path: str | Path, batch_id: int, rank: int, rows: Sequence[JsonDict], summary: JsonDict) -> None:
    round_path = Path(round_path)
    success_path = batch_success_path(round_path, batch_id, rank)
    summary_path = batch_summary_path(round_path, batch_id, rank)
    done_path = batch_done_path(round_path, batch_id, rank)
    atomic_write_jsonl(success_path, rows)
    atomic_write_json(summary_path, summary)
    done_path.write_text(json.dumps(summary, ensure_ascii=False) + "\n", encoding="utf-8")


def success_record_to_queue_rows(record: JsonDict) -> List[JsonDict]:
    """Flatten one successful trajectory record into trainable evo QA queue rows."""

    queue_rows: List[JsonDict] = []
    base = {
        "round_id": record.get("round_id"),
        "batch_id": record.get("batch_id"),
        "rank": record.get("rank"),
        "trajectory_id": record.get("trajectory_id"),
        "path_id": record.get("path_id"),
        "checkpoint": record.get("checkpoint"),
        "checkpoint_version": record.get("checkpoint_version"),
        "checkpoint_generation": record.get("checkpoint_generation"),
        "accepted": record.get("accepted"),
        "vl2a_accepted": record.get("vl2a_accepted"),
        "va2l_accepted": record.get("va2l_accepted"),
        "sr": record.get("sr", record.get("success")),
        "osr": record.get("osr"),
        "spl": record.get("spl"),
        "steps": record.get("steps"),
        "created_at": utc_timestamp(),
    }

    def add_sample(sample_type: str, sample: JsonDict, sample_index: int) -> None:
        sample = copy.deepcopy(sample)
        meta = sample.setdefault("meta", {})
        if isinstance(meta, dict):
            meta.update(
                {
                    "evo_queue_type": sample_type,
                    "source_round_id": record.get("round_id"),
                    "source_batch_id": record.get("batch_id"),
                    "source_rank": record.get("rank"),
                    "source_trajectory_id": record.get("trajectory_id"),
                    "source_path_id": record.get("path_id"),
                    "source_checkpoint": record.get("checkpoint"),
                    "source_checkpoint_version": record.get("checkpoint_version"),
                    "source_checkpoint_generation": record.get("checkpoint_generation"),
                    "source_sample_index": sample_index,
                    "source_sr": record.get("sr", record.get("success")),
                    "source_osr": record.get("osr"),
                    "source_spl": record.get("spl"),
                    "source_vl2a_accepted": record.get("vl2a_accepted"),
                    "source_va2l_accepted": record.get("va2l_accepted"),
                }
            )
        queue_rows.append({**base, "sample_type": sample_type, "sample_index": sample_index, "sample": sample})

    for idx, sample in enumerate(record.get("vl2a_samples") or []):
        if isinstance(sample, dict):
            add_sample("vl2a", sample, idx)
    va2l_sample = record.get("va2l_sample")
    if isinstance(va2l_sample, dict):
        add_sample("va2l", va2l_sample, 0)
    return queue_rows


def append_live_success_outputs(round_path: str | Path, batch_id: int, record: JsonDict) -> JsonDict:
    """Append one accepted trajectory's trainable rows to batch-local live files."""

    round_path = Path(round_path)
    queue_rows = success_record_to_queue_rows(record)
    vl2a_samples = [
        copy.deepcopy(row["sample"])
        for row in queue_rows
        if row.get("sample_type") == "vl2a" and isinstance(row.get("sample"), dict)
    ]
    va2l_samples = [
        copy.deepcopy(row["sample"])
        for row in queue_rows
        if row.get("sample_type") == "va2l" and isinstance(row.get("sample"), dict)
    ]
    trajectory_row = {k: v for k, v in record.items() if k not in {"vl2a_samples", "va2l_sample"}}

    counts = {
        "live_accepted_evo_qa": append_jsonl_many(live_batch_accepted_evo_qa_path(round_path, batch_id), queue_rows)
        if queue_rows
        else 0,
        "live_accepted_trajectories": append_jsonl_many(
            live_batch_accepted_trajectories_path(round_path, batch_id),
            [trajectory_row],
        ),
        "live_accepted_vl2a": append_jsonl_many(live_batch_accepted_vl2a_path(round_path, batch_id), vl2a_samples)
        if vl2a_samples
        else 0,
        "live_accepted_va2l": append_jsonl_many(live_batch_accepted_va2l_path(round_path, batch_id), va2l_samples)
        if va2l_samples
        else 0,
    }
    counts["paths"] = {
        "live_accepted_evo_qa_jsonl": str(live_batch_accepted_evo_qa_path(round_path, batch_id)),
        "live_accepted_trajectories_jsonl": str(live_batch_accepted_trajectories_path(round_path, batch_id)),
        "live_accepted_vl2a_jsonl": str(live_batch_accepted_vl2a_path(round_path, batch_id)),
        "live_accepted_va2l_jsonl": str(live_batch_accepted_va2l_path(round_path, batch_id)),
    }
    return counts


def aggregate_batch_success(
    round_path: str | Path,
    batch_id: int,
    keep_shards: bool = False,
    expected_ranks: int = 0,
) -> JsonDict:
    round_path = Path(round_path)
    aggregate_path = batch_aggregate_path(round_path, batch_id)
    if aggregate_path.exists():
        return read_json(aggregate_path)

    batch_path = round_path / f"rollout_batch_{batch_id:06d}"
    if expected_ranks:
        missing = []
        for rank in range(expected_ranks):
            if not batch_done_path(round_path, batch_id, rank).exists():
                missing.append(rank)
        if missing:
            missing_text = ", ".join(f"{rank:03d}" for rank in missing)
            raise RuntimeError(
                f"rollout batch {batch_id} is incomplete: expected {expected_ranks} rank shards, "
                f"missing done files for rank(s) {missing_text} under {batch_path}"
            )

    success_paths = sorted(batch_path.glob("rank_*.success.jsonl"))
    success_records: List[JsonDict] = []
    queue_rows: List[JsonDict] = []
    rank_summaries: List[JsonDict] = []
    for path in success_paths:
        for record in read_jsonl(path):
            success_records.append(record)
            queue_rows.extend(success_record_to_queue_rows(record))
        summary_path = path.with_name(path.name.replace(".success.jsonl", ".summary.json"))
        if summary_path.exists():
            rank_summaries.append(read_json(summary_path))

    before_state = read_queue_state(round_path)
    appended = append_jsonl_many(pending_success_path(round_path), queue_rows) if queue_rows else 0
    state = read_queue_state(round_path)
    state["successful_trajectories_total"] = int(state.get("successful_trajectories_total", 0) or 0) + len(success_records)
    state = write_queue_state(round_path, state)
    print_queue_pool_change("rollout_append", before_state, state, batch_id=batch_id, appended=appended)
    accepted_vl2a_trajectories = sum(int(summary.get("accepted_vl2a_trajectories", 0) or 0) for summary in rank_summaries)
    accepted_va2l_trajectories = sum(int(summary.get("accepted_va2l_trajectories", 0) or 0) for summary in rank_summaries)
    accepted_vl2a_samples = sum(int(summary.get("accepted_vl2a_samples", 0) or 0) for summary in rank_summaries)
    accepted_va2l_samples = sum(int(summary.get("accepted_va2l_samples", 0) or 0) for summary in rank_summaries)
    summary = {
        "round_id": int(state.get("round_id", 0) or 0),
        "batch_id": batch_id,
        "success_shards": len(success_paths),
        "successful_trajectories": len(success_records),
        "success_rows": len(success_records),
        "evo_qa_rows": len(queue_rows),
        "accepted_vl2a_trajectories": accepted_vl2a_trajectories,
        "accepted_va2l_trajectories": accepted_va2l_trajectories,
        "accepted_vl2a_samples": accepted_vl2a_samples,
        "accepted_va2l_samples": accepted_va2l_samples,
        "appended_to_pending": appended,
        "pending_success_jsonl": str(pending_success_path(round_path)),
        "rank_summaries": rank_summaries,
        "created_at": utc_timestamp(),
    }
    atomic_write_json(aggregate_path, summary)

    if not keep_shards:
        for path in list(success_paths) + sorted(batch_path.glob("rank_*.summary.json")) + sorted(batch_path.glob("rank_*.done")):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    return summary


def count_train_update_samples(round_path: str | Path, filename: str) -> int:
    total = 0
    for path in sorted(Path(round_path).glob(f"train_update_*/{filename}")):
        total += count_jsonl(path)
    return total


def aggregate_round_summaries(round_path: str | Path) -> JsonDict:
    round_path = Path(round_path)
    status = read_json(success_rate_path(round_path))
    queue_state = read_queue_state(round_path)
    accepted = int(queue_state.get("successful_trajectories_total", queue_state.get("queue_total", 0)) or 0)
    failed = count_jsonl(failed_path(round_path))
    candidates = accepted + failed
    return {
        "batches": int(status.get("batches", 0)),
        "candidates": candidates,
        "accepted": accepted,
        "pending_queue_head": int(queue_state.get("queue_head", 0)),
        "pending_queue_size": int(queue_state.get("queue_size", 0)),
        "train_updates": int(queue_state.get("next_update_id", 0)),
        "accepted_vl2a_samples": count_train_update_samples(round_path, "accepted_vl2a.jsonl"),
        "accepted_va2l_samples": count_train_update_samples(round_path, "accepted_va2l.jsonl"),
        "spatial_qa_samples": count_train_update_samples(round_path, "spatial_qa.jsonl"),
        "stop_qa_samples": count_train_update_samples(round_path, "stop_qa.jsonl"),
        "failed": failed,
        "success_rate": (accepted / candidates) if candidates else 0.0,
    }


def next_batch_id(round_path: str | Path) -> int:
    round_path = Path(round_path)
    status = read_json(success_rate_path(round_path))
    try:
        return int(status["next_batch_id"])
    except (KeyError, TypeError, ValueError):
        pass

    ids = []
    for batch in round_path.glob("rollout_batch_*"):
        try:
            ids.append(int(batch.name.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    for path in (failed_path(round_path), pending_success_path(round_path)):
        if not path.exists():
            continue
        for row in read_jsonl(path):
            values = [row.get("batch_id")]
            if isinstance(row.get("meta"), dict):
                values.append(row["meta"].get("batch_id"))
            for value in values:
                try:
                    ids.append(int(value))
                except (TypeError, ValueError):
                    continue
    return (max(ids) + 1) if ids else 0


def write_round_success_rate(round_path: str | Path, round_id: int, last_batch_id: Optional[int] = None) -> JsonDict:
    round_path = Path(round_path)
    path = success_rate_path(round_path)
    status = read_json(path)
    queue_state = read_queue_state(round_path)
    accepted = int(queue_state.get("successful_trajectories_total", queue_state.get("queue_total", 0)) or 0)
    failed = count_jsonl(failed_path(round_path))
    candidates = accepted + failed
    batches = int(status.get("batches", 0) or 0)
    next_id = int(status.get("next_batch_id", 0) or 0)
    if last_batch_id is not None:
        batches = max(batches, int(last_batch_id) + 1)
        next_id = max(next_id, int(last_batch_id) + 1)
    summary: JsonDict = {
        "round_id": round_id,
        "batches": batches,
        "next_batch_id": next_id,
        "candidates": candidates,
        "accepted": accepted,
        "pending_queue_head": int(queue_state.get("queue_head", 0)),
        "pending_queue_size": int(queue_state.get("queue_size", 0)),
        "train_updates": int(queue_state.get("next_update_id", 0)),
        "accepted_vl2a_samples": count_train_update_samples(round_path, "accepted_vl2a.jsonl"),
        "accepted_va2l_samples": count_train_update_samples(round_path, "accepted_va2l.jsonl"),
        "spatial_qa_samples": count_train_update_samples(round_path, "spatial_qa.jsonl"),
        "stop_qa_samples": count_train_update_samples(round_path, "stop_qa.jsonl"),
        "failed": failed,
        "success_rate": (accepted / candidates) if candidates else 0.0,
        "updated_at": utc_timestamp(),
    }
    write_json_locked(path, summary)
    state = read_queue_state(round_path)
    state["round_id"] = round_id
    write_queue_state(round_path, state)
    return summary


def merge_round_shards(round_path: str | Path) -> JsonDict:
    """Compatibility wrapper for the old shard-merge entrypoint."""

    round_path = Path(round_path)
    status = read_json(success_rate_path(round_path))
    round_id = int(status.get("round_id", 0) or 0)
    return write_round_success_rate(round_path, round_id)


def row_with_absolute_images(
    row: JsonDict,
    data_path: str | Path,
    source_name: str,
    meta_source_key: str = "evo_spatial_source",
) -> JsonDict:
    out = copy.deepcopy(row)
    base = Path(data_path)

    def absolutize(value: Any) -> Any:
        if not isinstance(value, str) or not value:
            return value
        path = Path(value)
        return str(path if path.is_absolute() else base / path)

    images = out.get("image")
    if isinstance(images, list):
        out["image"] = [absolutize(img) for img in images]
    elif isinstance(images, str):
        out["image"] = absolutize(images)
    meta = out.get("meta") if isinstance(out.get("meta"), dict) else {}
    meta = dict(meta)
    meta[meta_source_key] = source_name
    out["meta"] = meta
    return out


def reservoir_sample_jsonl(path: str | Path, k: int, rng: random.Random) -> List[JsonDict]:
    if k <= 0:
        return []
    sample: List[JsonDict] = []
    seen = 0
    for row in read_jsonl(path):
        seen += 1
        if len(sample) < k:
            sample.append(row)
            continue
        j = rng.randrange(seen)
        if j < k:
            sample[j] = row
    return sample


def sample_annotation_rows(path: str | Path, k: int, rng: random.Random) -> List[JsonDict]:
    path = Path(path)
    if k <= 0 or not path.exists():
        return []
    if path.suffix == ".jsonl":
        return reservoir_sample_jsonl(path, k, rng)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data if isinstance(data, list) else [data]
    if len(rows) <= k:
        return list(rows)
    return rng.sample(rows, k)


def resolve_spatial_qa_jsonls(repo_root: str | Path, sources: Sequence[str] | None) -> List[Path]:
    repo_root = Path(repo_root)
    values = [source.strip() for source in (sources or []) if source.strip()] or list(DEFAULT_SPATIAL_QA_JSONLS)
    paths: List[Path] = []
    for value in values:
        resolved = SPATIAL_QA_JSONL_ALIASES.get(value, value)
        path = Path(resolved)
        if not path.is_absolute():
            path = repo_root / path
        paths.append(path)
    return paths


def spatial_qa_data_path(path: str | Path) -> Path:
    path = Path(path)
    configured = DEFAULT_SPATIAL_QA_DATA_PATHS.get(str(path))
    return Path(configured) if configured else path.parent


def spatial_qa_row_id(row: JsonDict) -> Optional[str]:
    value = row.get("id")
    return str(value) if value is not None else None


def load_trained_spatial_qa_ids(round_path: str | Path) -> set[str]:
    ids, _ = load_trained_spatial_qa_ids_with_status(round_path)
    return ids


def load_trained_spatial_qa_ids_with_status(round_path: str | Path) -> tuple[set[str], JsonDict]:
    path = Path(round_path) 
    path = path.parent / "trained_spatial_qa.jsonl"
    status: JsonDict = {
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

    ids: set[str] = set()
    for row in read_jsonl(path):
        row_id = spatial_qa_row_id(row)
        if row_id:
            ids.add(row_id)
    status["ids"] = len(ids)
    return ids, status


def load_spatial_qa_pool(repo_root: str | Path, sources: Sequence[str] | None) -> List[tuple[JsonDict, Path, str]]:
    paths = resolve_spatial_qa_jsonls(repo_root, sources)
    cache_key = tuple(str(path) for path in paths)
    cached = _SPATIAL_QA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rows: List[tuple[JsonDict, Path, str]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"spatial QA jsonl does not exist: {path}")
        source_name = path.stem
        data_path = spatial_qa_data_path(path)
        for row in read_jsonl(path):
            rows.append((row_with_absolute_images(row, data_path, source_name), data_path, source_name))
    _SPATIAL_QA_CACHE[cache_key] = rows
    return rows


def shuffled_spatial_qa_pool(
    repo_root: str | Path,
    sources: Sequence[str] | None,
    seed: int,
) -> List[tuple[JsonDict, Path, str]]:
    paths = resolve_spatial_qa_jsonls(repo_root, sources)
    cache_key = tuple(str(path) for path in paths) + (str(seed),)
    cached = _SPATIAL_QA_SHUFFLED_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rows = list(load_spatial_qa_pool(repo_root, sources))
    random.Random(seed).shuffle(rows)
    _SPATIAL_QA_SHUFFLED_CACHE[cache_key] = rows
    return rows


def build_prefiltered_spatial_qa_pool(
    repo_root: str | Path,
    dataset_names: Sequence[str] | None,
    seed: int,
    processed_ids: Optional[set[str]] = None,
) -> tuple[List[JsonDict], JsonDict]:
    spatial_sources = [source.strip() for source in (dataset_names or []) if source.strip()] or list(DEFAULT_SPATIAL_QA_JSONLS)
    pool = shuffled_spatial_qa_pool(repo_root, spatial_sources, seed)
    processed_ids = processed_ids or set()
    original = len(pool)
    if processed_ids:
        pool = [entry for entry in pool if spatial_qa_row_id(entry[0]) not in processed_ids]
    used_id_deduped = original - len(pool)

    unique_pool: List[tuple[JsonDict, Path, str]] = []
    seen_pool_ids: set[str] = set()
    duplicate_pool_ids = 0
    for entry in pool:
        row_id = spatial_qa_row_id(entry[0])
        if row_id and row_id in seen_pool_ids:
            duplicate_pool_ids += 1
            continue
        if row_id:
            seen_pool_ids.add(row_id)
        unique_pool.append(entry)

    rows = [row_with_absolute_images(row, data_path, source_name) for row, data_path, source_name in unique_pool]
    meta: JsonDict = {
        "spatial_sources": spatial_sources,
        "original": original,
        "used_id_deduped": used_id_deduped,
        "duplicate_pool_ids": duplicate_pool_ids,
        "remaining": len(rows),
        "trained_spatial_ids": len(processed_ids),
    }
    return rows, meta


def sample_spatial_qa_rows(
    repo_root: str | Path,
    dataset_names: Sequence[str] | None,
    total_count: int,
    seed: int,
    offset: int = 0,
    processed_ids: Optional[set[str]] = None,
    print_dedup: bool = False,
    prefiltered_pool: Optional[Sequence[JsonDict]] = None,
) -> List[JsonDict]:
    if total_count <= 0:
        return []
    if prefiltered_pool is None:
        pool, meta = build_prefiltered_spatial_qa_pool(repo_root, dataset_names, seed, processed_ids=processed_ids)
        if print_dedup:
            print(
                "[evo_utils][dedup] "
                f"data=spatial original={meta['original']} used_id_deduped={meta['used_id_deduped']} "
                f"duplicate_pool_ids={meta['duplicate_pool_ids']} remaining={meta['remaining']} "
                f"trained_spatial_ids={meta['trained_spatial_ids']}",
                flush=True,
            )
    else:
        pool = list(prefiltered_pool)
    if not pool:
        return []
    start = max(0, offset)
    indices = [(start + idx) % len(pool) for idx in range(min(total_count, len(pool)))]
    selected = [copy.deepcopy(pool[idx]) for idx in indices]
    if print_dedup:
        selected_ids = [spatial_qa_row_id(row) for row in selected if spatial_qa_row_id(row)]
        print(
            "[evo_utils][spatial_qa_sample] "
            f"requested={total_count} selected={len(selected)} offset={offset} "
            f"selected_unique_ids={len(set(selected_ids))} "
            f"selected_duplicate_ids={len(selected_ids) - len(set(selected_ids))}",
            flush=True,
        )
    return selected


def resolve_stop_qa_jsonls(repo_root: str | Path, sources: Sequence[str] | None) -> List[Path]:
    repo_root = Path(repo_root)
    values = [source.strip() for source in (sources or []) if source.strip()] or list(DEFAULT_STOP_QA_JSONLS)
    paths: List[Path] = []
    for value in values:
        resolved = STOP_QA_JSONL_ALIASES.get(value, value)
        path = Path(resolved)
        if not path.is_absolute():
            path = repo_root / path
        paths.append(path)
    return paths


def stop_qa_data_path(path: str | Path) -> Path:
    path = Path(path)
    configured = DEFAULT_STOP_QA_DATA_PATHS.get(str(path))
    return Path(configured) if configured else path.parent


def stop_qa_row_id(row: JsonDict) -> Optional[str]:
    value = row.get("id")
    return str(value) if value is not None else None


def load_trained_stop_qa_ids(round_path: str | Path) -> set[str]:
    ids, _ = load_trained_stop_qa_ids_with_status(round_path)
    return ids


def load_trained_stop_qa_ids_with_status(round_path: str | Path) -> tuple[set[str], JsonDict]:
    path = Path(round_path)
    path = path.parent / "trained_stop_qa.jsonl"
    status: JsonDict = {
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

    ids: set[str] = set()
    for row in read_jsonl(path):
        row_id = stop_qa_row_id(row)
        if row_id:
            ids.add(row_id)
    status["ids"] = len(ids)
    return ids, status


def load_stop_qa_pool(repo_root: str | Path, sources: Sequence[str] | None) -> List[tuple[JsonDict, Path, str]]:
    paths = resolve_stop_qa_jsonls(repo_root, sources)
    cache_key = tuple(str(path) for path in paths)
    cached = _STOP_QA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rows: List[tuple[JsonDict, Path, str]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"stop QA jsonl does not exist: {path}")
        source_name = path.stem
        data_path = stop_qa_data_path(path)
        for row in read_jsonl(path):
            rows.append((row_with_absolute_images(row, data_path, source_name, "evo_stop_source"), data_path, source_name))
    _STOP_QA_CACHE[cache_key] = rows
    return rows


def shuffled_stop_qa_pool(
    repo_root: str | Path,
    sources: Sequence[str] | None,
    seed: int,
) -> List[tuple[JsonDict, Path, str]]:
    paths = resolve_stop_qa_jsonls(repo_root, sources)
    cache_key = tuple(str(path) for path in paths) + (str(seed),)
    cached = _STOP_QA_SHUFFLED_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rows = list(load_stop_qa_pool(repo_root, sources))
    random.Random(seed).shuffle(rows)
    _STOP_QA_SHUFFLED_CACHE[cache_key] = rows
    return rows


def build_prefiltered_stop_qa_pool(
    repo_root: str | Path,
    dataset_names: Sequence[str] | None,
    seed: int,
    processed_ids: Optional[set[str]] = None,
) -> tuple[List[JsonDict], JsonDict]:
    stop_sources = [source.strip() for source in (dataset_names or []) if source.strip()] or list(DEFAULT_STOP_QA_JSONLS)
    pool = shuffled_stop_qa_pool(repo_root, stop_sources, seed)
    processed_ids = processed_ids or set()
    original = len(pool)
    if processed_ids:
        pool = [entry for entry in pool if stop_qa_row_id(entry[0]) not in processed_ids]
    used_id_deduped = original - len(pool)

    unique_pool: List[tuple[JsonDict, Path, str]] = []
    seen_pool_ids: set[str] = set()
    duplicate_pool_ids = 0
    for entry in pool:
        row_id = stop_qa_row_id(entry[0])
        if row_id and row_id in seen_pool_ids:
            duplicate_pool_ids += 1
            continue
        if row_id:
            seen_pool_ids.add(row_id)
        unique_pool.append(entry)

    rows = [row_with_absolute_images(row, data_path, source_name, "evo_stop_source") for row, data_path, source_name in unique_pool]
    meta: JsonDict = {
        "stop_sources": stop_sources,
        "original": original,
        "used_id_deduped": used_id_deduped,
        "duplicate_pool_ids": duplicate_pool_ids,
        "remaining": len(rows),
        "trained_stop_ids": len(processed_ids),
    }
    return rows, meta


def sample_stop_qa_rows(
    repo_root: str | Path,
    dataset_names: Sequence[str] | None,
    total_count: int,
    seed: int,
    offset: int = 0,
    processed_ids: Optional[set[str]] = None,
    print_dedup: bool = False,
    prefiltered_pool: Optional[Sequence[JsonDict]] = None,
) -> List[JsonDict]:
    if total_count <= 0:
        return []
    if prefiltered_pool is None:
        pool, meta = build_prefiltered_stop_qa_pool(repo_root, dataset_names, seed, processed_ids=processed_ids)
        if print_dedup:
            print(
                "[evo_utils][dedup] "
                f"data=stop original={meta['original']} used_id_deduped={meta['used_id_deduped']} "
                f"duplicate_pool_ids={meta['duplicate_pool_ids']} remaining={meta['remaining']} "
                f"trained_stop_ids={meta['trained_stop_ids']}",
                flush=True,
            )
    else:
        pool = list(prefiltered_pool)
    if not pool:
        return []
    start = max(0, offset)
    indices = [(start + idx) % len(pool) for idx in range(min(total_count, len(pool)))]
    selected = [copy.deepcopy(pool[idx]) for idx in indices]
    if print_dedup:
        selected_ids = [stop_qa_row_id(row) for row in selected if stop_qa_row_id(row)]
        print(
            "[evo_utils][stop_qa_sample] "
            f"requested={total_count} selected={len(selected)} offset={offset} "
            f"selected_unique_ids={len(set(selected_ids))} "
            f"selected_duplicate_ids={len(selected_ids) - len(set(selected_ids))}",
            flush=True,
        )
    return selected


def build_train_update_chunk(
    round_path: str | Path,
    accepted_target: int,
    spatial_qa_count: int,
    spatial_qa_datasets: Sequence[str] | None,
    seed: int,
    repo_root: str | Path,
    checkpoint: str = "",
    checkpoint_version: str = "",
    allow_partial: bool = False,
    prefiltered_spatial_pool: Optional[Sequence[JsonDict]] = None,
    stop_qa_count: int = 0,
    stop_qa_datasets: Sequence[str] | None = None,
    stop_qa_seed: Optional[int] = None,
    prefiltered_stop_pool: Optional[Sequence[JsonDict]] = None,
) -> Optional[JsonDict]:
    round_path = Path(round_path)
    state = read_queue_state(round_path)
    queue_size = int(state.get("queue_size", 0))
    if queue_size <= 0:
        return None
    if queue_size < accepted_target and not allow_partial:
        return None

    update_id = int(state.get("next_update_id", 0))
    queue_head = int(state.get("queue_head", 0))
    target_count = min(int(accepted_target), queue_size) if allow_partial else int(accepted_target)
    before_consume_state = dict(state)
    rows = read_jsonl_slice(pending_success_path(round_path), queue_head, target_count)
    if len(rows) < target_count:
        return None

    update_dir = train_update_dir(round_path, update_id)
    update_dir.mkdir(parents=True, exist_ok=True)
    trajectory_rows: List[JsonDict] = []
    seen_trajectories = set()
    vl2a_samples: List[JsonDict] = []
    va2l_samples: List[JsonDict] = []
    for row in rows:
        sample_type = row.get("sample_type")
        sample = copy.deepcopy(row.get("sample") or {})
        if not isinstance(sample, dict):
            continue
        sample.setdefault("meta", {})
        if isinstance(sample["meta"], dict):
            sample["meta"].update({"consumed_by_update_id": update_id})
        if sample_type == "vl2a":
            vl2a_samples.append(sample)
        elif sample_type == "va2l":
            va2l_samples.append(sample)

        traj_key = (row.get("trajectory_id"), row.get("path_id"))
        if traj_key not in seen_trajectories:
            seen_trajectories.add(traj_key)
            trajectory_rows.append({k: v for k, v in row.items() if k not in {"sample"}})

    spatial_sources = [source.strip() for source in (spatial_qa_datasets or []) if source.strip()] or list(DEFAULT_SPATIAL_QA_JSONLS)
    spatial_rows = sample_spatial_qa_rows(
        repo_root,
        spatial_sources,
        spatial_qa_count,
        seed,
        offset=update_id * spatial_qa_count,
        print_dedup=False,
        prefiltered_pool=prefiltered_spatial_pool,
    )
    stop_sources = [source.strip() for source in (stop_qa_datasets or []) if source.strip()] or (
        list(DEFAULT_STOP_QA_JSONLS) if stop_qa_count > 0 else []
    )
    stop_rows = sample_stop_qa_rows(
        repo_root,
        stop_sources,
        stop_qa_count,
        seed if stop_qa_seed is None else stop_qa_seed,
        offset=update_id * stop_qa_count,
        print_dedup=False,
        prefiltered_pool=prefiltered_stop_pool,
    ) if stop_qa_count > 0 else []

    accepted_evo_qa = update_dir / "accepted_evo_qa.jsonl"
    accepted_traj_path = update_dir / "accepted_trajectories.jsonl"
    accepted_vl2a = update_dir / "accepted_vl2a.jsonl"
    accepted_va2l = update_dir / "accepted_va2l.jsonl"
    spatial_qa = update_dir / "spatial_qa.jsonl"
    stop_qa = update_dir / "stop_qa.jsonl"
    atomic_write_jsonl(accepted_evo_qa, rows)
    atomic_write_jsonl(accepted_traj_path, trajectory_rows)
    atomic_write_jsonl(accepted_vl2a, vl2a_samples)
    atomic_write_jsonl(accepted_va2l, va2l_samples)
    atomic_write_jsonl(spatial_qa, spatial_rows)
    atomic_write_jsonl(stop_qa, stop_rows)

    next_state = dict(state)
    next_state["queue_head"] = queue_head + len(rows)
    next_state["next_update_id"] = update_id + 1
    next_state = write_queue_state(round_path, next_state)
    print_queue_pool_change(
        "train_consume",
        before_consume_state,
        next_state,
        update_id=update_id,
        target_count=target_count,
        consumed=len(rows),
        retained_after_consume=next_state.get("queue_size", 0),
    )

    manifest = {
        "round_id": int(read_json(success_rate_path(round_path)).get("round_id", 0) or 0),
        "update_id": update_id,
        "checkpoint": checkpoint,
        "checkpoint_version": checkpoint_version or checkpoint,
        "queue_head_start": queue_head,
        "queue_head_end": queue_head + len(rows),
        "accepted_target": accepted_target,
        "partial_update": len(rows) < accepted_target,
        "accepted_evo_qa": len(rows),
        "accepted_trajectories": len(trajectory_rows),
        "accepted_vl2a": len(vl2a_samples),
        "accepted_va2l": len(va2l_samples),
        "spatial_qa": len(spatial_rows),
        "stop_qa": len(stop_rows),
        "total_train_qa": len(vl2a_samples) + len(va2l_samples) + len(spatial_rows) + len(stop_rows),
        "spatial_qa_sources": spatial_sources,
        "stop_qa_sources": stop_sources,
        "paths": {
            "accepted_evo_qa_jsonl": str(accepted_evo_qa),
            "accepted_trajectories_jsonl": str(accepted_traj_path),
            "accepted_vl2a_jsonl": str(accepted_vl2a),
            "accepted_va2l_jsonl": str(accepted_va2l),
            "spatial_qa_jsonl": str(spatial_qa),
            "stop_qa_jsonl": str(stop_qa),
        },
        "created_at": utc_timestamp(),
    }
    atomic_write_json(update_dir / "train_manifest.json", manifest)
    return {"update_dir": str(update_dir), "manifest": manifest}


def rotate_checkpoints(
    runs_dir: str | Path,
    next_ckpt: str | Path | None = None,
    metadata: Optional[JsonDict] = None,
) -> None:
    """Replace ``current_ckpt`` with ``next_ckpt_tmp`` after training succeeds."""

    runs_dir = Path(runs_dir)
    current = runs_dir / "current_ckpt"
    next_path = Path(next_ckpt) if next_ckpt else runs_dir / "next_ckpt_tmp"

    if not next_path.exists():
        raise FileNotFoundError(f"next checkpoint does not exist: {next_path}")

    if current.exists() or current.is_symlink():
        if current.is_dir() and not current.is_symlink():
            shutil.rmtree(current)
        else:
            current.unlink()
    next_path.rename(current)
    state: JsonDict = {"path": str(current), "updated_at": utc_timestamp()}
    if metadata:
        state.update(metadata)
    atomic_write_json(runs_dir / "current_ckpt.json", state)


def resolve_current_ckpt(runs_dir: str | Path, fallback: Optional[str] = None) -> str:
    runs_dir = Path(runs_dir)
    current = runs_dir / "current_ckpt"
    if current.exists():
        return str(current)
    pointer = runs_dir / "current_ckpt.txt"
    if pointer.exists():
        value = pointer.read_text(encoding="utf-8").strip()
        if value:
            return value
    if fallback:
        return fallback
    raise FileNotFoundError(f"No current checkpoint under {runs_dir}; pass --initial-ckpt")
