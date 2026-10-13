#!/usr/bin/env python3
"""Eval-like evo rollout for RSI-Nav.

This script is intentionally self-contained under ``qwen3vl/evo``. It reuses the
model loading and constrained action decoding helpers from the existing eval
script, but does not modify that script.

Each process/rank handles its own strided candidate slice. Accepted VL2A and
VA2L samples are appended immediately to batch-local live JSONL files, then the
rank writes its final success shard plus summary. The controller aggregates those
shards into durable round-level queues after the torchrun batch completes.

Input format is deliberately permissive. The usual pair is:
  - --episodes-jsonl: Habitat episode metadata JSONL
  - --trajectory-jsonl: trajectory annotation JSONL with id/video/actions
Rows are matched by id/path_id/episode_id/trajectory_id.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import itertools
import json
import math
import os
import random
import re
import signal
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

def resolve_attn_implementation() -> str:
    return os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2")


os.environ["ATTN_IMPLEMENTATION"] = resolve_attn_implementation()
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("GLOG_minloglevel", "2")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
    logging as transformers_logging,
)
from qwenvl.train.fa2_compat import maybe_patch_qwen35_flash_attention

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - progress bar is optional at runtime.
    tqdm = None

import habitat
from habitat import Env
from habitat.config.default import get_config as get_habitat_config
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.datasets.utils import VocabDict
from habitat.datasets.vln.r2r_vln_dataset import VLNDatasetV1
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import InstructionData, VLNEpisode
transformers_logging.set_verbosity_error()


from evo_utils import (
    append_jsonl,
    append_live_success_outputs,
    append_timing,
    batch_dir,
    failed_path,
    read_jsonl,
    round_dir,
    utc_timestamp,
    write_rank_success_shard,
)


ACTION_ID_TO_TEXT = {0: "stop", 1: "forward", 2: "left", 3: "right"}
ACTION_TEXT_TO_ID = OrderedDict({"stop": [0], "forward": [1], "left": [2], "right": [3]})
CONJUNCTIONS = [
    "you can see ",
    "in front of you is ",
    "there is ",
    "you can spot ",
    "you are toward the ",
    "ahead of you is ",
    "in your sight is ",
]
NAV_PROMPT = (
    "You are an autonomous navigation assistant. Your task is to <instruction> "
    "Devise an action sequence to follow the instruction using the four actions: "
    "TURN LEFT (left) or TURN RIGHT (right) by 15 degrees, "
    "MOVE FORWARD (forward) by 25 centimeters, or STOP (stop)."
)
VA2L_SUFFIX = "Could you give me its corresponding navigation instruction in details?"


@contextlib.contextmanager
def suppress_stderr(enabled: bool = True):
    if not enabled:
        yield
        return
    sys.stderr.flush()
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)


@contextlib.contextmanager
def suppress_stdout(enabled: bool = True):
    if not enabled:
        yield
        return
    sys.stdout.flush()
    saved_stdout = os.dup(1)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 1)
            yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)


def patch_habitat_sim_semantics() -> None:
    """Disable semantic scene loading for RGB/depth-only evo rollout."""
    original_create_sim_config = HabitatSim.create_sim_config

    if getattr(original_create_sim_config, "_evo_semantic_patch", False):
        return

    def create_sim_config_without_semantics(self, sensor_suite):
        sim_config = original_create_sim_config(self, sensor_suite)
        sim_cfg = sim_config.sim_cfg

        # Different habitat-sim builds expose different semantic-related knobs.
        # Set every known one defensively; missing attributes are ignored.
        semantic_overrides = {
            "semantic_scene_id": "",
            "semantic_descriptor_filename": "",
            "semantic_descriptor_file": "",
            "semantic_asset_handle": "",
            "load_semantic_mesh": False,
            "enable_semantic_textures": False,
            "use_semantic_textures": False,
            "force_separate_semantic_scene_graph": False,
        }
        applied = {}
        for name, value in semantic_overrides.items():
            if hasattr(sim_cfg, name):
                try:
                    setattr(sim_cfg, name, value)
                    applied[name] = getattr(sim_cfg, name)
                except Exception as exc:
                    applied[name] = f"<failed: {exc}>"

        return sim_config

    create_sim_config_without_semantics._evo_semantic_patch = True
    HabitatSim.create_sim_config = create_sim_config_without_semantics


patch_habitat_sim_semantics()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one evo rollout batch.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-generation", default="")
    parser.add_argument("--episodes-jsonl", required=True)
    parser.add_argument("--trajectory-jsonl", required=True)
    parser.add_argument("--habitat-config-path", required=True)
    parser.add_argument("--runs-dir", default=str(REPO_ROOT / "evo_runs"))
    parser.add_argument("--round-id", type=int, default=0)
    parser.add_argument("--batch-id", type=int, default=0)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--scene-root", default="")
    parser.add_argument("--max-candidates-per-rank", type=int, default=64)
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument("--retry-jsonl", default=None)
    parser.add_argument(
        "--disable-live-success-write",
        action="store_true",
        default=False,
        help=(
            "Disable immediate batch-local live_accepted_*.jsonl writes. "
            "By default every accepted trajectory appends VL2A/VA2L rows during rollout."
        ),
    )
    parser.add_argument("--accepted-target", type=int, default=1024)
    parser.add_argument("--num-future-steps", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--va2l-num-frames", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--vl2a-min-sr", type=float, default=1.0)
    parser.add_argument("--vl2a-min-spl", type=float, default=0.9)
    parser.add_argument("--va2l-min-sr", type=float, default=1.0)
    parser.add_argument("--va2l-min-spl", type=float, default=1.0)
    parser.add_argument("--metric-success-distance", type=float, default=3.0)
    parser.add_argument(
        "--verify-heartbeat-seconds",
        type=float,
        default=30.0,
        help="Print a minimal per-rank verify heartbeat after this many seconds. 0 disables it.",
    )
    parser.add_argument(
        "--verify-timeout-seconds",
        type=float,
        default=900.0,
        help="Wall-clock timeout per candidate verification. 0 disables it.",
    )
    parser.add_argument("--save-video", action="store_true", default=False)
    parser.add_argument("--model-max-length", type=int, default=16384)
    parser.add_argument("--min-pixels", type=int, default=262144)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--instruction-max-new-tokens", type=int, default=128)
    parser.add_argument("--nav-max-new-tokens", type=int, default=64)
    parser.add_argument("--free-generation", action="store_true", default=False)
    parser.add_argument(
        "--worker-mode",
        choices=["independent", "torchrun"],
        default=os.environ.get("RSI_NAV_EVO_WORKER_MODE", "torchrun"),
        help=(
            "Rollout execution mode. torchrun follows eval/Nav/r2r_multiworker.py: "
            "one process and one full model replica per GPU; rank/world_size "
            "are used for candidate sharding and shard names."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def env_int(name: str, default: Optional[int]) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def fill_dist_args(args: argparse.Namespace) -> argparse.Namespace:
    args.rank = args.rank if args.rank is not None else env_int("RANK", 0)
    args.world_size = args.world_size if args.world_size is not None else env_int("WORLD_SIZE", 1)
    args.local_rank = args.local_rank if args.local_rank is not None else env_int("LOCAL_RANK", args.rank)
    args.gpu = args.gpu if args.gpu is not None else int(args.local_rank or 0)
    return args




def init_torchrun_process_group(args: argparse.Namespace) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
    if int(args.world_size or 1) <= 1:
        return
    # Rollout uses torchrun only as a multi-node process launcher. Candidate
    # sharding uses RANK/WORLD_SIZE from the environment, so no NCCL process
    # group is needed unless explicitly requested for debugging.
    if os.environ.get("RSI_NAV_EVO_INIT_PROCESS_GROUP", "0") not in {"1", "true", "True"}:
        return
    if dist.is_available() and not dist.is_initialized():
        kwargs = {
            "backend": os.environ.get("DIST_BACKEND", "nccl"),
            "init_method": os.environ.get("DIST_URL", "env://"),
            "world_size": int(args.world_size),
            "rank": int(args.rank),
            "timeout": datetime.timedelta(seconds=36000),
        }
        if torch.cuda.is_available() and kwargs["backend"] == "nccl":
            kwargs["device_id"] = torch.device(f"cuda:{int(args.gpu)}")
        try:
            dist.init_process_group(**kwargs)
        except TypeError:
            # Older PyTorch builds do not accept device_id.
            kwargs.pop("device_id", None)
            dist.init_process_group(**kwargs)


def dist_barrier() -> None:
    # Rollout/eval uses torchrun only as a multi-process launcher and for
    # RANK/WORLD_SIZE based data sharding. It does not need GPU collectives;
    # disabling barriers avoids NCCL failures before model work starts.
    if os.environ.get("RSI_NAV_EVO_DISABLE_DIST_BARRIER", "1") in {"1", "true", "True"}:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def candidate_key(row: Dict[str, Any]) -> str:
    for key in ("id", "path_id", "episode_id", "trajectory_id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    raise KeyError(f"Cannot infer candidate id from keys={sorted(row.keys())}")


def load_trajectory_map(
    path: str,
    processed_source_ids: Optional[set[str]] = None,
    print_dedup: bool = False,
) -> Dict[str, Dict[str, Any]]:
    processed_source_ids = processed_source_ids or set()
    original = 0
    deduped = 0
    kept_rows: List[Dict[str, Any]] = []
    for row in read_jsonl(path):
        original += 1
        key = candidate_key(row)
        if key in processed_source_ids:
            deduped += 1
            continue
        kept_rows.append(row)
    if print_dedup:
        print(
            "[evo_rollout][dedup] "
            f"data=evo original={original} deduped={deduped} remaining={len(kept_rows)} "
            f"trained_va2l_ids={len(processed_source_ids)}",
            flush=True,
        )
    return {candidate_key(row): row for row in kept_rows}


def trained_va2l_path(runs_dir: str | Path) -> Path:
    return Path(runs_dir) / "data" / "trained_accepted_va2l.jsonl"


def annotated_trajectory_path(runs_dir: str | Path) -> Path:
    return Path(runs_dir) / "data" / "annotated_trajectory_ids.jsonl"


def append_annotated_trajectory(
    args: argparse.Namespace,
    record_id: str,
    path_id: str,
    accepted: bool,
    failure_reason: str = "",
    metrics: Optional[Dict[str, Any]] = None,
    vl2a_accepted: bool = False,
    va2l_accepted: bool = False,
) -> None:
    row = {
        "id": record_id,
        "trajectory_id": record_id,
        "path_id": path_id,
        "accepted": bool(accepted),
        "vl2a_accepted": bool(vl2a_accepted),
        "va2l_accepted": bool(va2l_accepted),
        "failure_reason": failure_reason,
        "round_id": args.round_id,
        "batch_id": args.batch_id,
        "rank": args.rank,
        "checkpoint_generation": args.checkpoint_generation,
        "created_at": utc_timestamp(),
    }
    if isinstance(metrics, dict):
        row.update(
            {
                "sr": metrics.get("sr", metrics.get("success")),
                "osr": metrics.get("osr"),
                "spl": metrics.get("spl"),
                "steps": metrics.get("steps"),
            }
        )
    append_jsonl(annotated_trajectory_path(args.runs_dir), row)


def va2l_source_id(sample_id: Any) -> Optional[str]:
    if sample_id is None:
        return None
    value = str(sample_id)
    suffix = "_va2l"
    return value[: -len(suffix)] if value.endswith(suffix) else value


def load_trained_va2l_source_ids(runs_dir: str | Path) -> set[str]:
    source_ids, _ = load_trained_va2l_source_ids_with_status(runs_dir)
    return source_ids


def load_trained_va2l_source_ids_with_status(runs_dir: str | Path) -> tuple[set[str], Dict[str, Any]]:
    path = trained_va2l_path(runs_dir)
    status: Dict[str, Any] = {
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
        source_id = va2l_source_id(row.get("id"))
        if source_id:
            source_ids.add(source_id)
    status["ids"] = len(source_ids)
    return source_ids, status


def load_retry_ids(path: Optional[str]) -> Optional[set[str]]:
    if not path:
        return None
    retry_ids = set()
    for row in read_jsonl(path):
        retry_ids.add(str(row.get("trajectory_id") or row.get("id") or row.get("path_id") or row.get("episode_id")))
    return retry_ids


def load_episode_rows(path: str, trajectory_map: Dict[str, Dict[str, Any]], retry_ids: Optional[set[str]]) -> List[Dict[str, Any]]:
    out = []
    for row in read_jsonl(path):
        key = candidate_key(row)
        if key not in trajectory_map:
            continue
        if retry_ids is not None and key not in retry_ids:
            continue
        out.append({"episode": row, "trajectory": trajectory_map[key], "id": key})
    return out


def retry_failure_record(base_record: Dict[str, Any], item: Dict[str, Any], failure_reason: str) -> Dict[str, Any]:
    row = dict(base_record)
    row.update(
        {
            "source_episode": item.get("episode"),
            "source_trajectory": item.get("trajectory"),
            "failure_reason": failure_reason,
        }
    )
    return row


def set_processor_pixels(processor, min_pixels: int, max_pixels: int) -> None:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = min_pixels
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = max_pixels
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        size["shortest_edge"] = min_pixels
        size["longest_edge"] = max_pixels


def load_model_for_eval(model_path: str, rank: int, gpu: int):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", None) or ""
    chosen_attn = os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2")
    maybe_patch_qwen35_flash_attention(model_type, chosen_attn)
    load_kw = dict(
        dtype=torch.bfloat16,
        attn_implementation=chosen_attn,
        device_map={"": gpu},
        trust_remote_code=True,
    )
    if model_type not in {"qwen3_5", "qwen3_5_moe", "qwen3_vl_moe", "qwen3_vl"}:
        raise ValueError(
            f"Unsupported model_type={model_type!r} for model_path={model_path!r}. "
            "Use a Qwen3-VL / Qwen3-VL-MoE / Qwen3.5 multimodal checkpoint."
        )
    model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kw)
    print(
        f"Initialized eval model from {model_path} "
        f"({model.__class__.__name__}, model_type={model_type}, "
        f"attn_implementation={chosen_attn}, rank={rank}, gpu={gpu})",
        flush=True,
    )
    return model


def progress_iter(items: List[Dict[str, Any]], args: argparse.Namespace):
    # Keep stdout readable under torchrun. tqdm writes many carriage-return
    # updates per rank and corrupts interleaved multi-process logs.
    return items


def log_rank_path(args: argparse.Namespace, path_id: str, candidate_index: int, candidates_total: int) -> None:
    print(
        f"[evo_rollout][rank {int(args.rank):03d}] "
        f"candidate {candidate_index}/{candidates_total} path_id={path_id}",
        flush=True,
    )


def log_rank_instruction(args: argparse.Namespace, path_id: str, instruction: str) -> None:
    print(
        f"[evo_rollout][rank {int(args.rank):03d}] "
        f"path_id={path_id} instruction={json.dumps(instruction, ensure_ascii=False)}",
        flush=True,
    )


def log_rank_invalid_instruction(args: argparse.Namespace, path_id: str, instruction: str, reason: str) -> None:
    print(
        f"[evo_rollout][rank {int(args.rank):03d}] "
        f"path_id={path_id} invalid_instruction reason={reason} "
        f"instruction={json.dumps(instruction, ensure_ascii=False)}",
        flush=True,
    )


def log_rank_error(args: argparse.Namespace, path_id: str, error: Exception) -> None:
    print(
        f"[evo_rollout][rank {int(args.rank):03d}] "
        f"path_id={path_id} error={error.__class__.__name__}: {error}",
        flush=True,
    )




def log_rank_verify(args: argparse.Namespace, path_id: str, step_id: int, elapsed_seconds: float) -> None:
    print(
        f"[evo_rollout][rank {int(args.rank):03d}] "
        f"verifying path_id={path_id} steps={step_id} elapsed={elapsed_seconds:.1f}s",
        flush=True,
    )


@contextlib.contextmanager
def candidate_timeout(seconds: float, path_id: str):
    if seconds is None or seconds <= 0:
        yield
        return

    def _raise_timeout(signum, frame):
        raise TimeoutError(f"verification timeout after {seconds:.1f}s for path_id={path_id}")

    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer and old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])

def resolve_path(path: str, root: str = "") -> str:
    if not path:
        return path
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((Path(root) / p).resolve()) if root else str(p)


def normalize_actions(actions: Iterable[Any]) -> List[int]:
    out = []
    for action in actions or []:
        try:
            action_id = int(action)
        except Exception:
            name = str(action).strip().lower()
            action_id = {"stop": 0, "forward": 1, "left": 2, "right": 3}.get(name, -1)
        if action_id == -1:
            continue
        out.append(action_id)
    if out and out[-1] != 0:
        out.append(0)
    return out


def metric_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def action_text(action: int) -> str:
    return ACTION_ID_TO_TEXT.get(int(action), "forward")


def action_chunk_to_text(actions: Iterable[int]) -> str:
    return ",".join(action_text(a) for a in actions)


def uniform_sample_indices(num_frames: int, num_samples: int) -> List[int]:
    if num_frames <= 0 or num_samples <= 0:
        return []
    n = min(num_frames, num_samples)
    return [int(idx) for idx in np.linspace(0, num_frames - 1, num=n, dtype=int)]


def uniform_span_boundaries(num_actions: int, num_spans: int) -> List[int]:
    if num_actions <= 0:
        return []
    if num_spans <= 0:
        return list(range(num_actions + 1))
    return [(idx * num_actions) // num_spans for idx in range(num_spans + 1)]


def compress_action_span(actions: Iterable[int]) -> str:
    names = [action_text(action) for action in actions if int(action) != -1]
    if not names:
        return "stop"
    spans: List[Tuple[str, int]] = []
    for name in names:
        if spans and spans[-1][0] == name:
            spans[-1] = (name, spans[-1][1] + 1)
        else:
            spans.append((name, 1))
    parts = []
    for name, count in spans:
        if name == "forward":
            parts.append("forward" if count == 1 else f"forward {count} steps")
        elif name in {"left", "right"}:
            parts.append(name if count == 1 else f"{name} {count} times")
        elif name == "stop":
            parts.append("stop")
        else:
            parts.append(name if count == 1 else f"{name} {count} times")
    return ", then ".join(parts)


def annotation_image_paths(traj: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    images = traj.get("image") or []
    if isinstance(images, str):
        images = [images]
    if images:
        return [resolve_path(img, args.image_root) for img in images]

    video = traj.get("video")
    actions = normalize_actions(traj.get("actions") or [])
    if not video or not actions:
        return []
    base = Path(resolve_path(video, args.image_root))
    action_count = len(actions)

    def stored_path(idx: int) -> str:
        candidates = [base / f"{idx:03d}.jpg", base / "rgb" / f"{idx + 1:03d}.jpg"]
        chosen = next((p for p in candidates if p.exists()), candidates[0])
        return str(chosen)

    return [stored_path(idx) for idx in range(action_count)]


def open_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def va2l_images_action_texts(frame_paths: List[str], actions: List[int], max_images: int) -> Tuple[List[str], List[str], List[int]]:
    n = min(len(frame_paths), len(actions))
    if n <= 0:
        return [], [], []
    num_samples = min(n, max_images) if max_images > 0 else n
    boundaries = uniform_span_boundaries(n, num_samples)
    selected_indices = uniform_sample_indices(n, num_samples)
    images = [frame_paths[idx] for idx in selected_indices]
    action_texts: List[str] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        span_start = min(start, n - 1)
        span_end = min(max(end, span_start + 1), n)
        action_texts.append(compress_action_span(actions[span_start:span_end]))
    return images, action_texts, selected_indices


def build_va2l_conversation(traj: Dict[str, Any], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str], List[str], List[int]]:
    actions = normalize_actions(traj.get("actions") or [])
    frame_paths = annotation_image_paths(traj, args)
    selected_images, action_texts, selected_indices = va2l_images_action_texts(frame_paths, actions, args.va2l_num_frames)
    if not selected_images:
        raise ValueError("trajectory must provide actions and image/video paths for VA2L generation")
    parts = ["This is a navigation trajectory consists of several image sequences: "]
    for view_idx, action_desc in enumerate(action_texts, start=1):
        parts.append(f"(Viewpoint {view_idx}: Image:<image>, Action: {action_desc})")
        parts.append(", " if view_idx < len(action_texts) else ". ")
    parts.append(VA2L_SUFFIX)
    content = []
    image_iter = iter(selected_images)
    for seg in re.split(r"(<image>)", "".join(parts)):
        if seg == "<image>":
            content.append({"type": "image", "image": open_rgb(next(image_iter))})
        elif seg:
            content.append({"type": "text", "text": seg})
    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a navigation-instruction generator. Output only the requested instruction.",
                }
            ],
        },
        {"role": "user", "content": content},
    ]
    return conversation, selected_images, action_texts, selected_indices

def strip_repeated_action_tail(text: str) -> str:
    # Remove action-token leakage at the end of VA2L generations, e.g.
    # "... stop forward forward forward" or "... left,left,left". Keep normal
    # navigation prose such as "turn right" by requiring either repeated action
    # tokens or a stop token followed by more action tokens at the very end.
    action_tail = re.compile(
        r"(?:\s*[,.;]?\s*\b(?:left|right|forward|stop)\b){6,}\s*$",
        flags=re.IGNORECASE,
    )
    stop_then_actions_tail = re.compile(
        r"\s+\bstop\b(?:\s*[,.;]?\s*\b(?:left|right|forward|stop)\b){2,}\s*$",
        flags=re.IGNORECASE,
    )
    repeated_same_action_tail = re.compile(
        r"\s+\b(left|right|forward|stop)\b(?:\s*[,.;]?\s*\b\1\b){2,}\s*$",
        flags=re.IGNORECASE,
    )
    previous = None
    while previous != text:
        previous = text
        text = action_tail.sub("", text).rstrip(" ,.;")
        text = stop_then_actions_tail.sub("", text).rstrip(" ,.;")
        text = repeated_same_action_tail.sub("", text).rstrip(" ,.;")
    return text.strip()


def clean_generated_instruction(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"<\|[^>]+\|>", " ", text)
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    if re.search(r"</think>", text, flags=re.IGNORECASE):
        before, after = re.split(r"</think>", text, maxsplit=1, flags=re.IGNORECASE)
        text = before if len(before.split()) >= 3 else after
    text = re.sub(r"</?think>", " ", text, flags=re.IGNORECASE)
    text = " ".join(text.strip().split())

    text = re.sub(r"(?i)^\s*(assistant|gpt|model)\s*[:：-]?\s*", "", text).strip()
    role_chunks = [chunk.strip() for chunk in re.split(r"(?i)\s+\b(?:assistant|gpt|model|user|human)\b\s*[:：-]?\s*", text) if chunk.strip()]
    if len(role_chunks) > 1:
        text = role_chunks[0]

    text = strip_repeated_action_tail(text)

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    deduped: List[str] = []
    seen = set()
    for sentence in sentences or [text]:
        key = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(sentence)
    return " ".join(deduped).strip()


def resolve_pad_token_id(processor) -> int:
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        raise ValueError("processor/tokenizer must define pad_token_id or eos_token_id")
    return int(pad_id)


def strip_qwen35_thinking_prompt(model_inputs, processor, model_type: str):
    if model_type not in ("qwen3_5", "qwen3_5_moe"):
        return model_inputs

    tokenizer = getattr(processor, "tokenizer", processor)
    input_ids = model_inputs["input_ids"]
    old_len = input_ids.shape[-1]
    for suffix in ("<think>\n", "<think>"):
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        suffix_len = len(suffix_ids)
        if suffix_len == 0 or suffix_len > old_len:
            continue
        suffix_tensor = input_ids.new_tensor(suffix_ids)
        if torch.equal(input_ids[0, -suffix_len:], suffix_tensor):
            new_len = old_len - suffix_len
            model_inputs["input_ids"] = input_ids[:, :new_len]
            if "attention_mask" in model_inputs:
                model_inputs["attention_mask"] = model_inputs["attention_mask"][:, :new_len]
            if "position_ids" in model_inputs:
                position_ids = model_inputs["position_ids"]
                if position_ids.shape[-1] == old_len:
                    model_inputs["position_ids"] = position_ids[..., :new_len]
            return model_inputs
    return model_inputs


def build_action_token_language(tokenizer, max_actions: int):
    max_actions = int(max_actions)
    if max_actions <= 0:
        return {}, {}, 0

    movement_actions = ("forward", "left", "right")
    all_final_actions = (*movement_actions, "stop")
    action_sequences = []

    for length in range(1, max_actions):
        for prefix in itertools.product(movement_actions, repeat=length - 1):
            action_sequences.append((*prefix, "stop"))

    for prefix in itertools.product(movement_actions, repeat=max_actions - 1):
        for final_action in all_final_actions:
            action_sequences.append((*prefix, final_action))

    terminal_actions = {}
    trie_children = {}
    max_token_length = 0
    for actions in action_sequences:
        action_text = ",".join(actions)
        token_ids = tuple(
            int(token_id)
            for token_id in tokenizer.encode(action_text, add_special_tokens=False)
        )
        if not token_ids:
            raise ValueError(f"Action string tokenized to an empty sequence: {action_text!r}")
        previous = terminal_actions.setdefault(token_ids, actions)
        if previous != actions:
            raise ValueError(
                f"Tokenizer collision for action strings: {previous!r} and {actions!r}"
            )
        max_token_length = max(max_token_length, len(token_ids))
        for token_index, token_id in enumerate(token_ids):
            prefix = token_ids[:token_index]
            trie_children.setdefault(prefix, set()).add(token_id)

    return trie_children, terminal_actions, max_token_length


class ActionTokenLanguageStoppingCriteria(StoppingCriteria):
    def __init__(self, prompt_length: int, terminal_token_sequences):
        self.prompt_length = int(prompt_length)
        self.terminal_token_sequences = terminal_token_sequences

    def __call__(self, input_ids, scores, **kwargs):
        stopped = [
            tuple(int(token_id) for token_id in row[self.prompt_length :].tolist())
            in self.terminal_token_sequences
            for row in input_ids
        ]
        return torch.tensor(stopped, dtype=torch.bool, device=input_ids.device)


def action_constrained_generate(
    model,
    processor,
    model_inputs,
    max_actions: int,
    stop_at_stop: bool = True,
) -> str:
    del stop_at_stop
    max_actions = int(max_actions)
    if max_actions <= 0:
        return ""
    if model_inputs["input_ids"].shape[0] != 1:
        raise ValueError("Action-constrained decoding currently requires batch size 1")

    tokenizer = getattr(processor, "tokenizer", processor)
    language_cache = getattr(tokenizer, "_RSI_NAV_action_token_languages", None)
    if language_cache is None:
        language_cache = {}
        setattr(tokenizer, "_RSI_NAV_action_token_languages", language_cache)
    if max_actions not in language_cache:
        language_cache[max_actions] = build_action_token_language(
            tokenizer, max_actions=max_actions
        )
    trie_children, terminal_actions, max_token_length = language_cache[max_actions]
    prompt_length = model_inputs["input_ids"].shape[-1]

    def allow_action_tokens(_batch_id, generated_input_ids):
        generated_prefix = tuple(
            int(token_id)
            for token_id in generated_input_ids[prompt_length:].tolist()
        )
        allowed = trie_children.get(generated_prefix)
        if not allowed:
            raise RuntimeError(
                f"Generated token prefix left the action language: {generated_prefix}"
            )
        return sorted(allowed)

    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_token_length,
            do_sample=False,
            use_cache=True,
            pad_token_id=resolve_pad_token_id(processor),
            prefix_allowed_tokens_fn=allow_action_tokens,
            stopping_criteria=StoppingCriteriaList(
                [
                    ActionTokenLanguageStoppingCriteria(
                        prompt_length, terminal_actions
                    )
                ]
            ),
        )
    generated_token_ids = tuple(
        int(token_id) for token_id in output_ids[0, prompt_length:].tolist()
    )

    generated_actions = terminal_actions.get(generated_token_ids)
    if generated_actions is None:
        decoded = tokenizer.decode(list(generated_token_ids), skip_special_tokens=False)
        raise RuntimeError(
            "Constrained decoding did not terminate on a valid action string: "
            f"tokens={generated_token_ids}, decoded={decoded!r}"
        )
    return ",".join(generated_actions)


def should_breakpoint(args: argparse.Namespace, env_name: str) -> bool:
    if os.environ.get(env_name, "0") not in {"1", "true", "True", "yes"}:
        return False
    rank_filter = os.environ.get("RSI_NAV_EVO_PDB_RANK", "")
    return not rank_filter or str(args.rank) == rank_filter


def generate_instruction(model, processor, traj: Dict[str, Any], args: argparse.Namespace, device: torch.device) -> Tuple[str, Dict[str, Any]]:
    conversation, selected_images, action_texts, selected_indices = build_va2l_conversation(traj, args)
    template_kwargs = {
        "conversation": conversation,
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    try:
        model_inputs = processor.apply_chat_template(**template_kwargs, enable_thinking=False)
    except TypeError:
        model_inputs = processor.apply_chat_template(**template_kwargs)
    input_len = model_inputs["input_ids"].shape[1]
    model_inputs = model_inputs.to(device)
    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.instruction_max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(getattr(processor, "tokenizer", None), "eos_token_id", None),
        )
    gen_ids = output_ids[:, input_len:]
    raw_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    text = clean_generated_instruction(raw_text)
    if should_breakpoint(args, "RSI_NAV_EVO_PDB_AFTER_GENERATE"):
        import pdb; pdb.set_trace()
    meta = {
        "raw_generated_instruction": raw_text,
        "va2l_images": selected_images,
        "va2l_action_texts": action_texts,
        "va2l_selected_indices": selected_indices,
        "va2l_num_frames": args.va2l_num_frames,
        "va2l_selected_frames": len(selected_images),
        "instruction_max_new_tokens": args.instruction_max_new_tokens,
    }
    return text, meta


def trajectory_path_id(item: Dict[str, Any], fallback: str) -> str:
    for section in ("trajectory", "episode"):
        row = item.get(section) or {}
        for key in ("path_id", "id", "episode_id", "trajectory_id"):
            value = row.get(key)
            if value is not None:
                return str(value)
    return fallback


def build_nav_conversation(
    instruction: str,
    history_imgs: List[Image.Image],
    slice_rgbs: List[Image.Image],
    slice_actions: List[int],
    slice_conjunctions: List[str],
    num_future_steps: int,
    num_history: int,
) -> List[Dict[str, Any]]:
    first_user_parts: List[Dict[str, Any]] = [
        {"type": "text", "text": NAV_PROMPT.replace("<instruction>", normalize_instruction(instruction))}
    ]
    if history_imgs:
        if len(history_imgs) <= num_history:
            hist_idx = list(range(len(history_imgs)))
        else:
            stride = max(len(history_imgs) // num_history, 1)
            hist_idx = list(range(0, len(history_imgs), stride))[:num_history]
        first_user_parts.append({"type": "text", "text": " These are your historical observations:"})
        for idx in hist_idx:
            first_user_parts.append({"type": "image", "image": history_imgs[int(idx)]})
        first_user_parts.append({"type": "text", "text": " This is your current view:"})
    else:
        first_user_parts.append({"type": "text", "text": " This is your current view:"})
    first_user_parts.append({"type": "image", "image": slice_rgbs[0]})

    conversation: List[Dict[str, Any]] = [{"role": "user", "content": first_user_parts}]
    n_done_turns = len(slice_actions) // num_future_steps
    for turn_idx in range(n_done_turns):
        acts = slice_actions[turn_idx * num_future_steps : (turn_idx + 1) * num_future_steps]
        conversation.append({"role": "assistant", "content": [{"type": "text", "text": action_chunk_to_text(acts)}]})
        next_frame_pos = (turn_idx + 1) * num_future_steps
        if next_frame_pos < len(slice_rgbs):
            conj = slice_conjunctions[turn_idx]
            conversation.append({"role": "user", "content": [{"type": "text", "text": conj.strip()}, {"type": "image", "image": slice_rgbs[next_frame_pos]}]})
    return conversation


def parse_actions(output: str) -> List[int]:
    regex = re.compile("|".join(re.escape(a) for a in ACTION_TEXT_TO_ID))
    matches = regex.findall(output.lower())
    return list(itertools.chain.from_iterable(ACTION_TEXT_TO_ID[m] for m in matches))


def resolve_scene_id(raw_scene_id: str, scene_root: str = "") -> str:
    scene_id = raw_scene_id
    if scene_root and not os.path.isabs(scene_id):
        scene_id = str(Path(scene_root) / scene_id)

    # Prefer non-basis HM3D render assets when they exist; using .basis.glb can
    # make habitat-sim auto-probe a same-prefix .basis.scn semantic descriptor.
    if scene_id.endswith(".basis.glb"):
        non_basis = scene_id.replace(".basis.glb", ".glb")
        if Path(non_basis).exists():
            scene_id = non_basis
    return scene_id


def build_dataset(episode_rows: List[Dict[str, Any]], scene_root: str = "") -> VLNDatasetV1:
    episodes = []
    for item in episode_rows:
        raw = item["episode"]
        scene_id = resolve_scene_id(raw["scene_id"], scene_root=scene_root)
        episodes.append(
            VLNEpisode(
                episode_id=str(raw.get("episode_id") or item["id"]),
                trajectory_id=str(raw.get("trajectory_id") or item["id"]),
                scene_id=scene_id,
                start_position=raw["start_position"],
                start_rotation=raw["start_rotation"],
                goals=[NavigationGoal(**goal) for goal in raw["goals"]],
                instruction=InstructionData(instruction_text="", instruction_tokens=[]),
                reference_path=raw.get("reference_path") or [],
                info=raw.get("info", {}),
            )
        )
    dataset = VLNDatasetV1(config=None)
    dataset.episodes = episodes
    dataset.instruction_vocab = VocabDict(word_list=[])
    return dataset


def make_env(config_path: str, gpu: int, dataset: VLNDatasetV1) -> Env:
    config = get_habitat_config(config_path)
    with habitat.config.read_write(config):
        if torch.cuda.is_available():
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu
        config.habitat.simulator.should_setup_semantic_ids = False
        sim_sensors = config.habitat.simulator.agents.main_agent.sim_sensors
        if "semantic_sensor" in sim_sensors:
            del sim_sensors["semantic_sensor"]
        # Keep evo verification lightweight. Top-down-map, fog-of-war, and
        # oracle measurements are useful for visualization/debug but can
        # dominate per-step Habitat cost and are not needed for acceptance.
    return Env(config=config, dataset=dataset)


def verify_instruction(
    model,
    processor,
    env: Env,
    instruction: str,
    args: argparse.Namespace,
    device: torch.device,
    path_id: str = "",
) -> Dict[str, Any]:
    verify_started = time.monotonic()
    next_heartbeat = verify_started + max(0.0, float(args.verify_heartbeat_seconds or 0.0))
    observations = env.reset()
    step_id = 0
    rgb_list: List[Image.Image] = []
    executed_actions: List[int] = []
    slice_start = 0
    slice_actions: List[int] = []
    slice_conjunctions: List[str] = []
    pending_action_seq: List[int] = []
    agent_called_stop = False
    start_metrics = env.get_metrics()
    min_ne = float(start_metrics.get("distance_to_goal", float("inf")))
    while not env.episode_over:
        now = time.monotonic()
        if args.verify_heartbeat_seconds and args.verify_heartbeat_seconds > 0 and now >= next_heartbeat:
            log_rank_verify(args, path_id, step_id, now - verify_started)
            next_heartbeat = now + float(args.verify_heartbeat_seconds)
        if args.max_steps is not None and step_id >= args.max_steps:
            break
        image = Image.fromarray(observations["rgb"]).convert("RGB")
        rgb_list.append(image)

        if not pending_action_seq:
            in_slice_pos = len(rgb_list) - 1 - slice_start
            if in_slice_pos > 0 and in_slice_pos % args.num_future_steps == 0:
                conjunction_idx = len(slice_conjunctions) % len(CONJUNCTIONS)
                slice_conjunctions.append(CONJUNCTIONS[conjunction_idx])
            conversation = build_nav_conversation(
                instruction=instruction,
                history_imgs=rgb_list[:slice_start],
                slice_rgbs=rgb_list[slice_start:],
                slice_actions=slice_actions,
                slice_conjunctions=slice_conjunctions,
                num_future_steps=args.num_future_steps,
                num_history=args.num_history,
            )
            model_inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            model_type = getattr(getattr(model, "config", None), "model_type", "")
            model_inputs = strip_qwen35_thinking_prompt(model_inputs, processor, model_type)
            input_len = model_inputs["input_ids"].shape[1]
            model_inputs = model_inputs.to(device)
            if args.free_generation:
                with torch.inference_mode():
                    output_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=args.nav_max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=resolve_pad_token_id(processor),
                    )
                gen_text = processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0].strip()
            else:
                gen_text = action_constrained_generate(
                    model, processor, model_inputs, max_actions=args.num_future_steps
                )
            action_seq = parse_actions(gen_text)
            if not action_seq:
                action_seq = [0]
            action_seq = action_seq[: args.num_future_steps]
            if 0 in action_seq:
                action_seq = action_seq[: action_seq.index(0) + 1]
            elif len(action_seq) < args.num_future_steps:
                action_seq.append(0)
            slice_actions.extend(action_seq)
            pending_action_seq = action_seq[:]

        action = pending_action_seq.pop(0)
        executed_actions.append(int(action))
        if int(action) == 0:
            agent_called_stop = True
        observations = env.step(action)
        step_id += 1
        step_metrics = env.get_metrics()
        try:
            min_ne = min(min_ne, float(step_metrics.get("distance_to_goal", float("inf"))))
        except (TypeError, ValueError):
            pass
        if step_id % args.num_frames == 0:
            slice_start = len(rgb_list)
            slice_actions = []
            slice_conjunctions = []
            pending_action_seq = []

    metrics = env.get_metrics()
    ne = float(metrics.get("distance_to_goal", float("inf")))
    success_distance = float(getattr(args, "metric_success_distance", 3.0) or 3.0)
    distance_success = 1.0 if ne <= success_distance else 0.0
    success = 1.0 if agent_called_stop and distance_success > 0.0 else 0.0
    osr = 1.0 if min_ne <= success_distance else 0.0
    return {
        "success": success,
        "sr": success,
        "osr": osr,
        "spl": float(metrics.get("spl", 0.0)),
        "steps": step_id,
        "executed_actions": executed_actions,
    }


def normalize_instruction(text: str) -> str:
    text = clean_generated_instruction(text)
    text = " ".join((text or "").strip().split())
    if text and text[-1] not in ".!?":
        text += "."
    return text


def is_valid_instruction(text: str) -> bool:
    words = re.findall(r"[A-Za-z]+", text or "")
    if len(words) < 5:
        return False
    nav_verbs = {"go", "walk", "turn", "head", "continue", "move", "proceed", "stop", "enter", "pass"}
    return any(word.lower() in nav_verbs for word in words)


def history_frame_indices(start_idx: int, num_history: int) -> List[int]:
    if start_idx <= 0 or num_history <= 0:
        return []
    candidates = list(range(0, start_idx, 4))
    if len(candidates) <= num_history:
        return candidates
    positions = np.linspace(0, len(candidates) - 1, num=num_history, dtype=int)
    return [candidates[int(pos)] for pos in positions]


def build_nav_samples(record_id: str, instruction: str, frame_paths: List[str], actions: List[int], args: argparse.Namespace) -> List[Dict[str, Any]]:
    samples = []
    for start_idx in range(0, len(actions), args.num_frames):
        episode_actions = actions[start_idx : start_idx + args.num_frames]
        if not episode_actions or start_idx >= len(frame_paths):
            continue
        images: List[str] = []
        convs: List[Dict[str, str]] = []
        first_parts = [NAV_PROMPT.replace("<instruction>", normalize_instruction(instruction))]
        hist = history_frame_indices(start_idx, args.num_history)
        if hist:
            first_parts.append(" These are your historical observations: ")
            for idx in hist:
                first_parts.append("<image>")
                images.append(frame_paths[idx])
            first_parts.append(" This is your current view: ")
        else:
            first_parts.append(" This is your current view: ")
        first_parts.append("<image>")
        images.append(frame_paths[start_idx])
        convs.append({"from": "human", "value": "".join(first_parts)})

        for turn_start in range(0, len(episode_actions), args.num_future_steps):
            chunk = episode_actions[turn_start : turn_start + args.num_future_steps]
            convs.append({"from": "gpt", "value": action_chunk_to_text(chunk)})
            next_pos = turn_start + args.num_future_steps
            frame_idx = start_idx + next_pos
            if next_pos < len(episode_actions) and frame_idx < len(frame_paths):
                conj = CONJUNCTIONS[(next_pos // args.num_future_steps - 1) % len(CONJUNCTIONS)]
                convs.append({"from": "human", "value": f"{conj}<image>"})
                images.append(frame_paths[frame_idx])

        samples.append(
            {
                "id": f"{record_id}_vl2a_{len(samples)}",
                "data_source": "RSI-Nav_EVO_VL2A",
                "image": images,
                "conversations": convs,
                "meta": {
                    "source_id": record_id,
                    "round_id": args.round_id,
                    "batch_id": args.batch_id,
                    "rank": args.rank,
                    "start_action_index": start_idx,
                },
            }
        )
    return samples


def build_va2l_sample(record_id: str, instruction: str, frame_paths: List[str], actions: List[int], args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    images, action_texts, selected_indices = va2l_images_action_texts(frame_paths, actions, args.va2l_num_frames)
    if not images:
        return None
    parts = ["This is a navigation trajectory consists of several image sequences: "]
    for view_idx, action_desc in enumerate(action_texts, start=1):
        parts.append(f"(Viewpoint {view_idx}: Image:<image>, Action: {action_desc})")
        parts.append(", " if view_idx < len(action_texts) else ". ")
    parts.append(VA2L_SUFFIX)
    return {
        "id": f"{record_id}_va2l",
        "data_source": "RSI-Nav_EVO_VA2L",
        "image": images,
        "conversations": [
            {"from": "human", "value": "".join(parts)},
            {"from": "gpt", "value": normalize_instruction(instruction)},
        ],
        "meta": {
            "source_id": record_id,
            "round_id": args.round_id,
            "batch_id": args.batch_id,
            "rank": args.rank,
            "selected_action_start_indices": selected_indices,
            "actions": actions,
            "action_texts": action_texts,
        },
    }

def write_empty_rank_outputs(out_dir: Path, args: argparse.Namespace, candidates: int = 0) -> None:
    summary = {
        "round_id": args.round_id,
        "batch_id": args.batch_id,
        "rank": args.rank,
        "world_size": args.world_size,
        "checkpoint": args.model_path,
        "candidates": candidates,
        "generated": 0,
        "accepted": 0,
        "failed": 0,
        "created_at": utc_timestamp(),
    }
    write_rank_success_shard(out_dir, args.batch_id, args.rank, [], summary)


def run() -> None:
    run_started = time.monotonic()
    args = fill_dist_args(parse_args())
    random.seed(args.seed + int(args.rank))
    np.random.seed(args.seed + int(args.rank))
    init_torchrun_process_group(args)
    dist_barrier()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else args.device)
    out_dir = round_dir(args.runs_dir, args.round_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_path = batch_dir(args.runs_dir, args.round_id, args.batch_id)
    batch_path.mkdir(parents=True, exist_ok=True)

    setup_started = time.monotonic()
    world_size = int(args.world_size)
    rank = int(args.rank)
    traj_map = load_trajectory_map(args.trajectory_jsonl)
    retry_ids = load_retry_ids(args.retry_jsonl)
    all_items = load_episode_rows(args.episodes_jsonl, traj_map, retry_ids)
    start = args.candidate_offset
    if args.max_candidates_per_rank is None:
        batch_items = all_items[start:]
    else:
        batch_items = all_items[start : start + world_size * int(args.max_candidates_per_rank)]
    ranked_indexed = list(enumerate(batch_items))[rank::world_size]
    ranked = [item for _, item in ranked_indexed]
    data_setup_seconds = time.monotonic() - setup_started
    if not ranked:
        append_timing(
            out_dir,
            {
                "source": "rollout_worker",
                "stage": "rollout_worker_empty",
                "round_id": args.round_id,
                "batch_id": args.batch_id,
                "rank": args.rank,
                "world_size": args.world_size,
                "batch_candidates": len(batch_items),
                "data_setup_seconds": data_setup_seconds,
                "seconds": time.monotonic() - run_started,
            },
        )
        write_empty_rank_outputs(out_dir, args, candidates=0)
        dist_barrier()
        dist_barrier()
        return

    env_started = time.monotonic()
    # Match eval/Nav/r2r_multiworker.py more closely: every torchrun rank sees the
    # same batch dataset, then processes its rank::world_size episode subset.
    # The Habitat config data_path is not used for matching here; episode rows
    # are loaded from --episodes-jsonl and paired with trajectory rows by id /
    # path_id / episode_id / trajectory_id before being injected into Env.
    dataset = build_dataset(batch_items, scene_root=args.scene_root)
    with suppress_stderr():
        env = make_env(args.habitat_config_path, args.gpu, dataset)
    env_setup_seconds = time.monotonic() - env_started

    # Same model placement as eval/Nav/r2r_multiworker.py: every torchrun rank loads
    # one full model replica on its local GPU. The same model instance is reused
    # for both VA2L instruction annotation and L2A navigation verification.
    model_load_started = time.monotonic()
    with suppress_stdout(), suppress_stderr():
        model = load_model_for_eval(args.model_path, rank=int(args.rank), gpu=int(args.gpu))
    print(
        f"[evo_rollout] attn_implementation={os.environ.get('ATTN_IMPLEMENTATION')} "
        f"model_type={getattr(getattr(model, 'config', None), 'model_type', '')} "
        f"rank={args.rank}",
        flush=True,
    )
    model_load_seconds = time.monotonic() - model_load_started
    processor_load_started = time.monotonic()
    with suppress_stdout(), suppress_stderr():
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor_load_seconds = time.monotonic() - processor_load_started
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.model_max_length = args.model_max_length
        processor.tokenizer.padding_side = "right"
    set_processor_pixels(processor, args.min_pixels, args.max_pixels)
    model.eval()
    append_timing(
        out_dir,
        {
            "source": "rollout_worker",
            "stage": "rollout_worker_load",
            "round_id": args.round_id,
            "batch_id": args.batch_id,
            "rank": args.rank,
            "world_size": args.world_size,
            "gpu": args.gpu,
            "worker_mode": args.worker_mode,
            "data_setup_seconds": data_setup_seconds,
            "env_setup_seconds": env_setup_seconds,
            "model_load_seconds": model_load_seconds,
            "processor_load_seconds": processor_load_seconds,
            "seconds": data_setup_seconds + env_setup_seconds + model_load_seconds + processor_load_seconds,
            "model_path": args.model_path,
            "checkpoint_generation": args.checkpoint_generation,
        },
    )
    dist_barrier()

    success_rows: List[Dict[str, Any]] = []
    live_accepted_evo_qa_count = 0
    live_accepted_trajectories_count = 0
    live_accepted_vl2a_count = 0
    live_accepted_va2l_count = 0
    failed_count = 0
    generated_count = 0
    instruction_generation_total_seconds = 0.0
    nav_verification_total_seconds = 0.0
    candidate_total_seconds = 0.0

    progress = progress_iter(ranked_indexed, args)
    for local_idx, (episode_idx, item) in enumerate(progress):
        record_id = str(item["id"])
        path_id = trajectory_path_id(item, record_id)
        candidate_pos = local_idx + 1
        try:
            candidate_started = time.monotonic()
            log_rank_path(args, path_id, candidate_pos, len(ranked))
            instruction_started = time.monotonic()
            with suppress_stderr():
                instruction, gen_meta = generate_instruction(model, processor, item["trajectory"], args, device)
            instruction_seconds = time.monotonic() - instruction_started
            instruction_generation_total_seconds += instruction_seconds
            gen_meta["instruction_generation_seconds"] = instruction_seconds
            log_rank_instruction(args, path_id, instruction)
            candidate_record = {
                "round_id": args.round_id,
                "batch_id": args.batch_id,
                "rank": args.rank,
                "trajectory_id": record_id,
                "path_id": path_id,
                "checkpoint": args.model_path,
                "checkpoint_version": args.model_path,
                "checkpoint_generation": args.checkpoint_generation,
                "generated_instruction": instruction,
                "generation_meta": gen_meta,
                "created_at": utc_timestamp(),
            }
            generated_count += 1
            if not is_valid_instruction(instruction):
                reason = "invalid_generated_instruction"
                log_rank_invalid_instruction(args, path_id, instruction, "too_short_or_no_nav_verb")
                if should_breakpoint(args, "RSI_NAV_EVO_PDB_INVALID_INSTRUCTION"):
                    import pdb; pdb.set_trace()
                candidate_seconds = time.monotonic() - candidate_started
                candidate_total_seconds += candidate_seconds
                base_record = {
                    **candidate_record,
                    "accepted": False,
                    "candidate_seconds": candidate_seconds,
                }
                append_timing(
                    out_dir,
                    {
                        "source": "rollout_worker",
                        "stage": "rollout_candidate_invalid_instruction",
                        "round_id": args.round_id,
                        "batch_id": args.batch_id,
                        "rank": args.rank,
                        "candidate_index": candidate_pos,
                        "trajectory_id": record_id,
                        "path_id": path_id,
                        "accepted": False,
                        "instruction_generation_seconds": instruction_seconds,
                        "nav_verification_seconds": 0.0,
                        "seconds": candidate_seconds,
                        "failure_reason": reason,
                    },
                )
                append_jsonl(failed_path(out_dir), retry_failure_record(base_record, item, reason))
                append_annotated_trajectory(args, record_id, path_id, accepted=False, failure_reason=reason)
                failed_count += 1
                continue

            env.current_episode = env.episodes[episode_idx]
            verify_started = time.monotonic()
            with candidate_timeout(float(args.verify_timeout_seconds or 0.0), path_id):
                with suppress_stderr():
                    result = verify_instruction(model, processor, env, instruction, args, device, path_id=path_id)
            verify_seconds = time.monotonic() - verify_started
            nav_verification_total_seconds += verify_seconds
            result["nav_verification_seconds"] = verify_seconds
            sr = metric_float(result.get("sr", result.get("success")), 0.0)
            spl = metric_float(result.get("spl"), 0.0)
            osr = metric_float(result.get("osr"), 0.0)
            vl2a_accepted = sr >= float(args.vl2a_min_sr) and spl >= float(args.vl2a_min_spl)
            va2l_accepted = sr >= float(args.va2l_min_sr) and spl >= float(args.va2l_min_spl)
            accepted = bool(vl2a_accepted or va2l_accepted)
            candidate_seconds = time.monotonic() - candidate_started
            candidate_total_seconds += candidate_seconds
            base_record = {
                **candidate_record,
                **result,
                "accepted": accepted,
                "vl2a_accepted": vl2a_accepted,
                "va2l_accepted": va2l_accepted,
                "acceptance_policy": {
                    "vl2a": {
                        "rule": "sr >= vl2a_min_sr and spl >= vl2a_min_spl",
                        "vl2a_min_sr": float(args.vl2a_min_sr),
                        "vl2a_min_spl": float(args.vl2a_min_spl),
                    },
                    "va2l": {
                        "rule": "sr >= va2l_min_sr and spl >= va2l_min_spl",
                        "va2l_min_sr": float(args.va2l_min_sr),
                        "va2l_min_spl": float(args.va2l_min_spl),
                    },
                },
                "candidate_seconds": candidate_seconds,
            }
            append_timing(
                out_dir,
                {
                    "source": "rollout_worker",
                    "stage": "rollout_candidate",
                    "round_id": args.round_id,
                    "batch_id": args.batch_id,
                    "rank": args.rank,
                    "candidate_index": candidate_pos,
                    "trajectory_id": record_id,
                    "path_id": path_id,
                    "accepted": accepted,
                    "vl2a_accepted": vl2a_accepted,
                    "va2l_accepted": va2l_accepted,
                    "instruction_generation_seconds": instruction_seconds,
                    "nav_verification_seconds": verify_seconds,
                    "seconds": candidate_seconds,
                    "steps": result.get("steps"),
                    "sr": sr,
                    "osr": osr,
                    "spl": spl,
                },
            )
            append_annotated_trajectory(
                args,
                record_id,
                path_id,
                accepted=accepted,
                failure_reason="" if accepted else "not_selected_by_vl2a_or_va2l_policy",
                metrics=result,
                vl2a_accepted=vl2a_accepted,
                va2l_accepted=va2l_accepted,
            )
            if accepted:
                source_frame_paths = annotation_image_paths(item["trajectory"], args)
                source_actions = normalize_actions(item["trajectory"].get("actions") or [])
                nav_samples = (
                    build_nav_samples(record_id, instruction, source_frame_paths, source_actions, args)
                    if vl2a_accepted
                    else []
                )
                va2l_sample = (
                    build_va2l_sample(record_id, instruction, source_frame_paths, source_actions, args)
                    if va2l_accepted
                    else None
                )
                has_requested_vl2a = (not vl2a_accepted) or bool(nav_samples)
                has_requested_va2l = (not va2l_accepted) or isinstance(va2l_sample, dict)
                if has_requested_vl2a and has_requested_va2l and (nav_samples or isinstance(va2l_sample, dict)):
                    success_record = {
                        **base_record,
                        "source_episode": item.get("episode"),
                        "source_trajectory": item.get("trajectory"),
                        "vl2a_samples": nav_samples,
                        "va2l_sample": va2l_sample,
                    }
                    success_rows.append(success_record)
                    if not args.disable_live_success_write:
                        live_counts = append_live_success_outputs(out_dir, args.batch_id, success_record)
                        live_accepted_evo_qa_count += int(live_counts.get("live_accepted_evo_qa", 0) or 0)
                        live_accepted_trajectories_count += int(live_counts.get("live_accepted_trajectories", 0) or 0)
                        live_accepted_vl2a_count += int(live_counts.get("live_accepted_vl2a", 0) or 0)
                        live_accepted_va2l_count += int(live_counts.get("live_accepted_va2l", 0) or 0)
                else:
                    append_jsonl(failed_path(out_dir), retry_failure_record(base_record, item, "missing_training_sample"))
                    failed_count += 1
            else:
                append_jsonl(failed_path(out_dir), retry_failure_record(base_record, item, "not_selected_by_vl2a_or_va2l_policy"))
                failed_count += 1
        except Exception as exc:
            try:
                path_id = trajectory_path_id(item, record_id)
            except Exception:
                path_id = record_id
            candidate_seconds = time.monotonic() - locals().get("candidate_started", time.monotonic())
            candidate_total_seconds += candidate_seconds
            append_timing(
                out_dir,
                {
                    "source": "rollout_worker",
                    "stage": "rollout_candidate_error",
                    "round_id": args.round_id,
                    "batch_id": args.batch_id,
                    "rank": args.rank,
                    "candidate_index": candidate_pos,
                    "trajectory_id": record_id,
                    "path_id": path_id,
                    "seconds": candidate_seconds,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )
            log_rank_error(args, path_id, exc)
            failure_reason = f"{exc.__class__.__name__}: {exc}"
            append_jsonl(
                failed_path(out_dir),
                {
                    "round_id": args.round_id,
                    "batch_id": args.batch_id,
                    "rank": args.rank,
                    "trajectory_id": record_id,
                    "path_id": path_id,
                    "checkpoint": args.model_path,
                    "checkpoint_version": args.model_path,
                    "checkpoint_generation": args.checkpoint_generation,
                    "accepted": False,
                    "failure_reason": failure_reason,
                    "source_episode": item.get("episode"),
                    "source_trajectory": item.get("trajectory"),
                    "created_at": utc_timestamp(),
                },
            )
            append_annotated_trajectory(args, record_id, path_id, accepted=False, failure_reason=failure_reason)
            failed_count += 1

    dist_barrier()
    run_seconds = time.monotonic() - run_started
    accepted_vl2a_trajectories = sum(1 for row in success_rows if row.get("vl2a_accepted"))
    accepted_va2l_trajectories = sum(1 for row in success_rows if row.get("va2l_accepted"))
    accepted_vl2a_samples = sum(len(row.get("vl2a_samples") or []) for row in success_rows)
    accepted_va2l_samples = sum(1 for row in success_rows if isinstance(row.get("va2l_sample"), dict))
    summary = {
        "round_id": args.round_id,
        "batch_id": args.batch_id,
        "rank": args.rank,
        "world_size": args.world_size,
        "checkpoint": args.model_path,
        "checkpoint_generation": args.checkpoint_generation,
        "candidates": len(ranked),
        "generated": generated_count,
        "accepted": len(success_rows),
        "accepted_vl2a_trajectories": accepted_vl2a_trajectories,
        "accepted_va2l_trajectories": accepted_va2l_trajectories,
        "accepted_vl2a_samples": accepted_vl2a_samples,
        "accepted_va2l_samples": accepted_va2l_samples,
        "live_accepted_evo_qa": live_accepted_evo_qa_count,
        "live_accepted_trajectories": live_accepted_trajectories_count,
        "live_accepted_vl2a": live_accepted_vl2a_count,
        "live_accepted_va2l": live_accepted_va2l_count,
        "failed": failed_count,
        "timing": {
            "run_seconds": run_seconds,
            "data_setup_seconds": data_setup_seconds,
            "env_setup_seconds": env_setup_seconds,
            "model_load_seconds": model_load_seconds,
            "processor_load_seconds": processor_load_seconds,
            "instruction_generation_seconds": instruction_generation_total_seconds,
            "nav_verification_seconds": nav_verification_total_seconds,
            "candidate_seconds": candidate_total_seconds,
        },
        "created_at": utc_timestamp(),
    }
    append_timing(
        out_dir,
        {
            "source": "rollout_worker",
            "stage": "rollout_worker_total",
            "round_id": args.round_id,
            "batch_id": args.batch_id,
            "rank": args.rank,
            "world_size": args.world_size,
            "candidates": len(ranked),
            "batch_candidates": len(batch_items),
            "generated": generated_count,
            "accepted": len(success_rows),
            "accepted_vl2a_trajectories": accepted_vl2a_trajectories,
            "accepted_va2l_trajectories": accepted_va2l_trajectories,
            "accepted_vl2a_samples": accepted_vl2a_samples,
            "accepted_va2l_samples": accepted_va2l_samples,
            "live_accepted_evo_qa": live_accepted_evo_qa_count,
            "live_accepted_trajectories": live_accepted_trajectories_count,
            "live_accepted_vl2a": live_accepted_vl2a_count,
            "live_accepted_va2l": live_accepted_va2l_count,
            "failed": failed_count,
            "seconds": run_seconds,
            "data_setup_seconds": data_setup_seconds,
            "env_setup_seconds": env_setup_seconds,
            "model_load_seconds": model_load_seconds,
            "processor_load_seconds": processor_load_seconds,
            "instruction_generation_seconds": instruction_generation_total_seconds,
            "nav_verification_seconds": nav_verification_total_seconds,
            "candidate_seconds": candidate_total_seconds,
        },
    )
    write_rank_success_shard(out_dir, args.batch_id, args.rank, success_rows, summary)
    env.close()


if __name__ == "__main__":
    run()
