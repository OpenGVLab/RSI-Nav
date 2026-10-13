#!/usr/bin/env python3
"""Export an evo checkpoint directory into a HuggingFace inference checkpoint.

The evo training output already contains a merged ``model.safetensors`` at the
checkpoint root. This script copies inference sidecar files and optionally
reshards that safetensors file into the standard HF
``model.safetensors-00001-of-00004.safetensors`` + index layout.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Sequence, Tuple

from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_SIDECAR_FILES = (
    "LICENSE",
    "README.md",
    "config.json",
    "configuration.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


def parse_size(value: str) -> int:
    text = value.strip().upper()
    units = {
        "B": 1,
        "K": 1024,
        "KB": 1024,
        "M": 1024**2,
        "MB": 1024**2,
        "G": 1024**3,
        "GB": 1024**3,
        "T": 1024**4,
        "TB": 1024**4,
    }
    number = text
    unit = "B"
    for suffix in sorted(units, key=len, reverse=True):
        if text.endswith(suffix):
            number = text[: -len(suffix)].strip()
            unit = suffix
            break
    return int(float(number) * units[unit])


def read_safetensors_header(path: Path) -> Tuple[OrderedDict[str, int], Dict[str, str]]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len), object_pairs_hook=OrderedDict)

    metadata = dict(header.pop("__metadata__", {}) or {})
    tensor_sizes: OrderedDict[str, int] = OrderedDict()
    for name, info in header.items():
        start, end = info["data_offsets"]
        tensor_sizes[name] = int(end) - int(start)
    return tensor_sizes, metadata


def plan_shards(tensor_sizes: MutableMapping[str, int], max_shard_size: int) -> List[List[str]]:
    if max_shard_size <= 0:
        return [list(tensor_sizes)]

    shards: List[List[str]] = []
    current: List[str] = []
    current_size = 0
    for name, size in tensor_sizes.items():
        if current and current_size + size > max_shard_size:
            shards.append(current)
            current = []
            current_size = 0
        current.append(name)
        current_size += size
    if current:
        shards.append(current)
    return shards


def copy_sidecars(source_dirs: Sequence[Path], output_dir: Path, overwrite: bool) -> List[str]:
    copied: List[str] = []
    for filename in DEFAULT_SIDECAR_FILES:
        dst = output_dir / filename
        if dst.exists() and not overwrite:
            continue

        for source_dir in source_dirs:
            src = source_dir / filename
            if src.exists() and src.is_file():
                shutil.copy2(src, dst)
                copied.append(filename)
                break

    # Older/newer Qwen-VL checkpoints may use either name. Keep the original
    # processor_config.json, and create preprocessor_config.json only when no
    # real one was available from the checkpoint/base model.
    processor = output_dir / "processor_config.json"
    preprocessor = output_dir / "preprocessor_config.json"
    if processor.exists() and not preprocessor.exists():
        shutil.copy2(processor, preprocessor)
        copied.append("preprocessor_config.json")
    return copied


def remove_existing_weights(output_dir: Path) -> None:
    for path in output_dir.glob("model.safetensors*"):
        if path.is_file() or path.is_symlink():
            path.unlink()


def write_sharded_safetensors(
    source_model: Path,
    output_dir: Path,
    max_shard_size: int,
) -> Tuple[int, int]:
    tensor_sizes, metadata = read_safetensors_header(source_model)
    if not tensor_sizes:
        raise ValueError(f"no tensors found in {source_model}")
    metadata.setdefault("format", "pt")

    shards = plan_shards(tensor_sizes, max_shard_size)
    total_size = sum(tensor_sizes.values())

    if len(shards) == 1:
        dst = output_dir / "model.safetensors"
        shutil.copy2(source_model, dst)
        return 1, total_size

    digits = max(5, len(str(len(shards))))
    weight_map: Dict[str, str] = {}
    with safe_open(source_model, framework="pt", device="cpu") as handle:
        for shard_idx, tensor_names in enumerate(shards, start=1):
            shard_name = f"model.safetensors-{shard_idx:0{digits}d}-of-{len(shards):0{digits}d}.safetensors"
            tensors = {name: handle.get_tensor(name) for name in tensor_names}
            save_file(tensors, output_dir / shard_name, metadata=metadata)
            for name in tensor_names:
                weight_map[name] = shard_name
            del tensors

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with (output_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return len(shards), total_size


def find_source_model(checkpoint_dir: Path) -> Path:
    candidates = [
        checkpoint_dir / "model.safetensors",
        checkpoint_dir / "checkpoint-4" / "model.safetensors",
    ]
    candidates.extend(sorted(checkpoint_dir.glob("checkpoint-*/model.safetensors")))
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no model.safetensors found under {checkpoint_dir}")


def sync_tokenizer_class_from_base(output_dir: Path, base_model_dir: Path | None) -> bool:
    if base_model_dir is None:
        return False

    output_config = output_dir / "tokenizer_config.json"
    base_config = base_model_dir / "tokenizer_config.json"
    if not output_config.exists() or not base_config.exists():
        return False

    with output_config.open("r", encoding="utf-8") as handle:
        output_data = json.load(handle)
    with base_config.open("r", encoding="utf-8") as handle:
        base_data = json.load(handle)

    tokenizer_class = base_data.get("tokenizer_class")
    if not tokenizer_class:
        return False
    if output_data.get("tokenizer_class") == tokenizer_class:
        return False

    output_data["tokenizer_class"] = tokenizer_class
    with output_config.open("w", encoding="utf-8") as handle:
        json.dump(output_data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return True


def build_source_dirs(checkpoint_dir: Path, base_model_dir: Path | None) -> List[Path]:
    dirs = [checkpoint_dir]
    checkpoint_subdirs = sorted(
        [path for path in checkpoint_dir.glob("checkpoint-*") if path.is_dir()],
        reverse=True,
    )
    dirs.extend(checkpoint_subdirs)
    if base_model_dir is not None:
        dirs.append(base_model_dir)
    return dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, help="evo current_ckpt or saved_ckpts/... directory")
    parser.add_argument("--output-dir", required=True, help="directory to write the inference checkpoint")
    parser.add_argument(
        "--base-model-dir",
        default="",
        help="optional original HF model dir used only to fill missing sidecar files",
    )
    parser.add_argument("--max-shard-size", default="5GB", help="target shard size, e.g. 5GB; use 0 to keep one file")
    parser.add_argument("--overwrite", action="store_true", help="allow writing into an existing output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    base_model_dir = Path(args.base_model_dir).expanduser().resolve() if args.base_model_dir else None

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint dir does not exist: {checkpoint_dir}")
    if base_model_dir is not None and not base_model_dir.exists():
        raise FileNotFoundError(f"base model dir does not exist: {base_model_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output dir is not empty, pass --overwrite to reuse it: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    remove_existing_weights(output_dir)

    source_model = find_source_model(checkpoint_dir)
    source_dirs = build_source_dirs(checkpoint_dir, base_model_dir)
    copied = copy_sidecars(source_dirs, output_dir, overwrite=True)
    if sync_tokenizer_class_from_base(output_dir, base_model_dir):
        copied.append("tokenizer_config.json:tokenizer_class")
    shard_count, total_size = write_sharded_safetensors(source_model, output_dir, parse_size(args.max_shard_size))

    print(f"[export_infer_checkpoint] source_model={source_model}")
    print(f"[export_infer_checkpoint] output_dir={output_dir}")
    print(f"[export_infer_checkpoint] shards={shard_count} total_size={total_size}")
    print(f"[export_infer_checkpoint] sidecars={','.join(sorted(set(copied))) if copied else 'none'}")


if __name__ == "__main__":
    main()

#   python xxx/RSI-Nav/evo/export_infer_checkpoint.py \
#     --checkpoint-dir xx/evo-vln/evo_traj/evo_ckpts/saved_ckpts/round_000000_batch_000001_update_000001 \
#     --output-dir xx/evo-vln/weights/evo_round_000000_batch_000001_update_000001_hf \
#     --base-model-dir xx/evo-vln/weights/Qwen3.5-9B \
#     --max-shard-size 5GB
