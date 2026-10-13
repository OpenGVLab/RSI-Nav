#!/usr/bin/env python3
"""Train on evo JSONL without modifying the global dataset registry.

Usage pattern:
  torchrun ... qwen3vl/evo/evo_train.py \
    --evo_nav_jsonl /path/accepted_vl2a.jsonl \
    --evo_va2l_jsonl /path/accepted_va2l.jsonl \
    --evo_spatial_jsonl /path/spatial_qa.jsonl \
    --evo_stop_jsonl /path/stop_qa.jsonl \
    --dataset_use evo_nav,evo_va2l,evo_spatial,evo_stop \
    [normal train_qwen.py args]

The script monkey-patches only the in-memory ``data_list`` function used by
``qwenvl.data.data_processor``. It does not edit ``qwenvl/data/__init__.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_evo_args(argv: List[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--evo_nav_jsonl", default="")
    parser.add_argument("--evo_va2l_jsonl", default="")
    parser.add_argument("--evo_spatial_jsonl", default="")
    parser.add_argument("--evo_stop_jsonl", default="")
    parser.add_argument("--evo_image_root", default="/")
    parser.add_argument("--evo_nav_image_root", default="")
    parser.add_argument("--evo_va2l_image_root", default="")
    parser.add_argument("--evo_spatial_image_root", default="/")
    parser.add_argument("--evo_stop_image_root", default="/")
    parser.add_argument("--evo_nav_name", default="evo_nav")
    parser.add_argument("--evo_va2l_name", default="evo_va2l")
    parser.add_argument("--evo_spatial_name", default="evo_spatial")
    parser.add_argument("--evo_stop_name", default="evo_stop")
    known, remaining = parser.parse_known_args(argv)
    return known, remaining


def main() -> None:
    evo_args, remaining = parse_evo_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining

    import qwenvl.data as data_pkg
    import qwenvl.data.data_processor as data_processor
    import evo_train_qwen as train_qwen

    original_data_list = data_pkg.data_list
    nav_image_root = evo_args.evo_nav_image_root or evo_args.evo_image_root
    va2l_image_root = evo_args.evo_va2l_image_root or evo_args.evo_image_root
    spatial_image_root = evo_args.evo_spatial_image_root or "/"
    stop_image_root = evo_args.evo_stop_image_root or "/"

    def patched_data_list(dataset_names):
        configs = []
        fallback_names = []
        for name in dataset_names:
            if name == evo_args.evo_nav_name:
                if not evo_args.evo_nav_jsonl:
                    raise ValueError(f"dataset {name!r} requested but --evo_nav_jsonl is empty")
                configs.append(
                    {
                        "annotation_path": evo_args.evo_nav_jsonl,
                        "data_path": nav_image_root,
                        "sampling_rate": 1.0,
                    }
                )
            elif name == evo_args.evo_va2l_name:
                if not evo_args.evo_va2l_jsonl:
                    raise ValueError(f"dataset {name!r} requested but --evo_va2l_jsonl is empty")
                configs.append(
                    {
                        "annotation_path": evo_args.evo_va2l_jsonl,
                        "data_path": va2l_image_root,
                        "sampling_rate": 1.0,
                    }
                )
            elif name == evo_args.evo_spatial_name:
                if not evo_args.evo_spatial_jsonl:
                    raise ValueError(f"dataset {name!r} requested but --evo_spatial_jsonl is empty")
                configs.append(
                    {
                        "annotation_path": evo_args.evo_spatial_jsonl,
                        "data_path": spatial_image_root,
                        "sampling_rate": 1.0,
                    }
                )
            elif name == evo_args.evo_stop_name:
                if not evo_args.evo_stop_jsonl:
                    raise ValueError(f"dataset {name!r} requested but --evo_stop_jsonl is empty")
                configs.append(
                    {
                        "annotation_path": evo_args.evo_stop_jsonl,
                        "data_path": stop_image_root,
                        "sampling_rate": 1.0,
                    }
                )
            else:
                fallback_names.append(name)
        if fallback_names:
            configs.extend(original_data_list(fallback_names))
        return configs

    data_pkg.data_list = patched_data_list
    data_processor.data_list = patched_data_list
    train_qwen.train()


if __name__ == "__main__":
    main()
