#!/usr/bin/env python3
import argparse
import json
import os
import tempfile
from typing import Any, Dict, Iterator, Set


DEFAULT_BIG_JSON = "mantis.hm3d_round0_topk.3_enc.json"
DEFAULT_USED_JSON = "mantis.hm3d_round3_greedy_ndtw0.9_ranked_414k_rouge0.85.json"


def iter_json_array(path: str, chunk_size: int = 1024 * 1024) -> Iterator[Dict[str, Any]]:
    """Stream objects from a top-level JSON array without loading the whole file."""
    decoder = json.JSONDecoder()
    buffer = ""
    pos = 0
    in_array = False

    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer += chunk

            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1

                if not in_array:
                    if pos >= len(buffer):
                        break
                    if buffer[pos] != "[":
                        raise ValueError(f"{path} is expected to be a top-level JSON array")
                    pos += 1
                    in_array = True
                    continue

                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos >= len(buffer):
                    break

                if buffer[pos] == ",":
                    pos += 1
                    continue
                if buffer[pos] == "]":
                    return

                try:
                    item, next_pos = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    break

                if not isinstance(item, dict):
                    raise ValueError(f"{path} contains a non-object item: {type(item).__name__}")
                yield item
                pos = next_pos

            if pos:
                buffer = buffer[pos:]
                pos = 0

    if buffer.strip() not in ("", "]"):
        raise ValueError(f"{path} ended before a complete JSON array item could be parsed")


def collect_path_ids(path: str) -> Set[Any]:
    path_ids: Set[Any] = set()
    total = 0
    missing = 0

    for item in iter_json_array(path):
        total += 1
        if "path_id" in item:
            path_ids.add(item["path_id"])
        else:
            missing += 1

        if total % 100000 == 0:
            print(f"[used] scanned={total:,} unique_path_id={len(path_ids):,}", flush=True)

    print(
        f"[used] done scanned={total:,} unique_path_id={len(path_ids):,} missing_path_id={missing:,}",
        flush=True,
    )
    return path_ids


def write_filtered_json(big_path: str, used_path_ids: Set[Any], output_dir: str) -> str:
    fd, tmp_path = tempfile.mkstemp(prefix="srdf_tmp_", suffix=".json", dir=output_dir)
    kept = 0
    skipped = 0
    missing = 0
    scanned = 0

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write("[\n")
            first = True

            for item in iter_json_array(big_path):
                scanned += 1
                path_id = item.get("path_id")

                if "path_id" not in item:
                    missing += 1
                elif path_id in used_path_ids:
                    skipped += 1
                    if scanned % 100000 == 0:
                        print(
                            f"[big] scanned={scanned:,} kept={kept:,} skipped={skipped:,}",
                            flush=True,
                        )
                    continue

                if not first:
                    out.write(",\n")
                json.dump(item, out, ensure_ascii=False, separators=(",", ":"))
                first = False
                kept += 1

                if scanned % 100000 == 0:
                    print(
                        f"[big] scanned={scanned:,} kept={kept:,} skipped={skipped:,}",
                        flush=True,
                    )

            out.write("\n]\n")

        final_path = os.path.join(output_dir, f"srdf_{kept}.json")
        if os.path.exists(final_path):
            raise FileExistsError(f"Output already exists: {final_path}")
        os.replace(tmp_path, final_path)

        print(
            f"[big] done scanned={scanned:,} kept={kept:,} skipped={skipped:,} missing_path_id={missing:,}",
            flush=True,
        )
        return final_path
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=(
            "Remove records from the large SRDF JSON whose path_id already appears "
            "in the used 400k JSON, then write srdf_<kept_count>.json."
        )
    )
    parser.add_argument("--big-json", default=os.path.join(script_dir, DEFAULT_BIG_JSON))
    parser.add_argument("--used-json", default=os.path.join(script_dir, DEFAULT_USED_JSON))
    parser.add_argument("--output-dir", default=script_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    used_path_ids = collect_path_ids(args.used_json)
    output_path = write_filtered_json(args.big_json, used_path_ids, args.output_dir)
    print(f"wrote: {output_path}", flush=True)


if __name__ == "__main__":
    main()
