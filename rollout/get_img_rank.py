import argparse
import json
import os
import subprocess
import sys

# Quiet Habitat-Sim/Magnum C++ logs. This renderer only saves RGB frames, so
# missing HM3D semantic descriptor files are expected and otherwise flood stderr.
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("GLOG_minloglevel", "2")

import habitat
from PIL import Image
from habitat.config import read_write
from habitat.datasets.utils import VocabDict
from habitat.datasets.vln.r2r_vln_dataset import VLNDatasetV1
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.tasks.vln.vln import InstructionData, VLNEpisode
from habitat_baselines.config.default import get_config
from omegaconf import OmegaConf
from tqdm import tqdm

from habitat_extensions import measures


METADATA_PATH = "xxx/RSI-Nav/rollout/train_metadata.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-idx", "--idx", type=int, default=None, help="8 卡机器 ID，范围 0-7；只传该参数时启动 8 个本地进程")
    parser.add_argument("--rank_id", type=int, default=None, help="local rank 或旧版 global rank")
    parser.add_argument(
        "--episodes_dir",
        type=str,
        default="xxx/RSI-Nav/rollout/continuous_srdf",
        help="episodes jsonl 文件所在目录",
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default="xxx/RSI-Nav/rollout/continuous_srdf/anno",
        help="annotations jsonl 文件所在目录",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="xxx/RSI-Nav/rollout/vln_r2r.yaml",
        help="Habitat 配置文件路径",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="写入 Habitat config 的 gpu_device_id；配合 CUDA_VISIBLE_DEVICES 使用",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="xxx/RSI-Nav/data/trajectory_data/evo_traj",
        help="输出图片根目录",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="最多渲染并保存多少条 episode；默认处理全部",
    )
    return parser.parse_args()


def launch_local_workers(args):
    if args.idx is None:
        raise ValueError("launch_local_workers requires --idx")
    if not 0 <= args.idx <= 7:
        raise ValueError(f"--idx must be in [0, 7], got {args.idx}")

    script_path = os.path.abspath(__file__)
    processes = []
    for local_rank in range(8):
        global_rank = 8 * args.idx + local_rank
        cmd = [
            sys.executable,
            script_path,
            "--idx",
            str(args.idx),
            "--rank_id",
            str(local_rank),
            "--gpu_id",
            str(local_rank),
            "--episodes_dir",
            args.episodes_dir,
            "--annotations_dir",
            args.annotations_dir,
            "--config_path",
            args.config_path,
            "--output_root",
            args.output_root,
        ]
        if args.max_episodes is not None:
            cmd.extend(["--max_episodes", str(args.max_episodes)])

        print(
            f"[Launcher idx {args.idx}] start local_rank={local_rank} "
            f"global_rank={global_rank} gpu={local_rank}",
            flush=True,
        )
        processes.append((local_rank, subprocess.Popen(cmd)))

    exit_code = 0
    for local_rank, process in processes:
        status = process.wait()
        print(
            f"[Launcher idx {args.idx}] local_rank={local_rank} finished status={status}",
            flush=True,
        )
        if status != 0:
            exit_code = 1

    return exit_code


def build_rank_paths(episodes_dir: str, annotations_dir: str, rank_id: int):
    episode_path = os.path.join(episodes_dir, "srdf_episodes_merged.jsonl")
    annotation_path = os.path.join(annotations_dir, f"srdf_annotations_rank{rank_id}.jsonl")
    return episode_path, annotation_path


def load_annotations(annotation_path: str):
    annotations = []
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                annotations.append(json.loads(line))
    return annotations


def load_remaining_videos(annotations_dir: str):
    remaining_path = os.path.join(annotations_dir, "merged_annotations_remaining.jsonl")
    if not os.path.exists(remaining_path):
        return None

    remaining_videos = set()
    with open(remaining_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                if "video" in data:
                    remaining_videos.add(data["video"])
    return remaining_videos


def uniform_sample_indices(num_frames: int, num_samples: int):
    if num_frames <= 0 or num_samples <= 0:
        return []
    if num_samples >= num_frames:
        return list(range(num_frames))
    if num_samples == 1:
        return [0]

    last_index = num_frames - 1
    last_sample = num_samples - 1
    return [
        (idx * last_index + last_sample // 2) // last_sample
        for idx in range(num_samples)
    ]


def select_saved_frame_ids(action_count: int, nav_stride: int = 4, annotation_frames: int = 32):
    del nav_stride

    if action_count <= 0:
        return set()

    return set(uniform_sample_indices(action_count, annotation_frames))


def load_episodes(episode_path: str, scenes_dir: str, episode_ids=None):
    episode_ids = set(episode_ids) if episode_ids is not None else None
    episodes = []
    with open(episode_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if episode_ids is not None and str(raw["episode_id"]) not in episode_ids:
                continue

            scene_id = raw["scene_id"]
            if not os.path.isabs(scene_id):
                scene_id = os.path.join(scenes_dir, scene_id)

            episode = VLNEpisode(
                episode_id=str(raw["episode_id"]),
                trajectory_id=raw["trajectory_id"],
                scene_id=scene_id,
                start_position=raw["start_position"],
                start_rotation=raw["start_rotation"],
                goals=[NavigationGoal(**goal) for goal in raw["goals"]],
                instruction=InstructionData(
                    instruction_text=raw.get("instruction", {}).get("instruction_text", ""),
                    instruction_tokens=raw.get("instruction", {}).get("instruction_tokens", []),
                ),
                reference_path=raw["reference_path"],
                info=raw.get("info", {}),
            )
            episodes.append(episode)
    return episodes


def load_instruction_vocab():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    word_list = metadata.get("instruction_vocab", {}).get("word_list", [])
    return VocabDict(word_list=word_list)


def build_dataset(episodes, instruction_vocab):
    dataset = VLNDatasetV1(config=None)
    dataset.episodes = episodes
    dataset.instruction_vocab = instruction_vocab
    return dataset


def load_env(config_path: str, gpu_id: int, dataset, idx: int, local_rank: int, global_rank: int):
    config = get_config(config_path)
    with read_write(config):
        OmegaConf.set_struct(config, False)
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_id

    print_gpu_assignment(config, gpu_id, idx, local_rank, global_rank)
    env = habitat.Env(config=config, dataset=dataset)
    return env, config


def print_gpu_assignment(config, gpu_id: int, idx: int, local_rank: int, global_rank: int):
    render_gpu = config.habitat.simulator.habitat_sim_v0.gpu_device_id
    torch_cuda_current = "torch_unavailable"
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            torch_cuda_current = torch.cuda.current_device()
        else:
            torch_cuda_current = "cpu"
    except Exception as exc:
        torch_cuda_current = f"torch_error:{exc.__class__.__name__}"

    print(
        f"[idx {idx} local_rank {local_rank} global_rank {global_rank} pid {os.getpid()}] "
        f"Habitat render gpu_device_id={render_gpu}, "
        f"args.gpu_id={gpu_id}, torch_cuda_current={torch_cuda_current}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )


def main():
    args = parse_args()

    if args.idx is not None and args.rank_id is None:
        raise SystemExit(launch_local_workers(args))

    if args.idx is not None:
        if not 0 <= args.idx <= 7:
            raise ValueError(f"--idx must be in [0, 7], got {args.idx}")
        if args.rank_id is None or not 0 <= args.rank_id <= 7:
            raise ValueError(f"--rank_id must be a local rank in [0, 7], got {args.rank_id}")
        local_rank = args.rank_id
        global_rank = 8 * args.idx + local_rank
    else:
        # Backward-compatible single-rank mode: --rank_id is the original global rank.
        local_rank = args.rank_id if args.rank_id is not None else 0
        global_rank = local_rank
        args.idx = global_rank // 8
        args.gpu_id = args.gpu_id if args.rank_id is not None else 0
    print(global_rank)
    episode_path, annotation_path = build_rank_paths(args.episodes_dir, args.annotations_dir, global_rank)
    if not os.path.exists(episode_path):
        raise FileNotFoundError(f"Episode file not found: {episode_path}")
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    config = get_config(args.config_path)
    scenes_dir = config.habitat.dataset.scenes_dir

    goal_radius = 0.25

    annotations = load_annotations(annotation_path)
    annotation_map = {str(annot["id"]): annot for annot in annotations}

    all_episodes = load_episodes(episode_path, scenes_dir, episode_ids=annotation_map.keys())
    instruction_vocab = load_instruction_vocab()
    dataset = build_dataset(all_episodes, instruction_vocab)

    remaining_videos = load_remaining_videos(args.annotations_dir)

    env, config = load_env(args.config_path, args.gpu_id, dataset, args.idx, local_rank, global_rank)
    print(
        f"[idx {args.idx} local_rank {local_rank} global_rank {global_rank}] "
        f"episodes={len(all_episodes)} "
        f"episode_path={episode_path} annotation_path={annotation_path} gpu_id={args.gpu_id}"
    )

    rendered_count = 0
    skipped_remaining_count = 0
    skipped_existing_count = 0
    with tqdm(
        total=len(all_episodes),
        desc=f"Rank {global_rank}",
        unit="episode",
        dynamic_ncols=True,
    ) as pbar:
        for episode in all_episodes:
            if args.max_episodes is not None and rendered_count >= args.max_episodes:
                break

            episode_id = str(episode.episode_id)
            if episode_id not in annotation_map:
                raise KeyError(f"Episode {episode_id} not found in annotations: {annotation_path}")

            annotation = annotation_map[episode_id]
            video_id = annotation["video"]
            rgb_dir = os.path.join(args.output_root, video_id)
            selected_frame_ids = select_saved_frame_ids(len(annotation["actions"]))
            pbar.set_postfix(
                rendered=rendered_count,
                skipped_remaining=skipped_remaining_count,
                skipped_existing=skipped_existing_count,
                current=video_id,
            )

            # Check if this video should be processed based on remaining list
            if remaining_videos is not None and video_id not in remaining_videos:
                skipped_remaining_count += 1
                pbar.update(1)
                continue

            if os.path.isdir(rgb_dir):
                existing_frames = {f for f in os.listdir(rgb_dir) if f.endswith(".jpg")}
                expected_frames = {f"{frame_id:03d}.jpg" for frame_id in selected_frame_ids}
                if expected_frames.issubset(existing_frames):
                    skipped_existing_count += 1
                    pbar.update(1)
                    continue

            env.current_episode = episode
            ShortestPathFollower(sim=env.sim, goal_radius=goal_radius, return_one_hot=False)
            observation = env.reset()

            reference_actions = list(annotation["actions"][1:]) + [0]
            step_id = 0

            os.makedirs(rgb_dir, exist_ok=True)
            while not env.episode_over:
                if step_id in selected_frame_ids:
                    rgb = observation["rgb"]
                    Image.fromarray(rgb).convert("RGB").save(os.path.join(rgb_dir, f"{step_id:03d}.jpg"))

                if not reference_actions:
                    raise RuntimeError(
                        f"Episode {episode_id} ran out of reference actions before env terminated."
                    )

                action = reference_actions.pop(0)
                observation = env.step(action)
                step_id += 1

            rendered_count += 1
            pbar.update(1)

    env.close()


if __name__ == "__main__":
    main()
