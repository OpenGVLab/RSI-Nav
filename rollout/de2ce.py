"""
Discrete waypoint trajectory -> Continuous action sequence converter.

Based on: "Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments"

Action space:
  1 = MoveForward (25cm)
  2 = TurnLeft (15°)
  3 = TurnRight (15°)
  0 = Stop
"""

import gzip
import json
import math
import os
import random
import subprocess
import sys
from contextlib import contextmanager

# Quiet Habitat-Sim/Magnum C++ logs. Missing semantic files are expected for
# geometry-only HM3D conversion and otherwise flood stderr in 8-process runs.
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("GLOG_minloglevel", "2")
import numpy as np
import MatterSim
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import habitat
import habitat_sim
from habitat import get_config
from habitat.sims import make_sim
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from omegaconf import OmegaConf, read_write

# ============================================================
# Constants
# ============================================================
FORWARD_STEP = 0.25                # meters per forward action
TURN_ANGLE = math.radians(15)      # 15° per turn action

WIDTH = 336
HEIGHT = 336
VFOV = 90
HFOV = 90
DISCRETE_HEADING_STEP = math.radians(30)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QWEN3VL_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_INPUT_JSON = os.path.join(SCRIPT_DIR, "srdf_2476268.json")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "continuous_srdf")
DEFAULT_OUTPUT_EPISODES = os.path.join(DEFAULT_OUTPUT_DIR, "srdf_episodes.jsonl")
DEFAULT_OUTPUT_ANNOTATIONS = os.path.join(DEFAULT_OUTPUT_DIR, "srdf_annotations.jsonl")
DEFAULT_CONNECTIVITY_DIR = (
    "xxx/datasets/REVERIE/connectivity"
)
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "vln_r2r.yaml")
DEFAULT_SCENE_ROOT = os.path.join(QWEN3VL_DIR, "data/scene/hm3d")
DEFAULT_MAX_EPISODE_STEPS = 500


@contextmanager
def maybe_suppress_stderr(enabled=True):
    if not enabled:
        yield
        return

    saved_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)

# ============================================================
# Simulator builders
# ============================================================
def build_mattersim(connectivity_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim


def resolve_hm3d_scene_path(scan, scene_root=DEFAULT_SCENE_ROOT):
    scene_name = scan.split("-", 1)[1] if "-" in scan else scan
    candidates = [
        os.path.join(scene_root, scan, f"{scene_name}.basis.glb"),
        os.path.join(scene_root, scan, f"{scene_name}.glb"),
        os.path.join(scene_root, scan, f"{scan}.basis.glb"),
        os.path.join(scene_root, scan, f"{scan}.glb"),
    ]
    for scene_path in candidates:
        if os.path.exists(scene_path):
            return scene_path
    return candidates[0]


def make_scene_id(scene_path, scene_root=DEFAULT_SCENE_ROOT):
    scene_base = os.path.dirname(scene_root)
    try:
        return os.path.relpath(scene_path, scene_base)
    except ValueError:
        return scene_path


def build_habitat_sim(
    scan,
    config_path=DEFAULT_CONFIG_PATH,
    scene_root=DEFAULT_SCENE_ROOT,
    gpu_id=0,
    suppress_sim_logs=True,
):
    scene_path = resolve_hm3d_scene_path(scan, scene_root)
    config = get_config(config_path)
    with read_write(config):
        OmegaConf.set_struct(config, False)
        config.habitat.simulator.scene = scene_path
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = int(gpu_id)
        config.habitat.dataset.scenes_dir = os.path.dirname(scene_root)
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = HFOV
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = HEIGHT
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = WIDTH
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov = HFOV
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = HEIGHT
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = WIDTH
    print(
        f"[PID {os.getpid()}] scan={scan} scene={scene_path} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"habitat_visible_gpu={config.habitat.simulator.habitat_sim_v0.gpu_device_id}",
        flush=True,
    )
    with maybe_suppress_stderr(suppress_sim_logs):
        hsim = make_sim(id_sim=config.habitat.simulator.type, config=config.habitat.simulator)
        hsim.reset()
    return hsim, scene_path

def snap_heading_to_discrete_view(heading):                                                                                           
     """Match MatterSim's discretized viewing angles before converting to Habitat rotation."""                                         
     normalized_heading = heading % (2 * math.pi)                                                                                      
     snapped_heading = round(normalized_heading / DISCRETE_HEADING_STEP) * DISCRETE_HEADING_STEP                                       
     if math.isclose(snapped_heading, 2 * math.pi):                                                                                    
        snapped_heading = 0.0                                                                                                         
     return snapped_heading 


def iter_json_array(input_path, chunk_size=1024 * 1024):
    """Stream dict records from a top-level JSON array."""
    open_fn = gzip.open if input_path.endswith(".gz") else open
    decoder = json.JSONDecoder()
    buffer = ""
    pos = 0
    in_array = False

    with open_fn(input_path, "rt", encoding="utf-8") as f:
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
                        raise ValueError(f"{input_path} is expected to be a top-level JSON array")
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
                    raise ValueError(f"{input_path} contains a non-object item: {type(item).__name__}")
                yield item
                pos = next_pos

            if pos:
                buffer = buffer[pos:]
                pos = 0

    if buffer.strip() not in ("", "]"):
        raise ValueError(f"{input_path} ended before a complete JSON array item could be parsed")


def iter_discrete_data(input_path):
    """Stream trajectories from .json/.jsonl and their .gz variants."""
    open_fn = gzip.open if input_path.endswith(".gz") else open

    if input_path.endswith(".jsonl") or input_path.endswith(".jsonl.gz"):
        with open_fn(input_path, "rt", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL record in {input_path} at line {line_no}: {exc}"
                    ) from exc
                if not isinstance(item, dict):
                    raise ValueError(f"{input_path} line {line_no} is not a JSON object")
                yield item
        return

    yield from iter_json_array(input_path)


def load_discrete_data(input_path):
    """Load trajectories from .json/.jsonl and their .gz variants."""
    return list(iter_discrete_data(input_path))


def count_discrete_data(input_path, show_pbar=False):
    """Count records in .json/.jsonl and their .gz variants."""
    iterator = iter_discrete_data(input_path)
    if show_pbar:
        iterator = tqdm(iterator, desc="Counting records", unit="item", dynamic_ncols=True)
    return sum(1 for _ in iterator)


def get_split_dir(input_path, world_size):
    return os.path.join(os.path.dirname(input_path), f"splits_w{world_size}")


def split_suffix_for(input_path):
    return ".jsonl" if input_path.endswith(".jsonl") or input_path.endswith(".jsonl.gz") else ".json"


def split_file_for(input_path, world_size, rank_id):
    return os.path.join(get_split_dir(input_path, world_size), f"split_rank{rank_id}{split_suffix_for(input_path)}")


def prepare_rank_splits(input_path, world_size, force=False):
    """Create evenly sized rank split files before launching workers."""
    split_dir = get_split_dir(input_path, world_size)
    os.makedirs(split_dir, exist_ok=True)
    split_files = [split_file_for(input_path, world_size, rank) for rank in range(world_size)]
    manifest_path = os.path.join(split_dir, "manifest.json")

    if not force and os.path.exists(manifest_path) and all(os.path.exists(p) for p in split_files):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("input_json") == os.path.abspath(input_path) and manifest.get("world_size") == world_size:
            print(f"[Launcher] using existing splits from {split_dir}", flush=True)
            return manifest

    print(f"[Launcher] preparing {world_size} even splits in {split_dir}", flush=True)
    total = count_discrete_data(input_path, show_pbar=True)
    base = total // world_size
    remainder = total % world_size
    sizes = [base + (1 if rank < remainder else 0) for rank in range(world_size)]
    boundaries = []
    cursor = 0
    for size in sizes:
        boundaries.append((cursor, cursor + size))
        cursor += size

    suffix = split_suffix_for(input_path)
    writers = [open(p, "w", encoding="utf-8") for p in split_files]
    first_flags = [True] * world_size
    try:
        if suffix == ".json":
            for f in writers:
                f.write("[\n")

        rank = 0
        with tqdm(total=total, desc="Writing rank splits", unit="item", dynamic_ncols=True) as pbar:
            for idx, item in enumerate(iter_discrete_data(input_path)):
                while rank + 1 < world_size and idx >= boundaries[rank][1]:
                    rank += 1
                if suffix == ".jsonl":
                    writers[rank].write(json.dumps(item, ensure_ascii=False) + "\n")
                else:
                    if not first_flags[rank]:
                        writers[rank].write(",\n")
                    writers[rank].write(json.dumps(item, ensure_ascii=False))
                    first_flags[rank] = False
                pbar.update(1)

        if suffix == ".json":
            for f in writers:
                f.write("\n]\n")
    finally:
        for f in writers:
            f.close()

    manifest = {
        "input_json": os.path.abspath(input_path),
        "world_size": world_size,
        "total": total,
        "sizes": sizes,
        "split_files": split_files,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Launcher] split sizes: {sizes}", flush=True)
    return manifest


# ============================================================
# Main conversion
# ============================================================
def convert_discrete_to_continuous(
    input_json,
    output_episodes_json,
    output_annotations_json,
    rank_id=0,
    world_size=8,
    metadata_json=None,
    dataset_name="srdf",
    connectivity_dir="connectivity_mp3d",
    config_path=DEFAULT_CONFIG_PATH,
    scene_root=DEFAULT_SCENE_ROOT,
    gpu_id=0,
    suppress_sim_logs=True,
    max_items=None,
    max_episode_steps=DEFAULT_MAX_EPISODE_STEPS,
):
    print(
        f"[Rank {rank_id}] PID={os.getpid()} physical_gpu={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"habitat_visible_gpu={gpu_id}",
        flush=True,
    )
    print(f"Reading discrete data from: {input_json}")
    split_file = split_file_for(input_json, world_size, rank_id)
    if not os.path.exists(split_file):
        prepare_rank_splits(input_json, world_size)
    print(f"[Rank {rank_id}] Loading split file: {split_file}", flush=True)
    discrete_data = load_discrete_data(split_file)
    print(f"[Rank {rank_id}] Loaded {len(discrete_data):,} items from its split", flush=True)

    metadata = {}
    if metadata_json and os.path.exists(metadata_json):
        with open(metadata_json, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"Loaded metadata keys: {list(metadata.keys())}")
    else:
        print("Warning: No valid metadata_json provided. Output episodes won't have global vocab/metadata.")

    # Group by scan
    scan_groups = {}
    for item in discrete_data:
        scan_groups.setdefault(item['scan'], []).append(item)
    scan_ids = list(scan_groups.keys())
    random.shuffle(scan_ids)

    num_episodes_written = 0
    num_annotations_written = 0
    processed_items = 0
    skipped_step_limit = 0

    # Stream episodes/annotations to disk immediately so partial progress is preserved.
    with open(output_episodes_json, "w", encoding="utf-8") as ep_f, open(output_annotations_json, "w", encoding="utf-8") as ann_f:
        pbar_total = len(discrete_data) if max_items is None else min(len(discrete_data), max_items)
        with tqdm(
            total=pbar_total,
            desc=f"Rank {rank_id} items",
            position=rank_id,
            dynamic_ncols=True,
        ) as pbar:
            for scan_id in scan_ids:
                if scan_id == 'V2XKFyX4ASd' or scan_id == 'YmJkqBEsHnH':
                    pbar.update(min(len(scan_groups[scan_id]), pbar_total - pbar.n))
                    continue
                items = scan_groups[scan_id]
                if max_items is not None and processed_items >= max_items:
                    break

                msim = build_mattersim(connectivity_dir)
                hsim, scene_path = build_habitat_sim(
                    scan_id,
                    config_path=config_path,
                    scene_root=scene_root,
                    gpu_id=gpu_id,
                    suppress_sim_logs=suppress_sim_logs,
                )
                scene_id = make_scene_id(scene_path, scene_root)

                for item in items:
                    if max_items is not None and processed_items >= max_items:
                        break

                    path = item['path']
                    path_id = item.get('path_id')
                    heading = snap_heading_to_discrete_view(item['heading']) #heading = item['heading']

                    # --- Build reference_path & precise start elevation via MatterSim ---
                    reference_path = []
                    habitat_rotation = None
                    initial_elevation = 0.0

                    for vp_idx, vp_id in enumerate(path):
                        msim.newEpisode([scan_id], [vp_id], [0], [0])
                        state = msim.getState()[0]
                        x, y, z = state.location.x, state.location.y, state.location.z
                        
                        if vp_idx == 0:
                            initial_elevation = state.elevation

                        # Snap to the 3D NavMesh to obtain the precise elevation.
                        approx_pos = [x, z - 1.25, -y]
                        snapped = hsim.pathfinder.snap_point(approx_pos)
                        if not math.isnan(snapped[0]):
                            habitat_position = [x, snapped[1], -y]
                        else:
                            habitat_position = approx_pos
                        reference_path.append(habitat_position)

                    # --- Check reachability ---
                    geodesic_distance = hsim.geodesic_distance(reference_path[0], reference_path[-1])
                    if geodesic_distance >= 100.0:
                        pbar.update(1)
                        continue

                    # --- Initialize Agent Physics State in Habitat ---
                    agent = hsim.get_agent(0)
                    agent_state = agent.get_state()
                    
                    # 1. Set the initial position in the simulator.
                    agent_state.position = reference_path[0]
                    
                    # 2. Set the initial quaternion rotation from the heading and elevation.
                    hm3d_h = np.array([0, 2 * math.pi - heading, 0])
                    hm3d_e = np.array([initial_elevation, 0, 0])
                    rotvec = R.from_rotvec(hm3d_h) * R.from_rotvec(hm3d_e)
                    quat_scipy = rotvec.as_quat() # [x, y, z, w]
                    
                    # Convert to the quaternion format expected by Habitat.
                    exact_start_rotation = habitat_sim.utils.common.quat_from_coeffs(quat_scipy)
                    agent_state.rotation = exact_start_rotation
                    agent.set_state(agent_state)

                    habitat_rotation = quat_scipy.tolist()

                    # --- Use 3D Physics Tracker for Continuous Actions ---
                    # Use goal_radius=0.25 for a waypoint tolerance of 0.25 m.
                    follower = ShortestPathFollower(hsim, goal_radius=0.25, return_one_hot=False)
                    continuous_actions = []
                    hit_step_limit = False

                    # Navigate toward each subsequent waypoint.
                    for tgt_pos in reference_path[1:]:
                        while True:
                            if len(continuous_actions) >= max_episode_steps:
                                hit_step_limit = True
                                break

                            # Get the next action.
                            action = follower.get_next_action(tgt_pos)
                            
                            # Returns 0 (STOP) or None when within goal_radius of tgt_pos or when no route is available.
                            if action is None or action == 0:
                                break
                            
                            # Step the simulator to apply collision handling and update the position.
                            hsim.step(action)
                            continuous_actions.append(action)

                        if hit_step_limit:
                            break

                    if hit_step_limit:
                        skipped_step_limit += 1
                        pbar.update(1)
                        continue

                    # Prepend the dummy action token (-1) used for VLN training.
                    actions_with_dummy = [-1] + continuous_actions
                    processed_items += 1

                    # --- Output Formatting ---
                    # Keep one record per path_id. Instructions are intentionally omitted
                    # because they will be regenerated later and are large.
                    ce_episode = {
                        "path_id": path_id,
                        "episode_id": path_id,
                        "trajectory_id": path_id,
                        "scene_id": scene_id,
                        "scan": scan_id,
                        "start_position": reference_path[0],
                        "start_rotation": habitat_rotation,
                        "info": {"geodesic_distance": geodesic_distance},
                        "goals": [{"position": reference_path[-1], "radius": 3.0}],
                        "reference_path": reference_path
                    }
                    ep_f.write(json.dumps(ce_episode, ensure_ascii=False) + "\n")
                    ep_f.flush()
                    num_episodes_written += 1

                    ce_annotation = {
                        "path_id": path_id,
                        "id": path_id,
                        "video": f"images/{scan_id}_{dataset_name}_{path_id}",
                        "actions": actions_with_dummy
                    }
                    ann_f.write(json.dumps(ce_annotation, ensure_ascii=False) + "\n")
                    ann_f.flush()
                    num_annotations_written += 1

                    pbar.update(1)

                hsim.close()

    print(f"\n✅ [Rank {rank_id}] Done! Generated {num_episodes_written} continuous episodes.")
    print(f"[Rank {rank_id}] Episodes saved to: {output_episodes_json}")
    print(f"[Rank {rank_id}] Annotations saved to: {output_annotations_json} ({num_annotations_written} lines)")
    print(f"[Rank {rank_id}] Skipped step-limit items: {skipped_step_limit}")

    return num_episodes_written, num_annotations_written


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank_id", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input_json", default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output_episodes", default=DEFAULT_OUTPUT_EPISODES)
    parser.add_argument("--output_annotations", default=DEFAULT_OUTPUT_ANNOTATIONS)
    # Specify the metadata JSON path prepared in the first step.
    parser.add_argument("--metadata_json", default="./train_metadata.json")
    parser.add_argument("--dataset_name", default="srdf")
    parser.add_argument("--connectivity_dir", default=DEFAULT_CONNECTIVITY_DIR)
    parser.add_argument("--config_path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--scene_root", default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--max_items", type=int, default=100000000,
                        help="Only process N items (for debugging)")
    parser.add_argument("--max_episode_steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS,
                        help="Skip records whose generated Habitat action sequence reaches this length")
    parser.add_argument("--show_sim_logs", action="store_true",
                        help="Show Habitat-Sim/Magnum C++ warnings on stderr")
    args = parser.parse_args()

    if not args.worker:
        processes = []
        manifest = prepare_rank_splits(args.input_json, args.num_processes)
        print(
            f"[Launcher] starting {args.num_processes} workers with {args.num_gpus} GPUs; "
            f"total_items={manifest['total']:,}",
            flush=True,
        )
        for rank in range(args.num_processes):
            gpu_id = rank % args.num_gpus
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--worker",
                "--rank_id", str(rank),
                "--world_size", str(args.num_processes),
                "--gpu_id", str(gpu_id),
                "--num_gpus", str(args.num_gpus),
                "--input_json", args.input_json,
                "--output_episodes", args.output_episodes,
                "--output_annotations", args.output_annotations,
                "--metadata_json", args.metadata_json,
                "--dataset_name", args.dataset_name,
                "--connectivity_dir", args.connectivity_dir,
                "--config_path", args.config_path,
                "--scene_root", args.scene_root,
                "--max_items", str(args.max_items),
                "--max_episode_steps", str(args.max_episode_steps),
            ]
            if args.show_sim_logs:
                cmd.append("--show_sim_logs")
            print(
                f"[Launcher] rank={rank} physical_gpu={gpu_id} "
                f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}",
                flush=True,
            )
            processes.append(subprocess.Popen(cmd, env=env))

        return_codes = [proc.wait() for proc in processes]
        failed = [idx for idx, code in enumerate(return_codes) if code != 0]
        if failed:
            raise SystemExit(f"Workers failed: {failed}; return_codes={return_codes}")
        print("[Launcher] all workers finished successfully", flush=True)
        raise SystemExit(0)

    if args.gpu_id is None:
        args.gpu_id = args.rank_id % args.num_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    render_gpu_id = 0
    print(
        f"[Rank {args.rank_id}] assigned physical_gpu={args.gpu_id}; "
        f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}; "
        f"habitat_visible_gpu={render_gpu_id} (visible GPU 0 maps to physical GPU {args.gpu_id})",
        flush=True,
    )

    if args.output_episodes.endswith(".jsonl"):
        ep_name_base = args.output_episodes[:-6]
    elif args.output_episodes.endswith(".json.gz"):
        ep_name_base = args.output_episodes[:-8]
    elif args.output_episodes.endswith(".json"):
        ep_name_base = args.output_episodes[:-5]
    else:
        ep_name_base = args.output_episodes
    out_ep = f"{ep_name_base}_rank{args.rank_id}.jsonl"
    if args.output_annotations.endswith(".jsonl"):
        ann_name_base = args.output_annotations[:-6]
    elif args.output_annotations.endswith(".json"):
        ann_name_base = args.output_annotations[:-5]
    else:
        ann_name_base = args.output_annotations
    out_ann = f"{ann_name_base}_rank{args.rank_id}.jsonl"

    os.makedirs(os.path.dirname(out_ep), exist_ok=True)
    os.makedirs(os.path.dirname(out_ann), exist_ok=True)

    convert_discrete_to_continuous(
        input_json=args.input_json,
        output_episodes_json=out_ep,
        output_annotations_json=out_ann,
        rank_id=args.rank_id,
        world_size=args.world_size,
        metadata_json=args.metadata_json,
        dataset_name=args.dataset_name,
        connectivity_dir=args.connectivity_dir,
        config_path=args.config_path,
        scene_root=args.scene_root,
        gpu_id=render_gpu_id,
        suppress_sim_logs=not args.show_sim_logs,
        max_items=args.max_items,
        max_episode_steps=args.max_episode_steps,
    )
