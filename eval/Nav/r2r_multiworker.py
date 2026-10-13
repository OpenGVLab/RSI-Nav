import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
os.environ.setdefault("ATTN_IMPLEMENTATION", "flash_attention_2")
import re
import time
import tqdm
import copy
import json
import argparse
import itertools
import glob
import numpy as np

from typing import Any, List
from omegaconf import OmegaConf
from PIL import Image
from collections import OrderedDict

import torch

import habitat
from habitat import Env
# from habitat_extensions import measures  # noqa: F401  registers OracleSuccess / OracleNavigationError
from habitat.config.default import get_agent_config, get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat_extensions.measures import (
    OracleSuccessMeasurementConfig,
    OracleNavigationErrorMeasurementConfig,
)
from habitat.utils.visualizations.utils import images_to_video, observations_to_image

from utils.dist import *

from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)
from qwenvl.train.fa2_compat import maybe_patch_qwen35_flash_attention
# from transformers import Qwen3VLForConditionalGeneration


DEBUG = False
EVAL_SCHEMA_VERSION = 3
_DIST_GET_RANK = get_rank
_DIST_GET_WORLD_SIZE = get_world_size
_DIST_INIT_DISTRIBUTED_MODE = init_distributed_mode
_RUNTIME_RANK = None
_RUNTIME_WORLD_SIZE = None


def _env_int(name: str, default: int = 0) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    return int(value)


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y"}


def _using_platform_data_parallel() -> bool:
    return (
        _env_true("DISTRIBUTED_JOB")
        and "NODE_RANK" in os.environ
        and "NODE_COUNT" in os.environ
        and "PROC_PER_NODE" in os.environ
    )


def get_rank():
    if _RUNTIME_RANK is not None:
        return int(_RUNTIME_RANK)
    return _DIST_GET_RANK()


def get_world_size():
    if _RUNTIME_WORLD_SIZE is not None:
        return int(_RUNTIME_WORLD_SIZE)
    return _DIST_GET_WORLD_SIZE()


def init_eval_runtime(args):
    """Initialize either torch distributed or platform file-sharded evaluation.

    eval.sh launches one local torchrun per node. In that mode torchrun's
    RANK/WORLD_SIZE are only local to the node, so use the platform-injected
    NODE_* values for global data parallel sharding and aggregate from files.
    """
    global _RUNTIME_RANK, _RUNTIME_WORLD_SIZE

    if _using_platform_data_parallel():
        node_rank = _env_int("NODE_RANK")
        node_count = _env_int("NODE_COUNT", 1)
        proc_per_node = _env_int("PROC_PER_NODE", max(torch.cuda.device_count(), 1))
        local_rank = _env_int("LOCAL_RANK", int(getattr(args, "local_rank", 0)))
        local_world_size = _env_int("LOCAL_WORLD_SIZE", proc_per_node)
        if local_world_size != proc_per_node:
            print(
                f"[runtime] LOCAL_WORLD_SIZE={local_world_size} differs from "
                f"PROC_PER_NODE={proc_per_node}; using PROC_PER_NODE for global sharding.",
                flush=True,
            )

        global_rank = node_rank * proc_per_node + local_rank
        global_world_size = node_count * proc_per_node
        num_gpus = torch.cuda.device_count()
        gpu = local_rank % num_gpus if num_gpus > 0 else local_rank

        args.rank = global_rank
        args.world_size = global_world_size
        args.local_rank = local_rank
        args.gpu = gpu
        args.distributed = False
        args.platform_data_parallel = True
        _RUNTIME_RANK = global_rank
        _RUNTIME_WORLD_SIZE = global_world_size

        if torch.cuda.is_available():
            torch.cuda.set_device(gpu)
        setup_for_distributed(global_rank == 0)
        print(
            "[runtime] platform data parallel "
            f"NODE_RANK={node_rank}/{node_count} LOCAL_RANK={local_rank} "
            f"PROC_PER_NODE={proc_per_node} rank={global_rank} "
            f"world_size={global_world_size} gpu={gpu} "
            f"MASTER_ADDR={os.environ.get('MASTER_ADDR', '<unset>')} "
            f"JOB_ID={os.environ.get('JOB_ID', '<unset>')}",
            force=True,
            flush=True,
        )
        return

    args.platform_data_parallel = False
    _DIST_INIT_DISTRIBUTED_MODE(args)


class VLNEvaluator:
    """R2R evaluator for Qwen3VL / Qwen3.5-VL.

    Aligned with the training data format defined by
    xxx/RSI-Nav/data/datasets/r2r/json/r2r_qwen_32step_episodes.jsonl
    The training sample layout for one 32-step "slice" is:

      user:      "<prompt-with-instruction>
                  [ These are your historical observations: <image>*8 ]   # only when slice_idx>0
                  This is your current view: <image>"
      assistant: "<4 actions>"
      user:      "<conjunction> <image>"
      assistant: "<4 actions>"
      ...                                       # up to 8 turns / 32 env steps
      user:      "<conjunction> <image>"        # pending turn we want a reply for

    During evaluation we call the model every ``num_future_steps`` (=4) env steps
    and grow the conversation turn by turn, mirroring the training format
    exactly. Every ``num_frames`` (=32) env steps we close the slice and start
    a new one whose history is uniformly sampled from all frames before the
    slice start.
    """

    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: str = None,
        model: Any = None,
        processor: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else args.device)
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        self.epoch = epoch
        self.config_path = config_path
        self.config = get_habitat_config(config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors
        self.success_distance = 3.0

        with habitat.config.read_write(self.config):
            self.config.habitat.dataset.split = self.split
            if torch.cuda.is_available():
                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.gpu
            measurements = {
                "oracle_success": OracleSuccessMeasurementConfig(
                    success_distance=self.success_distance
                ),
                "oracle_navigation_error": OracleNavigationErrorMeasurementConfig(),
            }
            if self.save_video:
                measurements.update(
                    {
                        "top_down_map": TopDownMapMeasurementConfig(
                            map_padding=3,
                            map_resolution=1024,
                            draw_source=True,
                            draw_border=True,
                            draw_shortest_path=True,
                            draw_view_points=True,
                            draw_goal_positions=True,
                            draw_goal_aabbs=True,
                            fog_of_war=FogOfWarConfig(
                                draw=True,
                                visibility_dist=5.0,
                                fov=90,
                            ),
                        ),
                        "collisions": CollisionsMeasurementConfig(),
                    }
                )
            self.config.habitat.task.measurements.update(measurements)

        if DEBUG:
            print(f"config = {type(self.config)}")
            print(OmegaConf.to_yaml(self.config))

        self.model = model
        self.processor = processor
        self.model_type = getattr(getattr(model, "config", None), "model_type", "")

        # Prompt body identical to the training data. The "<instruction>"
        # placeholder is filled in at runtime.
        self.prompt = (
            "You are an autonomous navigation assistant. Your task is to <instruction> "
            "Devise an action sequence to follow the instruction using the four actions: "
            "TURN LEFT (left) or TURN RIGHT (right) by 15 degrees, "
            "MOVE FORWARD (forward) by 25 centimeters, or STOP (stop)."
        )

        self.actions2idx = OrderedDict({
            "stop": [0],
            "forward": [1],
            "left": [2],
            "right": [3],
        })
        self.idx2actions = {
            0: "stop",
            1: "forward",
            2: "left",
            3: "right",
        }

        # Conjunctions used as the leading text of every follow-up user turn,
        # matching the training data.
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]

        self.num_frames = args.num_frames                # 32
        self.num_future_steps = args.num_future_steps    # 4
        self.num_history = args.num_history              # 8

    # ------------------------------------------------------------------
    # Env helpers
    # ------------------------------------------------------------------
    def config_env(self) -> Env:
        rank = get_rank()
        render_gpu = self.config.habitat.simulator.habitat_sim_v0.gpu_device_id
        torch_cuda_current = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        print(
            f"[rank {rank} pid {os.getpid()}] Habitat render gpu_device_id={render_gpu}, "
            f"args.gpu={self.args.gpu}, torch_device={self.device}, "
            f"torch_cuda_current={torch_cuda_current}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
            force=True,
            flush=True,
        )
        return Env(config=self.config)

    def actions2text(self, actions: List[int]) -> str:
        return ",".join(self.idx2actions[int(a)] for a in actions)

    @staticmethod
    def normalize_instruction(instruction: str) -> str:
        """Match the normalized instruction text stored in the R2R SFT data."""
        instruction = str(instruction or "").strip()
        if instruction and instruction[-1] not in ".!?":
            instruction += "."
        return instruction

    def history_sample_indices(self, history_len: int) -> List[int]:
        """Reproduce the R2R32 generator's fixed-stride history sampling."""
        if history_len <= 0:
            return []
        if history_len <= self.num_history:
            return list(range(history_len))
        stride = max(history_len // self.num_history, 1)
        return list(range(0, history_len, stride))[: self.num_history]

    # ------------------------------------------------------------------
    # Conversation builder (matches r2r_qwen_32step_episodes.jsonl)
    # ------------------------------------------------------------------
    def build_conversation(
        self,
        instruction: str,
        history_imgs: List[Image.Image],
        slice_rgbs: List[Image.Image],
        slice_actions: List[int],
        slice_conjunctions: List[str],
    ):
        """Build a Qwen3VL chat-template-ready conversation for one slice.

        Args:
          instruction:        the task instruction text.
          history_imgs:       all RGB frames captured BEFORE the current slice
                              started (used for the 8-frame uniformly-sampled
                              history). Empty for the first slice.
          slice_rgbs:         RGB frames captured INSIDE the current slice
                              (>=1). The last element is the current view we
                              want the next action chunk for.
          slice_actions:      actions already emitted in this slice. Must be a
                              multiple of num_future_steps; produces
                              ``len // num_future_steps`` completed assistant
                              turns in the dialogue.
          slice_conjunctions: conjunction strings already chosen for the
                              follow-up user turns inside this slice. There
                              should be one entry per follow-up user turn,
                              i.e. ``len(slice_rgbs) - 1`` entries.
        """
        num_future = self.num_future_steps  # 4
        num_history = self.num_history       # 8

        # ---- 1. First user turn (slice opener) ----
        prompt_text = self.prompt.replace(
            "<instruction>", self.normalize_instruction(instruction)
        )

        first_user_parts: list = []  # list of {"type":"text"|"image",...}
        first_user_parts.append({"type": "text", "text": prompt_text})

        if len(history_imgs) > 0:
            sampled_idx = self.history_sample_indices(len(history_imgs))
            # The training data writes literally:
            #   " These are your historical observations: <image><image>..."
            # The Qwen3VL chat template expands {"type":"image"} entries into
            # the model's image placeholder tokens, so the final string the
            # tokenizer sees matches the training string verbatim.
            first_user_parts.append(
                {"type": "text", "text": " These are your historical observations:"}
            )
            for i in sampled_idx:
                img = history_imgs[int(i)]
                first_user_parts.append({"type": "image", "image": img})
            first_user_parts.append({"type": "text", "text": " This is your current view:"})
        else:
            first_user_parts.append({"type": "text", "text": " This is your current view:"})

        # current-view image of the opening turn = first frame in the slice
        first_user_parts.append({"type": "image", "image": slice_rgbs[0]})

        conversation = [{"role": "user", "content": first_user_parts}]

        # ---- 2. Completed (assistant, user) pairs inside this slice ----
        n_done_turns = len(slice_actions) // num_future
        # sanity: we must have one conjunction recorded per follow-up user turn
        # that we've shown to the model so far. Follow-up user turns are
        # slice_rgbs[num_future], slice_rgbs[2*num_future], ...
        for turn_idx in range(n_done_turns):
            acts = slice_actions[turn_idx * num_future:(turn_idx + 1) * num_future]
            conversation.append({
                "role": "assistant",
                "content": [{"type": "text", "text": self.actions2text(acts)}],
            })

            next_frame_pos = (turn_idx + 1) * num_future
            if next_frame_pos < len(slice_rgbs):
                # follow-up user turn
                conj_idx = turn_idx  # 0-th follow-up uses slice_conjunctions[0]
                conj = slice_conjunctions[conj_idx]
                conversation.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": conj.strip()},
                        {"type": "image", "image": slice_rgbs[next_frame_pos]},
                    ],
                })

        return conversation

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------
    def eval_action(self, idx) -> None:
        rank = get_rank()
        print(f"[rank {rank}] creating Habitat Env...", force=True, flush=True)
        env = self.config_env()
        print(
            f"[rank {rank}] Habitat Env created, episodes={len(env.episodes)}",
            force=True,
            flush=True,
        )
        scene_episode_dict = {}
        for episode in env.episodes:
            scene_episode_dict.setdefault(episode.scene_id, []).append(episode)
        print(f"[rank {rank}] scenes={len(scene_episode_dict)}", flush=True)

        all_episode_items = []
        for scene in sorted(scene_episode_dict.keys()):
            scene_id = scene.split('/')[-2]
            for episode in scene_episode_dict[scene]:
                episode_instruction = (
                    episode.instruction.instruction_text
                    if 'objectnav' not in self.config_path
                    else episode.object_category
                )
                all_episode_items.append(
                    (
                        str(scene_id),
                        str(episode.episode_id),
                        str(episode_instruction),
                    )
                )
        all_episode_items.sort()
        assigned_episode_keys = {
            item for pos, item in enumerate(all_episode_items) if pos % self.env_num == idx
        }
        print(
            f"[rank {rank}] assigned episodes={len(assigned_episode_keys)} "
            f"of total={len(all_episode_items)} world_size={self.env_num}",
            force=True,
            flush=True,
        )

        sucs, spls, oss, ones = [], [], [], []
        done_res = set()
        if DEBUG:
            print(self.output_path)
        os.makedirs(self.output_path, exist_ok=True)
        result_file = os.path.join(self.output_path, f"result_rank{rank}.jsonl")
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding="utf-8") as f:
                for line in f.readlines():
                    res = json.loads(line)
                    try:
                        if res.get("eval_schema_version") != EVAL_SCHEMA_VERSION:
                            raise RuntimeError(
                                f"Legacy Nav result found in {result_file}. "
                                "Use a new output directory so official and legacy metrics are not mixed."
                            )
                        done_res.add(
                            (
                                str(res["scene_id"]),
                                str(res["episode_id"]),
                                str(res["episode_instruction"]),
                            )
                        )
                    except Exception:
                        raise
                    if res.get('success') is not None:
                        sucs.append(float(res['success']))
                        spls.append(res['spl'])
                        oss.append(res['os'])
                        ones.append(res['ne'])

        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            print(f"[rank {rank}] scene_id = {scene_id}", flush=True)
            assigned_episodes = []
            for episode in episodes:
                episode_instruction = (
                    episode.instruction.instruction_text
                    if 'objectnav' not in self.config_path
                    else episode.object_category
                )
                episode_key = (
                    str(scene_id),
                    str(episode.episode_id),
                    str(episode_instruction),
                )
                if episode_key in assigned_episode_keys:
                    assigned_episodes.append(episode)
            process_bar = tqdm.tqdm(
                range(len(assigned_episodes)), desc=f"rank {rank} scene {scene_id}"
            )
            for episode in assigned_episodes:
                if getattr(self.args, "max_episodes", None) is not None:
                    completed = len(sucs)
                    if completed >= self.args.max_episodes:
                        env.close()
                        return self.build_return_tensors(sucs, spls, oss, ones)

                episode_instruction = (
                    episode.instruction.instruction_text
                    if 'objectnav' not in self.config_path
                    else episode.object_category
                )
                if DEBUG:
                    print("episode start", episode_instruction)
                episode_id = episode.episode_id
                done_key = (str(scene_id), str(episode_id), str(episode_instruction))
                if done_key in done_res:
                    continue
                env.current_episode = episode
                observations = env.reset()
                start_metrics = env.get_metrics()
                start_distance = float(start_metrics["distance_to_goal"])
                previous_position = np.array(env.sim.get_agent_state().position, dtype=np.float32)
                path_length = 0.0
                os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
                Image.fromarray(observations['rgb']).save(
                    os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{idx}.jpg')
                )

                vis_frames = []
                step_id = 0
                if self.save_video:
                    os.makedirs(
                        os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}_{episode_id}'),
                        exist_ok=True,
                    )

                rgb_list: List[Image.Image] = []        # ALL frames seen this episode
                slice_start: int = 0                    # index in rgb_list where current slice begins
                slice_actions: List[int] = []           # actions emitted within current slice
                slice_conjunctions: List[str] = []       # conjunctions used inside current slice
                pending_action_seq: List[int] = []      # actions queued for env.step
                executed_actions: List[int] = []
                generated_action_chunks: List[dict] = []
                agent_called_stop = False
                termination_reason = None

                while not env.episode_over:
                    if (
                        getattr(self.args, "max_steps", None) is not None
                        and step_id >= self.args.max_steps
                    ):
                        print(
                            f"[rank {rank}] hit --max_steps={self.args.max_steps}, "
                            f"stop episode {scene_id}_{episode_id}",
                            flush=True,
                        )
                        termination_reason = "max_steps"
                        break

                    self.model.eval()

                    rgb = observations["rgb"]
                    image = Image.fromarray(rgb).convert('RGB')
                    rgb_list.append(image)

                    if self.save_video:
                        info = env.get_metrics()
                        if info.get('top_down_map') is not None:
                            frame = observations_to_image({'rgb': observations['rgb']}, info)
                            vis_frames.append(frame)

                    # If we've drained the previous chunk, query the model again
                    # for the next ``num_future_steps`` actions.
                    if len(pending_action_seq) == 0:
                        # If this new frame is a follow-up user turn (not the
                        # slice opener), pre-pick its conjunction so the same
                        # string is used both in the prompt now and in the
                        # context next time.
                        in_slice_pos = len(rgb_list) - 1 - slice_start
                        if in_slice_pos > 0 and in_slice_pos % self.num_future_steps == 0:
                            # The R2R32 training data cycles through these in a
                            # fixed order within every 32-step slice.
                            conjunction_idx = len(slice_conjunctions) % len(self.conjunctions)
                            slice_conjunctions.append(self.conjunctions[conjunction_idx])

                        history_imgs = rgb_list[:slice_start]
                        slice_rgbs = rgb_list[slice_start:]

                        conversation = self.build_conversation(
                            instruction=episode.instruction.instruction_text,
                            history_imgs=history_imgs,
                            slice_rgbs=slice_rgbs,
                            slice_actions=slice_actions,
                            slice_conjunctions=slice_conjunctions,
                        )

                        # Sanity: if there ARE completed turns, the builder
                        # already appended the next user turn for us. If there
                        # are NO completed turns, the opening user turn is the
                        # pending one. Either way the last message must be
                        # ``user``.
                        assert conversation[-1]["role"] == "user", conversation[-1]["role"]

                        t1 = time.time()
                        model_inputs = self.processor.apply_chat_template(
                            conversation,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                        )
                        model_inputs = strip_qwen35_thinking_prompt(
                            model_inputs, self.processor, self.model_type
                        )
                        input_len = model_inputs["input_ids"].shape[1]
                        model_inputs = model_inputs.to(self.device)
                        if getattr(self.args, "free_generation", False):
                            with torch.inference_mode():
                                output_ids = self.model.generate(
                                    **model_inputs,
                                    max_new_tokens=getattr(self.args, "max_new_tokens", 64),
                                    do_sample=False,
                                    use_cache=True,
                                    pad_token_id=resolve_pad_token_id(self.processor),
                                )
                            # Slice off the prompt to keep only the new tokens.
                            gen_ids = output_ids[:, input_len:]
                            gen_text = self.processor.batch_decode(
                                gen_ids, skip_special_tokens=True
                            )[0].strip()
                        else:
                            gen_text = action_constrained_generate(
                                self.model,
                                self.processor,
                                model_inputs,
                                max_actions=self.num_future_steps,
                            )

                        if DEBUG:
                            print(
                                f"[ep {episode_id}][step {step_id}] raw output: {gen_text!r}",
                                flush=True,
                            )

                        action_seq = self.parse_actions(gen_text)
                        if len(action_seq) == 0:
                            action_seq = [0]
                        action_seq = action_seq[: self.num_future_steps]
                        if 0 in action_seq:
                            # STOP is terminal in the training trajectories;
                            # never execute or retain actions after it.
                            action_seq = action_seq[: action_seq.index(0) + 1]
                        elif len(action_seq) < self.num_future_steps:
                            # Only the unrestricted debug path can under-produce
                            # a non-STOP chunk. Fail closed by ending the episode.
                            action_seq.append(0)

                        slice_actions.extend(action_seq)
                        pending_action_seq = action_seq[:]
                        generated_action_chunks.append(
                            {
                                "step": step_id,
                                "text": gen_text,
                                "actions": [int(action_id) for action_id in action_seq],
                            }
                        )

                        if DEBUG:
                            print(f"infer cost {time.time() - t1:.2f}s", flush=True)

                    action = pending_action_seq.pop(0)
                    # If the model emits STOP, honor it immediately.
                    executed_actions.append(int(action))
                    if int(action) == 0:
                        agent_called_stop = True
                    observations = env.step(action)
                    current_position = np.array(env.sim.get_agent_state().position, dtype=np.float32)
                    path_length += float(np.linalg.norm(current_position - previous_position, ord=2))
                    previous_position = current_position
                    step_id += 1
                    if int(action) == 0:
                        termination_reason = "agent_stop"
                    elif env.episode_over:
                        termination_reason = "env_episode_over"

                    # Close current slice every ``num_frames`` env steps.
                    if step_id % self.num_frames == 0:
                        slice_start = len(rgb_list)  # everything seen so far becomes history
                        slice_actions = []
                        slice_conjunctions = []
                        pending_action_seq = []

                process_bar.update(1)
                metrics = env.get_metrics()
                nav_error = metrics["distance_to_goal"]
                distance_success = self.success_from_distance(nav_error)
                # Habitat VLN success requires the navigation agent to call
                # STOP within the success radius. Being within range when a
                # step/time limit ends the episode is not a success.
                success = float(agent_called_stop and distance_success > 0.0)
                spl = self.compute_spl(success, start_distance, path_length)
                distance_spl = self.compute_spl(distance_success, start_distance, path_length)
                raw_habitat_success = float(metrics["success"])
                raw_habitat_spl = float(metrics["spl"])
                if self.save_video:
                    images_to_video(
                        vis_frames,
                        os.path.join(self.output_path, f'vis_{self.epoch}'),
                        f'{scene_id}_{episode_id}',
                        fps=6,
                        quality=9,
                    )
                vis_frames.clear()
                sucs.append(success)
                spls.append(spl)
                oss.append(metrics['oracle_success'])
                ones.append(nav_error)
                print(
                    f"[rank {rank}] scene_episode {scene_id}_{episode_id} "
                    f"success: {success}, spl: {spl}, "
                    f"stop: {agent_called_stop}, termination: {termination_reason}, "
                    f"distance_success: {distance_success}, "
                    f"os: {metrics['oracle_success']}, ne: {nav_error}",
                    flush=True,
                )
                result = {
                    "eval_schema_version": EVAL_SCHEMA_VERSION,
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "rank": rank,
                    "world_size": self.env_num,
                    "success": success,
                    "spl": spl,
                    "raw_habitat_success": raw_habitat_success,
                    "raw_habitat_spl": raw_habitat_spl,
                    "os": metrics['oracle_success'],
                    "ne": nav_error,
                    "distance_success": distance_success,
                    "distance_spl": distance_spl,
                    "agent_called_stop": agent_called_stop,
                    "termination_reason": termination_reason,
                    "path_length": path_length,
                    "start_distance": start_distance,
                    "steps": step_id,
                    "executed_actions": executed_actions,
                    "generated_action_chunks": generated_action_chunks,
                    "episode_instruction": episode_instruction,
                }
                with open(result_file, 'a', encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")

        env.close()
        return self.build_return_tensors(sucs, spls, oss, ones)

    def success_from_distance(self, nav_error) -> float:
        if nav_error is None:
            return 0.0
        return float(float(nav_error) <= self.success_distance)

    def compute_spl(self, success: float, start_distance: float, path_length: float) -> float:
        if success <= 0.0:
            return 0.0
        if start_distance <= 0.0:
            return success
        return success * start_distance / max(start_distance, path_length)

    def build_return_tensors(self, sucs, spls, oss, ones):
        return (
            torch.tensor(sucs, dtype=torch.float32).to(self.device),
            torch.tensor(spls, dtype=torch.float32).to(self.device),
            torch.tensor(oss, dtype=torch.float32).to(self.device),
            torch.tensor(ones, dtype=torch.float32).to(self.device),
            torch.tensor(len(sucs), dtype=torch.long).to(self.device),
        )

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------
    def parse_actions(self, output: str) -> List[int]:
        action_patterns = '|'.join(re.escape(a) for a in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[m] for m in matches]
        return list(itertools.chain.from_iterable(actions))


def load_model_for_eval(model_path: str, rank: int, gpu: int):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", None) or ""
    chosen_attn = os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2")

    fa2_mrope_patch = maybe_patch_qwen35_flash_attention(model_type, chosen_attn)

    load_kw = dict(
        dtype=torch.bfloat16,
        attn_implementation=chosen_attn,
        device_map={"": gpu},
        trust_remote_code=True,
    )

    if model_type in ("qwen3_5", "qwen3_5_moe"):
        model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kw)
    elif model_type == "qwen3_vl_moe":
        model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kw)
    elif model_type == "qwen3_vl":
        model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kw)
        # Original Qwen3-VL-specific loading path kept for reference:
        # model = Qwen3VLForConditionalGeneration.from_pretrained(
        #     model_path,
        #     dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     device_map={"": gpu},
        # )
    else:
        raise ValueError(
            f"Unsupported model_type={model_type!r} for model_path={model_path!r}. "
            "Use a Qwen3-VL / Qwen3-VL-MoE / Qwen3.5 multimodal checkpoint."
        )

    print(
        f"Initialized eval model from {model_path} "
        f"({model.__class__.__name__}, model_type={model_type}, "
        f"attn_implementation={chosen_attn}, action_decode=cached, "
        f"qwen35_fa2_mrope_patch={fa2_mrope_patch}, "
        f"rank={rank}, gpu={gpu})"
    )
    return model


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


def build_action_token_language(tokenizer, max_actions: int):
    """Build the exact token language used by comma-separated SFT actions.

    Qwen's tokenizer merges some separator/action pairs (for example
    ``,left`` and ``,right``) into one token. Therefore the decoder must
    constrain tokenized *complete strings*, not alternate standalone action and
    comma token IDs. STOP is terminal; non-STOP chunks contain exactly
    ``max_actions`` actions, matching the R2R32 training data.
    """
    max_actions = int(max_actions)
    if max_actions <= 0:
        return {}, {}, 0

    movement_actions = ("forward", "left", "right")
    all_final_actions = (*movement_actions, "stop")
    action_sequences = []

    # A STOP may terminate any partial chunk.
    for length in range(1, max_actions):
        for prefix in itertools.product(movement_actions, repeat=length - 1):
            action_sequences.append((*prefix, "stop"))

    # A full chunk may either continue navigation or end in STOP.
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
    del stop_at_stop  # STOP is always terminal in the training action language.
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


def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="xxx/Qwen3.5-9B")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./results/val_unseen/rynnvln')
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=16384)
    parser.add_argument("--min_pixels", type=int, default=262144)
    parser.add_argument("--max_pixels", type=int, default=262144)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--aggregation_timeout_seconds", type=int, default=86400)
    parser.add_argument(
        "--free_generation",
        action="store_true",
        default=False,
        help="Use unrestricted model.generate instead of action-constrained decoding.",
    )
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--port', default='1111')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    init_eval_runtime(args)
    rank = get_rank()

    model = load_model_for_eval(args.model_path, rank=rank, gpu=args.gpu)
    # Original Qwen3-VL-specific loading path kept for reference:
    # model = Qwen3VLForConditionalGeneration.from_pretrained(
    #     args.model_path,
    #     dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map=rank,
    # )

    print("loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.model_max_length = args.model_max_length
        processor.tokenizer.padding_side = "right"
    set_processor_pixels(processor, args.min_pixels, args.max_pixels)
    print("processor loaded.")

    evaluate(model, processor, args)


def evaluate(model, processor, args):
    model.eval()
    rank = get_rank()
    world_size = get_world_size()
    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        processor=processor,
        epoch=0,
        args=args,
    )
    sucs, spls, oss, ones, ep_num = evaluator.eval_action(rank)

    if getattr(args, "platform_data_parallel", False):
        write_platform_done_marker(args.output_path, rank, world_size, int(ep_num.item()))
        result_all = aggregate_platform_results(
            args.output_path,
            world_size=world_size,
            timeout_seconds=int(args.aggregation_timeout_seconds),
        )
        print(f"[rank {rank}] local episodes: {ep_num.item()}", force=True, flush=True)
        if rank == 0:
            print(result_all, force=True, flush=True)
            with open(os.path.join(args.output_path, 'result.json'), 'a', encoding="utf-8") as f:
                f.write(json.dumps(result_all) + "\n")
        return

    if world_size > 1:
        sucs_all = gather_variable_length_tensor(sucs, world_size)
        spls_all = gather_variable_length_tensor(spls, world_size)
        oss_all = gather_variable_length_tensor(oss, world_size)
        ones_all = gather_variable_length_tensor(ones, world_size)
    else:
        sucs_all, spls_all, oss_all, ones_all = sucs, spls, oss, ones

    if len(sucs_all) > 0:
        result_all = {
            "eval_schema_version": EVAL_SCHEMA_VERSION,
            "metric_mode": "habitat_official_stop_required",
            "sucs_all": sucs_all.mean().item(),
            "spls_all": spls_all.mean().item(),
            "oss_all": oss_all.mean().item(),
            "ones_all": ones_all.mean().item(),
            "length": len(sucs_all),
        }
    else:
        result_all = {
            "eval_schema_version": EVAL_SCHEMA_VERSION,
            "metric_mode": "habitat_official_stop_required",
            "sucs_all": None,
            "spls_all": None,
            "oss_all": None,
            "ones_all": None,
            "length": 0,
        }

    print(f"[rank {rank}] local episodes: {ep_num.item()}", flush=True)
    if rank == 0:
        print(result_all, flush=True)
        with open(os.path.join(args.output_path, 'result.json'), 'a', encoding="utf-8") as f:
            f.write(json.dumps(result_all) + "\n")


def _job_marker_suffix() -> str:
    job_id = os.environ.get("JOB_ID", "manual")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(job_id))


def platform_done_marker_path(output_path: str, rank: int) -> str:
    return os.path.join(output_path, f"done_rank{int(rank):06d}_{_job_marker_suffix()}.json")


def write_platform_done_marker(output_path: str, rank: int, world_size: int, episodes: int) -> None:
    os.makedirs(output_path, exist_ok=True)
    marker = {
        "rank": int(rank),
        "world_size": int(world_size),
        "episodes": int(episodes),
        "job_id": os.environ.get("JOB_ID", ""),
        "time": time.time(),
    }
    tmp_path = platform_done_marker_path(output_path, rank) + ".tmp"
    final_path = platform_done_marker_path(output_path, rank)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(marker, f)
    os.replace(tmp_path, final_path)


def wait_for_platform_done_markers(output_path: str, world_size: int, timeout_seconds: int) -> None:
    if get_rank() != 0:
        return
    deadline = time.time() + max(int(timeout_seconds), 0)
    suffix = _job_marker_suffix()
    pattern = os.path.join(output_path, f"done_rank*_{suffix}.json")
    while True:
        done_count = len(glob.glob(pattern))
        if done_count >= int(world_size):
            return
        if timeout_seconds >= 0 and time.time() > deadline:
            raise TimeoutError(
                f"Timed out waiting for platform eval ranks: "
                f"{done_count}/{world_size} done markers in {output_path}"
            )
        print(
            f"[rank 0] waiting for platform eval ranks: {done_count}/{world_size}",
            flush=True,
        )
        time.sleep(30)


def aggregate_platform_results(output_path: str, world_size: int, timeout_seconds: int):
    if get_rank() != 0:
        return None
    wait_for_platform_done_markers(output_path, world_size, timeout_seconds)

    sucs, spls, oss, ones = [], [], [], []
    seen = set()
    for path in sorted(glob.glob(os.path.join(output_path, "result_rank*.jsonl"))):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("eval_schema_version") != EVAL_SCHEMA_VERSION:
                    continue
                key = (
                    str(row.get("scene_id")),
                    str(row.get("episode_id")),
                    str(row.get("episode_instruction")),
                )
                if key in seen:
                    continue
                seen.add(key)
                if row.get("success") is not None:
                    sucs.append(float(row["success"]))
                    spls.append(float(row["spl"]))
                    oss.append(float(row["os"]))
                    ones.append(float(row["ne"]))

    if sucs:
        return {
            "eval_schema_version": EVAL_SCHEMA_VERSION,
            "metric_mode": "habitat_official_stop_required",
            "aggregation": "platform_result_files",
            "world_size": int(world_size),
            "sucs_all": float(np.mean(sucs)),
            "spls_all": float(np.mean(spls)),
            "oss_all": float(np.mean(oss)),
            "ones_all": float(np.mean(ones)),
            "length": len(sucs),
        }
    return {
        "eval_schema_version": EVAL_SCHEMA_VERSION,
        "metric_mode": "habitat_official_stop_required",
        "aggregation": "platform_result_files",
        "world_size": int(world_size),
        "sucs_all": None,
        "spls_all": None,
        "oss_all": None,
        "ones_all": None,
        "length": 0,
    }


def gather_variable_length_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    local_len = torch.tensor([tensor.numel()], dtype=torch.long, device=tensor.device)
    lengths = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(lengths, local_len)
    lengths = [int(length.item()) for length in lengths]

    max_len = max(lengths) if lengths else 0
    if tensor.numel() < max_len:
        pad = torch.zeros(max_len - tensor.numel(), dtype=tensor.dtype, device=tensor.device)
        tensor_to_gather = torch.cat([tensor, pad], dim=0)
    else:
        tensor_to_gather = tensor

    gathered = [torch.zeros(max_len, dtype=tensor.dtype, device=tensor.device) for _ in range(world_size)]
    dist.barrier()
    dist.all_gather(gathered, tensor_to_gather)
    dist.barrier()

    return torch.cat([part[:length] for part, length in zip(gathered, lengths)], dim=0)


if __name__ == "__main__":
    eval()
