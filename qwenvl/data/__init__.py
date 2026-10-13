import re


R2R_VA2L = {
    "annotation_path": "xxx/RSI-Nav/data/datasets/va2l/r2rva2l5x.jsonl",
    "data_path": "xxx/RSI-Nav/data/trajectory_data/R2R/",
}

RXR_VA2L = {
    "annotation_path": "xxx/RSI-Nav/data/datasets/va2l/rxr_va2l_qwen_32uniform.jsonl",
    "data_path": "xxx/RSI-Nav/data/trajectory_data/RxR/",
}

SRDF_VA2L = {
    "annotation_path": "xxx/RSI-Nav/data/datasets/va2l/srdf_va2l_qwen_32uniform.jsonl",
    "data_path": "xxx/RSI-Nav/data/trajectory_data/srdf/",
}

R2R32 = {
    "annotation_path": "xxx/RSI-Nav/data/datasets/r2r/json/r2r_qwen_32step_episodes_single.jsonl",
    "data_path": "xxx/RSI-Nav/data/trajectory_data/R2R/",
}

ENVDROP = {
    "annotation_path": "xxx/RSI-Nav/data/datasets/envdrop/envdrop_qwen_32step_episodes_single.jsonl",
    "data_path": "xxx/RSI-Nav/data/trajectory_data/EnvDrop",
}

RXR32 = {
    "annotation_path": "xxx/RSI-Nav/data/datasets/rxr/json/rxr_qwen_32step_episodes_single.jsonl",
    "data_path": "xxx/RSI-Nav/data/trajectory_data/RxR/",
}

SCALEVLN = {
    "annotation_path": "xxx/RSI-Nav/data/datasets/scalevln/scalevln_qwen_32step_episodes_single.jsonl",
    "data_path": "xxx/RSI-Nav/data/trajectory_data/ScaleVLN/",
}

SRDF = {
    "annotation_path": "xxx/RSI-Nav/data/datasets/srdf/srdf_qwen_32step_episodes_single.jsonl",
    "data_path": "xxx/RSI-Nav/data/trajectory_data/srdf/images/",
}

VSI_590K_VIDEO = {
    "annotation_path": "xxx/RSI-Nav/data/datasets/vsi_590k/vsi_590k_convert_image32.jsonl",
    "data_path": "xxx/RSI-Nav/data/datasets/vsi_590k/VSI-590K/",
}

LLaVA_3D_546K_SCANNET = {
    "annotation_path": "xxx/RSI-Nav/data/datasets/scanqa/LLaVA-3D-546K-scannet-only_image32.jsonl",
    "data_path": "xxx/RSI-Nav/data/datasets/vsi_590k/VSI-590K/",
}

MINDCUBE = {
    "annotation_path": "xxx/RSI-Nav/data/datasets/mindcube/MindCube.jsonl",
    "data_path": "xxx/RSI-Nav/data/datasets/mindcube",
}


data_dict = {
    "rxr_va2l": RXR_VA2L,
    "r2r_va2l": R2R_VA2L,
    "srdf_va2l": SRDF_VA2L,
    "r2r32": R2R32,
    "rxr32": RXR32,
    "srdf": SRDF,
    "envdrop": ENVDROP,
    "scalevln": SCALEVLN,
    "vsi_590k_video": VSI_590K_VIDEO,
    "llava_3d_546k_scannet": LLaVA_3D_546K_SCANNET,
    "mindcube": MINDCUBE,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["r2r_va2l"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
# :
#         print(config)
    print(config)
