import os
os.environ["USE_TORCH"] = "1"  # Ensure we use torch backend for datasets
import logging
import json
from pathlib import Path
import torch
import numpy as np
import pickle
import sys
from pandas import DataFrame
from tqdm import tqdm

from scripts.hooks import register_hook_encoder_header, register_hook_backbone
from utils.general import seed_everything

from local_datasets.groot_gr1 import init_groot_gr1, init_groot_gr1_with_sampling_strategy
from local_datasets.roboset import init_roboset

from gr00t.model.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP



def generate_fake_activations(random_seed, batch_size):
    """
    Backbone Eagle 2:

        batch_size x n_tokens x 2048

    DiT (Diffusion transformer  block)

        batch_size x n_tokens x 1536

    As mentioned in the meeting, n_tokens depends wheter it is an image or a text encoded

    Generally, for the image should be 576 tokens and for the text we must see the actual output :)
    -----------------------------------------------------------------------------------------------
    Argumements:
        random_seed - seed for reproducibility
    """
    np.random.seed(random_seed)
    data_ids = np.arange(batch_size)
    # Eagle 2 embeddings:
    H_train_eagle = np.random.rand(batch_size, 100,  2048)
    H_test_eagle = np.random.rand(batch_size, 100, 2048)
    # DiT embeddings: (just one layer for now)
    H_train_dit = np.random.rand(batch_size, 100, 1536)
    H_test_dit = np.random.rand(batch_size, 100, 1536)

    return H_train_eagle, H_test_eagle, H_train_dit, H_test_dit, data_ids

def get_dataset(dataset_name, tasks=None, data_key=None, subset_size=None, sampling_strategy=None, **sampling_kwargs):
    """
    Initialize the dataset based on the provided dataset name and task.
    """
    if dataset_name == "roboset":
        return init_roboset()
    elif dataset_name == "groot_gr1":
        if not sampling_strategy:
            return init_groot_gr1(tasks=tasks, data_key=data_key, subset=subset_size)
        else:
            return init_groot_gr1_with_sampling_strategy(tasks=tasks, data_key=data_key, sampling_strategy=sampling_strategy, subset=subset_size, **sampling_kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets are 'roboset' and 'groot_gr1'.")


def collect_activations(
        MODEL_PATH: Path,
        data_key: str,
        tasks: str | list[str],
        EMBODIMENT_TAG: EmbodimentTag,
        DATASET_NAME: str,
        subset_size: int,
        RESULTS_DIR: Path,
        random_seeds: list[int],
        MODEL_NAME: str = 'GR00T_N1',
        batch_size: int = 32,
        n_tokens_dit: int = 100,
        chunk_size: int = 200,
        sampling_strategy: str = None,
        sampling_kwargs: dict = {}
) -> tuple[DataFrame, DataFrame]:
    """"
    Directory structure: RESULTS_DIR / activations_{MODEL_NAME}_{DATASET_NAME} / model_layer_{layer}_seed_{random_seed}.pkl
    Sorts the dictionary according to: Layer+random_seed = filename -> ID (sample-wise) = key -> Activations = value
    Ooriginal activation dimensions:
        - Backbone
        batch_size x n_tokens x 2048
        - DiT (Diffusion transformer block)
        batch_size x n_tokens x 1536
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Check if a directories exist, if not create it
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    activations_dir = RESULTS_DIR / f"activations_{MODEL_NAME}_{DATASET_NAME}"
    activations_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset and dataloader
    dataset = get_dataset(DATASET_NAME,
                          tasks=tasks,
                          data_key=data_key,
                          subset_size=subset_size,
                          sampling_strategy=sampling_strategy,
                          **sampling_kwargs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # We don't need shuffling for collecting activations
        num_workers=4
    )

    data_config = DATA_CONFIG_MAP[data_key]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    # A wrapper for Gr00t model
    groot_policy = Gr00tPolicy(
        model_path=MODEL_PATH,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=device,
    )

    num_inference_timesteps = groot_policy.denoising_steps
    chunk_meta = {}
    layer_names = []
    collected_layer_names = False

    for random_seed in random_seeds:
        current_chunk_id = 0 # currently chunks per random seed
        current_chunk_size = 0

        # manual seed for reproducibility
        seed_everything(random_seed)

        logging.info(f"Creating embedding activations for {MODEL_NAME}, using {DATASET_NAME} with seed {random_seed}")

        # Create activations from the VLM part of the GR00T model
        vlm_backbone = groot_policy.model.backbone.eagle_model.base_model.language_model.model
        activation_backbone = register_hook_backbone(model = vlm_backbone)

        # Create activations dictionary from the Diffusion Transformer
        activation_action_head = register_hook_encoder_header(model = groot_policy.model.action_head.model)
        backbone_chunk = {}
        action_head_chunk = {}
        chunk_meta[random_seed] = { f"chunk_{current_chunk_id:03d}_ids": [] }

        action_file = open(activations_dir / f"actions_seed_{random_seed}.jsonl", "w", encoding="utf-8")

        for i, data_batch in tqdm(enumerate(dataloader)):
            if (data_ids := data_batch.pop("data_ids", None)) is None:
                data_ids = np.arange(batch_size) + i * batch_size  # Assuming data_ids are sequential for simplicity
                gr00t_inputs = data_batch
            else:
                gr00t_inputs = data_batch["gr00t_data"]

            if sampling_strategy and data_ids is None:
                raise ValueError(f"Data IDs not found in the batch, but sampling strategy {sampling_strategy} is used. This should not happen.")

            assert data_ids.shape == (batch_size,), f"Data IDs have incorrect shape: {data_ids.shape}"

            with torch.no_grad():
                action = groot_policy.get_action(gr00t_inputs)

            # Store the activations
            for idx, data_id in enumerate(data_ids):
                action_dict = { k: v[idx].tolist() for k, v in action.items() }
                action_file.write(json.dumps({"data_id": data_id.item(), "actions": action_dict}) + "\n")

                if current_chunk_size == chunk_size:
                    # store if chunk is full
                    for layer in backbone_chunk.keys(): #pylint: disable=consider-iterating-dictionary,consider-using-dict-items
                        with open(activations_dir/f'backbone_layer_{layer}_seed_{random_seed}_{current_chunk_id:03d}.pkl', 'wb') as f:
                            pickle.dump(backbone_chunk[layer], f)

                    for layer in action_head_chunk.keys(): #pylint: disable=consider-iterating-dictionary,consider-using-dict-items
                        with open(activations_dir/f'action_head_layer_{layer}_seed_{random_seed}_{current_chunk_id:03d}.pkl', 'wb') as f:
                            pickle.dump(action_head_chunk[layer], f)

                    if not collected_layer_names:
                        collected_layer_names = True
                        layer_names.extend(backbone_chunk.keys())
                        layer_names.extend(action_head_chunk.keys())

                    current_chunk_size = 0
                    current_chunk_id += 1
                    backbone_chunk.clear()
                    action_head_chunk.clear()
                    chunk_meta[random_seed][f"chunk_{current_chunk_id:03d}_ids"] = []

                chunk_meta[random_seed][f"chunk_{current_chunk_id:03d}_ids"].append(data_id.item())

                for layer in activation_backbone.keys(): #pylint: disable=consider-iterating-dictionary,consider-using-dict-items
                    # Convert the tensor to numpy and get the layer activations
                    layer_activations = torch.cat(activation_backbone[layer]).to(dtype=torch.float, device="cpu").numpy()
                    assert (layer_activations.shape[0], layer_activations.shape[-1]) == (batch_size, 2048), \
                        f"Seed {random_seed}, Layer {layer}, backbone activations have incorrect shape"

                    # Create a new dictionary for this layer
                    if backbone_chunk.get(layer, None) is None:
                        backbone_chunk[layer] = {}

                    # For each sample in the batch, create an entry with data_id as key
                    # Store the corresponding token embeddings (n_tokens x 2024) for this sample
                    backbone_chunk[layer][data_id.item()] = layer_activations[idx]

                for layer in activation_action_head.keys(): #pylint: disable=consider-iterating-dictionary,consider-using-dict-items
                    # Convert the tensor to numpy and get the layer activations
                    head_activations = torch.stack(activation_action_head[layer], dim=1).to(dtype=torch.float, device="cpu").numpy()
                    assert head_activations.shape == (batch_size, num_inference_timesteps, n_tokens_dit, 1536), \
                        f"Layer {layer} action head activations have incorrect shape"

                    # Create a new dictionary for this layer
                    if action_head_chunk.get(layer, None) is None:
                        action_head_chunk[layer] = {}

                    # For each sample in the batch, create an entry with data_id as key
                    action_head_chunk[layer][data_id.item()] = head_activations[idx]
                    if len(activation_action_head[layer]) == num_inference_timesteps:
                        activation_action_head[layer] = []

                current_chunk_size += 1

        # store leftover activations
        for layer in backbone_chunk.keys(): #pylint: disable=consider-iterating-dictionary,consider-using-dict-items
            with open(activations_dir/f'backbone_layer_{layer}_seed_{random_seed}_{current_chunk_id:03d}.pkl', 'wb') as f:
                pickle.dump(backbone_chunk[layer], f)
        backbone_chunk.clear()

        for layer in action_head_chunk.keys(): #pylint: disable=consider-iterating-dictionary,consider-using-dict-items
            with open(activations_dir/f'action_head_layer_{layer}_seed_{random_seed}_{current_chunk_id:03d}.pkl', 'wb') as f:
                pickle.dump(action_head_chunk[layer], f)
        action_head_chunk.clear()

        # Clear stored activations before next run
        activation_backbone.clear()
        activation_action_head.clear()

    # Think about using xarray instead of pandas... ?? :)
    with open(activations_dir / "chunk_meta.json", "w", encoding="utf-8") as f:
        json.dump(chunk_meta, f, indent=4)

    with open(activations_dir / "data_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "num_inference_timesteps": num_inference_timesteps,
            "chunk_size": chunk_size,
            "batch_size": batch_size,
            "tasks": tasks,
            "data_key": data_key,
            "subset_size": subset_size,
            "model": MODEL_NAME,
            "model_path": MODEL_PATH,
            "random_seeds": random_seeds,
            "layer_names": layer_names,
            "sampling_strategy": sampling_strategy,
            "sampling_kwargs": sampling_kwargs
        }, f, indent=4)

    action_file.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chosen_tasks = [
        "gr1_arms_waist.CupToDrawer"
    ]
    # Example usage
    collect_activations(
        MODEL_PATH="nvidia/GR00T-N1.5-3B",
        data_key="fourier_gr1_arms_waist", # change depending on the dataset, check DATA_CONFIG_MAP for possible values
        tasks=chosen_tasks,
        subset_size=100,
        EMBODIMENT_TAG=EmbodimentTag.GR1, # change once dataset which was actually used is set up correctly
        DATASET_NAME="groot_gr1",
        RESULTS_DIR=Path("results"),
        random_seeds=[42, 43],
        MODEL_NAME='GR00T_N1.5',
        batch_size=1,
        n_tokens_dit=49,
        chunk_size=10,
        sampling_strategy="psnr",
        sampling_kwargs={"target_psnr": 5.0}
        # sampling_strategy="even",
        # sampling_kwargs={"offset": 10}  # Example: take every 10th sample
    )
