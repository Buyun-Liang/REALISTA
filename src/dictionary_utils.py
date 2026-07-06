"""Loading utilities for the pre-computed REALISTA concept dictionaries
(rephrasing dictionary for stage 1, latent direction dictionary for stage 2).
This module only loads and reshapes them, it does not build them."""
import json
import os
import pickle

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from config import MMLU_DATASET, LATENT_DICT_PATH, REPHRASING_DICT_PATH, LAYER_NUM_REGISTRY


def load_rephrasing_prompts(subject):
    """Load the provided stage-1 rephrasing dictionary for one MMLU subject."""
    path = os.path.join(REPHRASING_DICT_PATH, f"{subject}_rephrasings.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No rephrasing dictionary found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_latent_dict(model_type, mmlu_subject, verbose=False):
    """Load the provided stage-2 latent perturbation direction dictionary."""
    path = (
        f"{LATENT_DICT_PATH}/{model_type}/{mmlu_subject}/"
        f"{model_type}_{mmlu_subject}_layer_{LAYER_NUM_REGISTRY[model_type]}"
        f"_latent_dictionary.pkl"
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No latent dictionary found at: {path}")

    if verbose:
        print(f"Loading latent direction dictionary from {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def get_padded_perturb_dict(args, latent_dict, model):
    """Zero-pad every concept's perturbation direction to the longest one
    among the concepts available for this question."""
    cur_latent_dict = latent_dict[args.mmlu_question_idx]

    max_len = 0
    for concept in cur_latent_dict:
        for trial in cur_latent_dict[concept]:
            max_len = max(max_len, trial["perturbation_direction"].shape[1])

    perturb_dir_dict = {}
    for concept in tqdm(cur_latent_dict, desc="Padding perturbation directions"):
        for trial_idx, trial in enumerate(cur_latent_dict[concept]):
            direction = torch.tensor(trial["perturbation_direction"]).to(model.device)
            padded = F.pad(direction, (0, 0, 0, max_len - direction.shape[1]), mode="constant", value=0)
            perturb_dir_dict[f"{concept}_{trial_idx}"] = padded

    return perturb_dir_dict, max_len


def get_original_latent(model, tokenizer, prompt, layer_num):
    """Hidden-state activations of `prompt` at `layer_num`, excluding the leading BOS token."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[layer_num][0][1:].unsqueeze(0)


def get_Z0(cur_task_dict, args, model, tokenizer):
    """Base latent Z0: the un-padded latent representation of the original question."""
    layer_num = LAYER_NUM_REGISTRY[args.model_type]
    latent = get_original_latent(model, tokenizer, cur_task_dict["question"], layer_num)
    return latent.clone().detach().to(model.device)


def get_dictionary_and_base_latent(args, model, tokenizer, latent_dict):
    """Build the dynamic concept dictionary D(Z0) and the padded base latent Z0."""
    device = model.device
    mmlu_dataset = load_dataset(MMLU_DATASET, args.mmlu_subject)
    cur_task_dict = mmlu_dataset["test"][args.mmlu_question_idx]

    perturb_dir_dict, max_seq_len = get_padded_perturb_dict(args, latent_dict, model)
    concepts = list(perturb_dir_dict.keys())

    layer_num = LAYER_NUM_REGISTRY[args.model_type]
    Z0_unpadded = get_original_latent(model, tokenizer, cur_task_dict["question"], layer_num)
    Z0 = F.pad(Z0_unpadded, (0, 0, 0, max_seq_len - Z0_unpadded.shape[1]), mode="constant", value=0)
    Z0 = Z0.clone().detach().to(device)

    D_Z0 = torch.stack([perturb_dir_dict[c].to(device) for c in concepts], dim=0)  # (n, 1, L, H)
    D_Z0 = D_Z0.squeeze(1)  # (n, L, H): n concepts, L tokens, H hidden size
    if args.verbose:
        print(f"D_Z0 shape: {D_Z0.shape}")

    return D_Z0, Z0, cur_task_dict, concepts
