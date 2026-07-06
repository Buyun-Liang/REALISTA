"""Prompt construction and answer-probability utilities for MMLU-style
multiple-choice QA."""
import torch

# Token id of the option letters "A"/"B"/"C"/"D" as the first generated
# token. These differ slightly between the Llama and Qwen tokenizers.
CHOICE_TOKEN_IDS = {
    "llama3_8b": {"A": 362, "B": 426, "C": 356, "D": 423},
    "llama3_3b": {"A": 362, "B": 426, "C": 356, "D": 423},
    "qwen2_5_7b": {"A": 362, "B": 425, "C": 356, "D": 422},
    "qwen2_5_14b": {"A": 362, "B": 425, "C": 356, "D": 422},
}


def _choice_token_ids(model_type):
    try:
        return CHOICE_TOKEN_IDS[model_type]
    except KeyError:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_probs(args, outputs):
    """Softmax over the four answer-choice token logits. Keeps gradient."""
    token_map = _choice_token_ids(args.model_type)
    logits_last = outputs.logits[0, -1, :]
    choice_ids = torch.tensor(list(token_map.values()), device=logits_last.device)
    choice_logits = logits_last[choice_ids]
    return torch.softmax(choice_logits, dim=0)


def get_probs_batch(args, outputs):
    """Batched `get_probs`: returns a [batch, 4] tensor of choice probs."""
    token_map = _choice_token_ids(args.model_type)
    logits_last = outputs.logits[:, -1, :]  # [B, V]
    choice_ids = torch.tensor(list(token_map.values()), device=logits_last.device)
    choice_logits = logits_last[:, choice_ids]  # [B, 4]
    return torch.softmax(choice_logits, dim=-1)


def format_probs(probs):
    """Render a 4-way choice probability tensor/list as 'A: xx.xx%  B: ...'."""
    return "  ".join(f"{letter}: {p * 100:5.2f}%" for letter, p in zip(["A", "B", "C", "D"], probs))


def get_prompt(cur_task_dict, is_reasoning=False):
    """Build the prefix/suffix that surround the question latent.

    Ref: https://github.com/bhaweshiitk/ConformalLLM/blob/main/conformal_llm_scores.py
    """
    subject_name = cur_task_dict["subject"]
    choices = cur_task_dict["choices"]

    prefix = f"You are the world's best expert in answering questions related to {subject_name.replace('_', ' ')}. "
    if not is_reasoning:
        prefix += "Answer the following question and give me the reason. \n"
    else:
        prefix += "Answer the following question. \n"

    suffix = "\n"
    for idx, letter in enumerate(["A", "B", "C", "D"]):
        suffix += "    " + letter + ". " + choices[idx] + "\n"
    if not is_reasoning:
        suffix += "The correct answer is option: "

    return prefix, suffix


def get_full_input_embeds(model, tokenizer, cur_task_dict, question_embeds):
    """Concatenate prefix/suffix embeddings around the (possibly perturbed) question latent."""
    prefix, suffix = get_prompt(cur_task_dict)

    prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    suffix_ids = tokenizer(suffix, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

    prefix_embeds = model.model.embed_tokens(prefix_ids)
    suffix_embeds = model.model.embed_tokens(suffix_ids)

    full_input_embeds = torch.cat([prefix_embeds, question_embeds, suffix_embeds], dim=1).to(torch.float16)
    return full_input_embeds, prefix, suffix


def get_second_largest_choice_index(args, full_input_embeds, model, ground_truth_idx):
    """Attack target: 2nd most confident choice if ground truth is top, else the model's top (wrong) choice."""
    outputs = model(inputs_embeds=full_input_embeds)
    probs = get_probs(args, outputs)

    sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    highest_index, _ = sorted_probs[0]
    second_highest_index, _ = sorted_probs[1]

    if ground_truth_idx == highest_index:
        target_choice_index = second_highest_index
        flag_ground_truth_not_most_confident = False
    else:
        target_choice_index = highest_index
        flag_ground_truth_not_most_confident = True

    print(f"Ground Truth Index: {ground_truth_idx}  |  Target Choice Index: {target_choice_index}")
    return target_choice_index, flag_ground_truth_not_most_confident


def print_full_prompt(cur_task_dict, new_question_prompt=None):
    prefix, suffix = get_prompt(cur_task_dict)
    question = new_question_prompt if new_question_prompt is not None else cur_task_dict["question"]
    print(f"Full Input Prompt:\n{prefix + question + suffix}")


def generate_full_response(model, tokenizer, cur_task_dict, question_text, max_new_tokens=200):
    """Generate the target model's full text response to `question_text`."""
    prefix, suffix = get_prompt(cur_task_dict)
    inputs = tokenizer(prefix + question_text + suffix, return_tensors="pt").to(model.device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)

    return tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def get_ground_truth_target_choice(args, cur_task_dict, model, tokenizer, Z0):
    ground_truth_idx = cur_task_dict["answer"]
    full_input_embeds, _, _ = get_full_input_embeds(model, tokenizer, cur_task_dict, question_embeds=Z0)
    target_choice_index, _ = get_second_largest_choice_index(args, full_input_embeds, model, ground_truth_idx)
    return ground_truth_idx, target_choice_index
