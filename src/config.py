"""Paths and constants for REALISTA."""
import os

from dotenv import load_dotenv

# Loads a local, gitignored `.env` file if present -- use it for
# machine-specific overrides instead of editing this file directly.
load_dotenv()

# OpenAI API, used for GPT-based reasoning targets and LLM judges. Set your
# key directly below, or (recommended) put OPENAI_API_KEY=... in `.env` so it
# never risks being committed.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

GPT_5_NANO = "gpt-5-nano-2025-08-07"
GPT_5_MINI = "gpt-5-mini-2025-08-07"
FEASIBILITY_CHECKER_MODEL = "gpt-4.1-mini-2025-04-14"
HALLUCINATION_EVALUATOR_MODEL = "gpt-4.1-2025-04-14"

REASONING_TARGET_MODEL_MAP = {
    "gpt_5_nano": GPT_5_NANO,
    "gpt_5_mini": GPT_5_MINI,
}

# Target open-source models (HuggingFace hub ids). Point at a local path
# instead if you already have the weights downloaded.
MODEL_REGISTRY = {
    "llama3_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2_5_7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2_5_14b": "Qwen/Qwen2.5-14B-Instruct",
}

# Residual-stream layer perturbed for each model (chosen empirically; see paper).
LAYER_NUM_REGISTRY = {
    "llama3_3b": 0,
    "llama3_8b": 0,
    "qwen2_5_7b": 0,
    "qwen2_5_14b": 3,
}

MMLU_DATASET = "cais/mmlu"

# Pre-computed dictionaries. REALISTA assumes these already exist for the
# (model, subject, question) you attack; this codebase only loads them.
#
#   REPHRASING_DICT_PATH/<subject>_rephrasings.json
#       concept -> list of candidate natural-language rephrasings (stage 1)
#   LATENT_DICT_PATH/<model_type>/<subject>/<model_type>_<subject>_layer_<N>_latent_dictionary.pkl
#       concept -> latent perturbation directions (stage 2)
#
# Both are overridable via env vars / `.env` without touching this file.
REPHRASING_DICT_PATH = os.environ.get("REPHRASING_DICT_PATH", "../data/rephrasing_prompts")
LATENT_DICT_PATH = os.environ.get("LATENT_DICT_PATH", "../data/latent_dict")

# Terminal colors for readable console logging.
RED_BACKGROUND = "\033[41m"
GREEN_BACKGROUND = "\033[42m"
YELLOW_BACKGROUND = "\033[43m"
BLUE_BACKGROUND = "\033[44m"
RESET = "\033[0m"
