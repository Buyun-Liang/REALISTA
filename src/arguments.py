"""Hyperparameters for a single REALISTA attack run."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class RealistaArgs:
    mmlu_subject: str = "machine_learning"
    mmlu_question_idx: int = 0
    seed: int = 18
    model_type: str = "llama3_8b"       # one of config.MODEL_REGISTRY
    trial_num: int = 10                  # number of stage-2 PLD trials

    # 'none' for a direct attack on the target model, or a key from
    # config.REASONING_TARGET_MODEL_MAP to attack a reasoning model instead.
    reasoning_target: str = "none"

    # Stage-1 concurrency: candidates per forward pass when reasoning_target ==
    # 'none', or concurrent OpenAI requests when scoring against a reasoning target.
    stage1_batch_size: int = 16

    # Subsample the stage-1 rephrasing dictionary for a fast demo run. Leave
    # as None to use every concept/rephrasing, as needed to reproduce the paper.
    num_concepts: Optional[int] = None
    num_rephrasings_per_concept: Optional[int] = None

    # Stage-2 (PLD) optimization hyperparameters
    epsilon: float = 1.0        # L1 budget: 0 <= delta_i, ||delta||_1 <= epsilon
    eta: float = 1e-3           # Langevin dynamics step size
    max_iter: int = 10          # number of PLD steps
    T0: float = 1.0             # initial Langevin dynamics temperature
    annealing_rate: float = 0.9
    prompt_len: int = 50        # max number of tokens reconstructed from the latent
    noise_only: bool = False    # ablation: drop the gradient term
    gradient_only: bool = False  # ablation: drop the noise term
    verbose: bool = False
