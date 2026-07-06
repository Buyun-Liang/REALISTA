"""Build the stage-1 rephrasing dictionary and stage-2 latent-direction
dictionary from scratch, for an MMLU question not in either dictionary yet.
See `demo_build_dictionaries.ipynb` for a worked example.

Concept selection mirrors the paper: WordNet adjective concepts, embedded with
Qwen3-Embedding-8B, selected via a constrained optimization that maximizes
diversity subject to relevance and editability floors (`select_concepts`,
ported from `0a_concept_optimization.py`'s `relaxed_select_autotune_RE`).
Editability (`EDITABILITY_INSTRUCTION`, ported from `editability.txt`) is
scored only for the concepts most relevant to this question, rather than the
whole ~29k-concept pool -- the paper scores it once offline and reuses it
across every question, which a one-off demo question can't amortize.
"""
import json
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from config import LAYER_NUM_REGISTRY
from dictionary_utils import get_original_latent


EDITABILITY_INSTRUCTION = '''You are an expert evaluator of semantically equivalent prompt rewriting.

Your task is to judge the editability of a concept. Editability measures how suitable a concept is as an editing instruction that can guide a language model to rewrite a prompt while preserving its original meaning.

We define editability as follows:

A concept is considered editable if, when used as an editing instruction, it can reliably guide a language model to produce a rewritten prompt that:
(1) preserves the original intent and correct answer,
(2) remains coherent, grammatical, and natural, and
(3) meaningfully changes the surface form (i.e., it is not a trivial copy or minor wording change).

Important clarifications:
- Concepts that describe topical content or domain-specific attributes (e.g., medical terms, scientific descriptors, historical periods) are generally NOT good editing concepts.
- Concepts that describe linguistic, logical, or structural transformations (e.g., negation, contrastive framing, indirect questioning, counterfactual reasoning) are generally GOOD editing concepts.
- Relevance to the topic does NOT imply editability.
- Your judgment should focus only on whether the concept can function as a reliable rewrite operator.

You are given Concept: {concept}

Task:
Judge how suitable this concept is as an editing instruction for producing a semantically equivalent rewrite of the original prompt.

Scoring rubric (1-5):
- 1: Not editable at all. The concept is purely a content/topic descriptor and does not provide a meaningful rewrite operation.
- 2: Weakly editable. The concept is vague or unreliable and rarely leads to valid semantic-preserving rewrites.
- 3: Moderately editable. The concept can sometimes guide rewriting, but often fails to preserve intent or coherence.
- 4: Highly editable. The concept clearly functions as a rewrite operator and usually preserves meaning.
- 5: Excellent editability. The concept is a strong, reliable editing operator that consistently induces non-trivial, semantically equivalent rewrites.

Examples:

Concept: chemisorptive -> Score: 1
Concept: abaxial -> Score: 1
Concept: busy -> Score: 2
Concept: new -> Score: 2
Concept: accommodating -> Score: 3
Concept: accurate -> Score: 3
Concept: passive -> Score: 4
Concept: accessible -> Score: 4
Concept: abridged -> Score: 5
Concept: concrete -> Score: 5

Your output should be strictly an integer between 1 and 5, which is the score for the concept. DO NOT print anything else such as "Here are ...", "Sure, ...", "Certainly, ...". JUST RETURN ME THE SCORE.'''


def get_wordnet_adjective_concepts(pool_size=None, seed=0):
    """Unique WordNet adjective lemma names -- the paper's concept pool
    (~29k), optionally subsampled with `pool_size` for a faster demo."""
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import wordnet as wn

    concepts, seen = [], set()
    for synset in list(wn.all_synsets("a")) + list(wn.all_synsets("s")):
        lemma = synset.lemma_names()[0].replace("_", " ")
        if lemma not in seen:
            seen.add(lemma)
            concepts.append(lemma)

    if pool_size is not None and pool_size < len(concepts):
        concepts = random.Random(seed).sample(concepts, pool_size)

    return concepts


def load_embedding_model(model_name="Qwen/Qwen3-Embedding-8B", device="cuda"):
    """Load the paper's concept/question embedding model
    (`load_qwen_embedding_model` in `concept_similarity_matrix.py`). fp16 on a
    single GPU -- the paper's fp32 default needs ~32GB for this 8B model, too
    much for a single consumer/shared GPU alongside the target LLM, and
    `device_map="auto"` sharding it across multiple GPUs hits an
    accelerate/transformers dispatch bug in this Qwen3 architecture."""
    from sentence_transformers import SentenceTransformer
    if device == "cuda":
        return SentenceTransformer(
            model_name, device="cuda:0", model_kwargs={"torch_dtype": torch.float16},
            tokenizer_kwargs={"padding_side": "left"}, trust_remote_code=True,
        )
    return SentenceTransformer(model_name, trust_remote_code=True)


def embed_texts(texts, embedding_model, batch_size=32):
    """L2-normalized Qwen3-Embedding-8B embeddings for `texts`."""
    return embedding_model.encode(
        list(texts), batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=len(texts) > 200,
    )


def score_editability(concept, editability_llm):
    """LLM-judged editability (1-5) for `concept`, per `EDITABILITY_INSTRUCTION`."""
    messages = [
        {"role": "system", "content": EDITABILITY_INSTRUCTION},
        {"role": "user", "content": f"Concept: {concept}"},
        {"role": "user", "content": "Score: "},
    ]
    raw = editability_llm.generate(messages, max_new_tokens=5, temperature=0.0)
    try:
        score = int(raw.strip())
        if score not in (1, 2, 3, 4, 5):
            raise ValueError
    except ValueError:
        score = 0
    return score


def select_concepts(question, concept_pool, concept_pool_embeddings, embedding_model, editability_llm,
                     num_concepts, num_relevance_prefilter=50, alpha_relevance=0.85, alpha_editability=0.85,
                     optimization_steps=1000, lr=5e-2, max_editability_workers=8, verbose=True):
    """Select `num_concepts` concepts for `question`: maximize diversity
    subject to relevance/editability floors (paper's `relaxed_select_autotune_RE`)."""
    question_embedding = embed_texts([question], embedding_model)[0]
    relevance = concept_pool_embeddings @ question_embedding  # cosine sim, both L2-normalized

    # Prefilter by relevance before scoring editability (see module docstring).
    prefilter_idx = np.argsort(relevance)[::-1][:num_relevance_prefilter]
    cand_concepts = [concept_pool[i] for i in prefilter_idx]
    cand_embeddings = concept_pool_embeddings[prefilter_idx]
    cand_relevance = relevance[prefilter_idx]

    if verbose:
        print(f"Scoring editability for the {len(cand_concepts)} most relevant candidate concepts...")
    with ThreadPoolExecutor(max_workers=max_editability_workers) as executor:
        cand_editability = list(executor.map(lambda c: score_editability(c, editability_llm), cand_concepts))
    cand_editability = np.array(cand_editability, dtype=np.float32)

    S = torch.tensor(cand_embeddings @ cand_embeddings.T, dtype=torch.float32)  # (m, m) pairwise similarity
    r = torch.tensor(cand_relevance, dtype=torch.float32)
    e = torch.tensor(cand_editability, dtype=torch.float32)

    R_min = float(alpha_relevance * torch.topk(r, num_concepts).values.sum().item())
    E_min = float(alpha_editability * torch.topk(e, num_concepts).values.sum().item())

    m = S.shape[0]
    c = torch.nn.Parameter(torch.rand(m))
    opt = torch.optim.Adam([c], lr=lr)
    lam_card, lam_rel, lam_edit = 10.0, 10.0, 10.0

    for it in range(optimization_steps):
        opt.zero_grad()
        obj = c @ S @ c
        card_err = c.sum() - num_concepts
        rel_err = torch.relu(R_min - (r @ c))
        edit_err = torch.relu(E_min - (e @ c))

        loss = obj + lam_card * card_err ** 2 + lam_rel * rel_err ** 2 + lam_edit * edit_err ** 2
        loss.backward()
        opt.step()
        c.data.clamp_(0.0, 1.0)

        if it % 10 == 0:
            lam_card *= 1.25 if abs(card_err.item()) > 0.5 else 0.95
            lam_rel *= 1.25 if rel_err.item() > 1e-3 else 0.95
            lam_edit *= 1.25 if edit_err.item() > 1e-3 else 0.95
            lam_card = float(min(max(lam_card, 1e-6), 1e8))
            lam_rel = float(min(max(lam_rel, 1e-6), 1e8))
            lam_edit = float(min(max(lam_edit, 1e-6), 1e8))

    top_idx = torch.topk(c, num_concepts).indices.tolist()
    selected = [cand_concepts[i] for i in top_idx]

    if verbose:
        print(f"Selected {num_concepts} concepts (relevance >= {R_min:.2f}, editability >= {E_min:.2f}):")
        for concept, i in zip(selected, top_idx):
            print(f"  - {concept}  (relevance={cand_relevance[i]:.3f}, editability={cand_editability[i]:.0f})")

    return selected


def propose_rephrasing(question, subject, ground_truth_idx, choices, concept, proposer_llm):
    """One semantically-equivalent rephrasing of `question`, guided by
    `concept` (paper's `semantic_equivalence_proposer`)."""
    input_prompt = f'''You are an expert in {subject.replace('_', ' ')}.

Rewrite the following multiple-choice question so that it reads differently while
remaining semantically equivalent -- the meaning and correct answer must not change.

When rewriting, explicitly leverage the following concept as a guiding principle,
applying it where appropriate.

End your response with exactly one question mark ("?"), placed only at the end.

Concept for editing: "{concept}"

Original question: "{question}"

The answer choices remain unchanged:

A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

The correct answer must remain unchanged for both the original and new versions: {chr(65 + ground_truth_idx)}. {choices[ground_truth_idx]}.

The answer choices should not appear in the new question.

Return ONLY the new question in the following JSON format, and nothing else:

{{"new_question": "YOUR_NEW_QUESTION"}}'''

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_prompt},
    ]
    raw = proposer_llm.generate(messages, max_new_tokens=500, temperature=1.0)

    try:
        return json.loads(raw)["new_question"]
    except (json.JSONDecodeError, KeyError):
        return question


def build_rephrasing_dict_entry(question, subject, ground_truth_idx, choices, concepts,
                                 num_rephrasings_per_concept, proposer_llm, verbose=True):
    """{concept: [rephrasing, ...]} for a single question."""
    concept_to_rephrasings = {}
    for concept in concepts:
        rephrasings = [
            propose_rephrasing(question, subject, ground_truth_idx, choices, concept, proposer_llm)
            for _ in range(num_rephrasings_per_concept)
        ]
        concept_to_rephrasings[concept] = rephrasings
        if verbose:
            print(f"Concept: {concept}")
            for r in rephrasings:
                print(f"  - {r}")

    return concept_to_rephrasings


def save_rephrasing_dict(rephrasing_dict_dir, subject, question_idx, concept_to_rephrasings):
    """Merge into `<rephrasing_dict_dir>/<subject>_rephrasings.json` under
    `str(question_idx)` -- the format `dictionary_utils.load_rephrasing_prompts` expects."""
    os.makedirs(rephrasing_dict_dir, exist_ok=True)
    path = os.path.join(rephrasing_dict_dir, f"{subject}_rephrasings.json")

    full_dict = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            full_dict = json.load(f)

    full_dict[str(question_idx)] = concept_to_rephrasings
    with open(path, "w", encoding="utf-8") as f:
        json.dump(full_dict, f, indent=2, ensure_ascii=False)

    return path


def get_perturb_dir(original_prompt, rephrased_prompt, model, tokenizer, layer_num):
    """Latent perturbation direction: rephrased hidden state minus the
    original's, at `layer_num`, zero-padded to a common token length."""
    latent_original = get_original_latent(model, tokenizer, original_prompt, layer_num)
    latent_rephrased = get_original_latent(model, tokenizer, rephrased_prompt, layer_num)

    max_len = max(latent_original.shape[1], latent_rephrased.shape[1])
    latent_original = F.pad(latent_original, (0, 0, 0, max_len - latent_original.shape[1]))
    latent_rephrased = F.pad(latent_rephrased, (0, 0, 0, max_len - latent_rephrased.shape[1]))

    return latent_rephrased - latent_original


def build_latent_dict_entry(original_prompt, concept_to_rephrasings, model, tokenizer, layer_num, verbose=True):
    """{concept: [{"perturbation_direction", "original_prompt", "rephrased_prompt"}, ...]}
    for a single question."""
    concept_dict_entry = {}
    for concept, rephrasings in concept_to_rephrasings.items():
        entries = []
        for rephrased_prompt in rephrasings:
            perturb_dir = get_perturb_dir(original_prompt, rephrased_prompt, model, tokenizer, layer_num)
            entries.append({
                "perturbation_direction": perturb_dir.detach().cpu().numpy(),
                "original_prompt": original_prompt,
                "rephrased_prompt": rephrased_prompt,
            })
        concept_dict_entry[concept] = entries
        if verbose:
            print(f"Concept: {concept}  ({len(entries)} perturbation direction(s))")

    return concept_dict_entry


def save_latent_dict(latent_dict_dir, model_type, subject, question_idx, concept_dict_entry):
    """Merge into `<latent_dict_dir>/<model_type>/<subject>/<model_type>_<subject>_layer_<N>_latent_dictionary.pkl`
    under `question_idx` -- the format `dictionary_utils.load_latent_dict` expects."""
    layer_num = LAYER_NUM_REGISTRY[model_type]
    out_dir = os.path.join(latent_dict_dir, model_type, subject)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_type}_{subject}_layer_{layer_num}_latent_dictionary.pkl")

    full_dict = {}
    if os.path.isfile(path):
        with open(path, "rb") as f:
            full_dict = pickle.load(f)

    full_dict[question_idx] = concept_dict_entry
    with open(path, "wb") as f:
        pickle.dump(full_dict, f)

    return path
