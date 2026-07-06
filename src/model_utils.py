"""Model loading utilities: the target open-source LLM, and a thin OpenAI
wrapper used for GPT-based reasoning targets and LLM judges."""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai

from config import OPENAI_API_KEY, MODEL_REGISTRY


def load_model_and_tokenizer(model_type: str):
    """Load a target open-source LLM and its tokenizer, frozen for inference."""
    try:
        model_path = MODEL_REGISTRY[model_type]
    except KeyError:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(MODEL_REGISTRY)}")

    print(f"Loading target LLM: {model_type} ({model_path})")

    tokenizer = AutoTokenizer.from_pretrained(model_path, low_cpu_mem_usage=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto"
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, tokenizer


class GPT:
    """Thin wrapper around the OpenAI API, used for reasoning targets and LLM
    judges (feasibility checking, hallucination scoring).

    Ref: https://github.com/patrickrchao/JailbreakingLLMs
    """

    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    def __init__(self, model_name: str, verbose: bool = False):
        self.model_name = model_name
        if verbose:
            print(f"Using OpenAI model: {self.model_name}")

    def generate(self, messages, max_new_tokens: int, temperature: float,
                 top_p: float = 1.0, reasoning_effort: str = "minimal"):
        """reasoning_effort is only used by the gpt-5 family."""
        output = self.API_ERROR_OUTPUT
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        for _ in range(self.API_MAX_RETRY):
            try:
                if self.model_name.startswith("gpt-5"):
                    result = client.responses.create(
                        model=self.model_name,
                        input=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_new_tokens,
                        reasoning={"effort": reasoning_effort},
                    )
                    output = result.output_text
                    break
                elif self.model_name.startswith("gpt-4"):
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_completion_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        timeout=self.API_TIMEOUT,
                        seed=42,
                    )
                    output = response.choices[0].message.content
                    break
                else:
                    raise ValueError(f"Unknown model name: {self.model_name}")

            except openai.APIError as e:
                print(f"OpenAI API returned an API Error: {e}")
                time.sleep(self.API_RETRY_SLEEP)
            except openai.APIConnectionError as e:
                print(f"Failed to connect to OpenAI API: {e}")
                time.sleep(self.API_RETRY_SLEEP)
            except openai.RateLimitError as e:
                print(f"OpenAI API request exceeded rate limit: {e}")
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)

        return output
