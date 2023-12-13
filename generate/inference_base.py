# This script is used to generate predictions using the base models of StableLM and LLaMa-2
# This script is modified from the original script provided by the LIT team: https://github.com/Lightning-AI/lit-gpt

## Usage:
# python generate/inference_base.py --model-type "stablelm"
# python generate/inference_lora.py --model-type "llama2"

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Any, Literal, Optional
import json

import lightning as L
import torch
import torch._dynamo.config
import torch._inductor.config
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy

## Add the lit_gpt folder to the path
sys.path.insert(0, os.path.abspath('../'))

from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.model import Block
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    gptq_quantization,
    load_checkpoint,
)

def multinomial_num_samples_1(probs: torch.Tensor) -> torch.Tensor:
    """
    This function is derived from the original file provided by the LIT team:

    Args:
        probs: Tensor of shape (..., N) containing probabilities for N events.

    Returns:
        Tensor of shape (...) containing samples from the multinomial distribution.
    """
    if torch._dynamo.is_compiling():
        # Faster alternative to `torch.multinomial(probs, num_samples=1)` that is also CUDAGraph friendly
        distribution = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / distribution, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
    """
    This function is derived from the original file provided by the LIT team:

    Args:
        logits: Tensor of shape (..., N) containing logits for N events.
        temperature: Scales the logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.

    Returns:
        Tensor of shape (...) containing samples from the multinomial distribution.
    """
    logits = logits[0, -1]
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, i = torch.topk(logits, min(top_k, logits.size(-1)))
        # do not use `torch.where` as in nanogpt because it will repeat top-k collisions
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
    # optionally scale the logits and sample from a probability distribution
    if temperature > 0.0:
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
        return multinomial_num_samples_1(probs)
    return torch.argmax(logits, dim=-1, keepdim=True)

def next_token(model: GPT, input_pos: torch.Tensor, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """
    This function is derived from the original file provided by the LIT team:

    Args:
        model: The model to use.
        input_pos: Tensor of shape (1) with the position of the last token in the input.
        x: Tensor of shape (1, T) with the input sequence.
        **kwargs: Keyword arguments passed to `sample`.

    Returns:
        Tensor of shape (1)
    """
    logits = model(x, input_pos)
    next = sample(logits, **kwargs)
    return next.type_as(x)

@torch.inference_mode()
def generate(
    model: GPT,
    prompt: torch.Tensor,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        prompt: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
    """
    T = prompt.size(0)
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

    device = prompt.device
    tokens = [prompt]
    input_pos = torch.tensor([T], device=device)
    token = next_token(
        model, torch.arange(0, T, device=device), prompt.view(1, -1), temperature=temperature, top_k=top_k
    ).clone()
    tokens.append(token)
    for _ in range(2, max_returned_tokens - T + 1):
        token = next_token(model, input_pos, token.view(1, -1), temperature=temperature, top_k=top_k).clone()
        tokens.append(token)
        if token == eos_id:
            break
        input_pos = input_pos.add_(1)
    return torch.cat(tokens)

def generate_prediction(model_type, prompt):
    """
    This function is used to generate predictions using the fine-tuned adapter models. It loads the model
    and generates and prints a sample prediction. Further, it generates predictions for all the samples
    in the test data and stores the predictions in a file.

    Args:
        model_type (str): The type of model to use for prediction
        prompt (str): The prompt to use for prediction

    Returns:
        None
    """
    
    # Set the model type and the paths
    if model_type == "stablelm":
        print("[INFO] Using StableLM-3B base model")
        checkpoint_dir: Path = Path("../checkpoints/stabilityai/stablelm-base-alpha-3b")

    if model_type == "llama2":
        print("[INFO] Using Llama2-7B base model")
        checkpoint_dir: Path = Path("../checkpoints/meta-llama/Llama-2-7b-hf")
    
    # Set the default arguments
    predictions_file_name = '../data/predictions-stablelm-base.json'
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None
    max_new_tokens: int = 50
    top_k: int = 200
    temperature: float = 0.1
    strategy: str = "auto"
    devices: int = 1
    precision: Optional[str] = None

    # Set the strategy
    if strategy == "fsdp":
        strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=False)
    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
    fabric.launch()

    # Check if the checkpoint directory is valid and load the model config
    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_json(checkpoint_dir / "lit_config.json")

    # Check if the quantization is required
    if quantize is not None and devices > 1:
        raise NotImplementedError
    if quantize == "gptq.int4":
        model_file = "lit_model_gptq.4bit.pth"
        if not (checkpoint_dir / model_file).is_file():
            raise ValueError("Please run `python quantize/gptq.py` first")
    else:
        model_file = "lit_model.pth"
    
    # Load the tokenizer and encode the prompt
    checkpoint_path = checkpoint_dir / model_file
    tokenizer = Tokenizer(checkpoint_dir)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    # Load the model
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True), gptq_quantization(quantize == "gptq.int4"):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()

    model = fabric.setup_module(model)
    t0 = time.perf_counter()
    load_checkpoint(fabric, model, checkpoint_path)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    # Set the seed and generate the predictions
    L.seed_everything(1234)
    y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k)
    for block in model.transformer.h:
        block.attn.kv_cache.reset_parameters()
    output = tokenizer.decode(y)
    fabric.print(output)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    # Parse the arguments
    parser = argparse.ArgumentParser(description="Entity Extraction Script")
    parser.add_argument('--model-type', type=str, choices=['stablelm', 'llama2'], default='stablelm', help="Type of model to use (stablelm or llama2)")
    args = parser.parse_args()

    # Single Sample
    example = {
        "input": "Natalie Cooper,\nncooper@example.com\n6789 Birch Street, Denver, CO 80203,\n303-555-6543, United States\n\nRelationship to XYZ Pharma Inc.: Patient\nReason for contacting: Adverse Event\n\nMessage: Hi, after starting Abilify for bipolar I disorder, I've noticed that I am experiencing nausea and vomiting. Are these typical reactions? Best, Natalie Cooper",
        "output": "{\"drug_name\": \"Abilify\", \"adverse_events\": [\"nausea\", \"vomiting\"]}"
    }

    prompt = f"""Act as an expert Analyst with 20+ years of experience\
            in Pharma and Healthcare industry. \
            For the following provided input you need to generate the output which \
            identifies and extracts entities like 'drug_name' and 'adverse_events' \
            use the format:\n\
            {{'drug_name':'DRUG_NAME_HERE', 'adverse_events':[## List of symptoms here]}}\n\

            ### Extract Entities from the follwing:\n\
            {example["input"]}\

            ### Response:
            """

    generate_prediction(model_type=args.model_type, prompt=prompt)