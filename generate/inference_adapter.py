# This script is used to generate predictions using the fine-tuned adapter models
# This script is modified from the original script provided by the LIT team: https://github.com/Lightning-AI/lit-gpt

## Usage:
# python generate/inference_adapter.py --model-type "stablelm" --input-file "..data/entity_extraction/entity-extraction-test-data.json"
# python generate/inference_adapter.py --model-type "llama2" --input-file "..data/entity_extraction/entity-extraction-test-data.json"

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Literal, Optional

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy

import os
## Add the lit_gpt folder to the path
sys.path.insert(0, os.path.abspath('../'))

from generate.base import generate
from lit_gpt import Tokenizer
from lit_gpt.adapter_v2 import GPT, Block, Config
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, gptq_quantization, lazy_load
from scripts.prepare_entity_extraction_data import generate_prompt

def generate_prediction(model_type, sample):
    """
    This function is used to generate predictions using the fine-tuned adapter models. It loads the model
    and generates and prints a sample prediction. Further, it generates predictions for all the samples
    in the test data and stores the predictions in a file.

    Args:
        model_type (str): The type of model to use for prediction
        sample (dict): The sample for which the prediction is to be generated

    Returns:
        None
    """

    # Check which model to use for prediction
    if model_type == "stablelm":
        print("[INFO] Using StableLM-3B Adapter Fine-tuned")
        adapter_path: Path = Path("../out/adapter_v2/Stable-LM/entity_extraction/lit_model_adapter_finetuned.pth")
        checkpoint_dir: Path = Path("../checkpoints/stabilityai/stablelm-base-alpha-3b")
        predictions_file_name = '../data/predictions-stablelm-adapter.json'

    if model_type == "llama2":
        print("[INFO] Using LLaMa-2-7B  Adapter Fine-tuned")
        adapter_path: Path = Path("../out/adapter_v2/Llama-2/entity_extraction/lit_model_adapter_finetuned.pth")
        checkpoint_dir: Path = Path("../checkpoints/meta-llama/Llama-2-7b-hf")
        predictions_file_name = '../data/predictions-llama2-adapter.json'

    # Set the model parameters
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None
    max_new_tokens: int = 100
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

    # Check if the checkpoint directory is valid and load the model configuration
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
    
    # Load the model from the checkpoint
    checkpoint_path = checkpoint_dir / model_file
    
    # Load the tokenizer
    tokenizer = Tokenizer(checkpoint_dir)
    
    # Generate the prompt from the given sample and encode it
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    
    # Set the max sequence length
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    # Load the model configuration
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

    # Load the model weights and setup the adapter
    t0 = time.perf_counter()
    checkpoint = lazy_load(checkpoint_path)
    adapter_checkpoint = lazy_load(adapter_path)
    checkpoint.update(adapter_checkpoint.get("model", adapter_checkpoint))
    model.load_state_dict(checkpoint)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    model = fabric.setup(model)

    # Set the seed and generate the prediction
    L.seed_everything(1234)
    t0 = time.perf_counter()
    y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
    t = time.perf_counter() - t0

    # Process the predicted completion
    output = tokenizer.decode(y)
    output = output.split("### Response:")[1].strip()
    fabric.print(output)
    tokens_generated = y.size(0) - prompt_length
    fabric.print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)
    
    # Generate predictions for all the samples in the test data
    test_data_with_prediction = []
    for sample in test_data:
        # Generate prompt from sample
        prompt = generate_prompt(sample)
        fabric.print(prompt)
        
        # Encode the prompt
        encoded = tokenizer.encode(prompt, device=fabric.device)
        
        # Generate the prediction from the LLM
        y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
        output = tokenizer.decode(y)
        
        # Process the predicted completion
        output = output.split("### Response:")[1].strip()
        
        # Store prediction along with input and ground truth
        sample['prediction'] = output
        test_data_with_prediction.append(sample)
        
        fabric.print(output)
        fabric.print("---------------------------------------------------------\n\n")
    
    # Write the predictions data to a file
    with open(predictions_file_name, 'w') as file:
        json.dump(test_data_with_prediction, file, indent=4)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    # Parse the arguments
    parser = argparse.ArgumentParser(description="Entity Extraction Script")
    parser.add_argument('--input-file', type=str, default='..data/entity_extraction/entity-extraction-test-data.json', help="Path to the test JSON file")
    parser.add_argument('--model-type', type=str, choices=['stablelm', 'llama2'], default='stablelm', help="Type of model to use (stablelm or llama2)")
    args = parser.parse_args()

    # Load the test data
    with open(args.input_file, 'r') as file:
        test_data = json.load(file)

    # Single Sample
    sample = {
        "input": "Natalie Cooper,\nncooper@example.com\n6789 Birch Street, Denver, CO 80203,\n303-555-6543, United States\n\nRelationship to XYZ Pharma Inc.: Patient\nReason for contacting: Adverse Event\n\nMessage: Hi, after starting Abilify for bipolar I disorder, I've noticed that I am experiencing nausea and vomiting. Are these typical reactions? Best, Natalie Cooper",
        "output": "{\"drug_name\": \"Abilify\", \"adverse_events\": [\"nausea\", \"vomiting\"]}"
    }

    generate_prediction(model_type=args.model_type, sample=sample)