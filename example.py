# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
import argparse


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert (
        world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=1024, max_batch_size=32, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(ckpt_dir: str, tokenizer_path: str, temperature: float = 0.8, top_p: float = 0.95, local_rank: int = 0):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')

    generator = load(ckpt_dir, tokenizer_path, local_rank, world_size)
    # prompts = ["The capital of Germany is the city of", "Here is my sonnet in the style of Shakespeare about an artificial intelligence:"]
    prompts = [
"""
hypothesis: when did i say that this was said created by the culprit crisis these articles that publish are well before the culver crisis\n
reference: oh and did i say that this is a set created by the covid crisis actually these articles have published ah well before the covid crisis\n
hypothesis: ah we are about to go back to shooting on wednesday we got we got shut down for a kobid ah alert\n
reference: ah we are about to go back to shooting on wednesday we got we got shutdown for a ah covid ah alert\n
hypothesis: happen'nt and he's going to test positive forks\n
reference:""",
"""
hypothesis: do you know how goggles using all the information he collects on you\n
reference: do you know how google is using all the information it collects on you\n
hypothesis: govil discovered a cibar attack on their systems\n
reference: google discovered a cyber attack on their systems\n
hypothesis: tell her what do you think when you go yourself\n
reference:""",
"""
hypothesis: by implementing so far programs for validation verification consequents in a novel interstructure called the box\n
reference: by implementing software programs for validation verification consensus in a novel infrastructure called the blockchain\n
hypothesis: anna was very good at managing processes but there wasn't much use a level soft air\n
reference: and it was very good at managing processes but there wasn't much user level software\n
hypothesis: so first of all the desport that looks like this you can see an over view of how many things you have running at the moment inside the sofa of course\n
reference:"""]

# happening and he's gonna test positive for covid
# taylor what do you think when you google yourself
# so first of all the dashboard that looks like this you can see an overview of how many things you have running at the moment inside the software of course



    results = generator.generate(prompts, max_gen_len=256, temperature=temperature, top_p=top_p)



    for result in results:
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
