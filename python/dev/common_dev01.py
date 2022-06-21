#!/usr/bin/env python

import hashlib
import json
import random

import numpy
import torch


def seed_str2int(seed: str):
    assert isinstance(seed, str), f"entropy must be a string, got {type(seed)}"
    assert seed.startswith("0x"), f"entropy must start with 0x, got {seed}"
    # [2:] cuts off the "0x"
    assert set(seed[2:]).issubset("0123456789abcdef"), f"entropy must be a hexadecimal string, got {seed}"
    return int(seed, base=16)


def seed_int2str(seed: int):
    assert isinstance(seed, int), f"entropy must be an integer, got {type(seed)}"
    return f"0x{seed:x}"


# these two functions make sure that all generators are of the same type

def create_seed() -> int:
    ss = numpy.random.SeedSequence()
    print(f"random seed generated from the system entropy using SeedSequence: {seed_int2str(ss.entropy)}")
    # torch's int is 64 bit most, so i reduce the seed to 64 bit here
    return int(ss.entropy % (2**64))

    
def create_numpy_random_generator(seed: int) -> numpy.random.Generator:
    assert isinstance(seed, int), f"seed must be an integer, got {type(seed)}"
    assert seed >= 0, f"seed must be >= 0, got {seed_int2str(seed)}"
    gen = numpy.random.Generator(numpy.random.PCG64(numpy.random.SeedSequence(seed)))
    print(f"random generator {gen} ({type(gen)}) instantiaed from the provided seed: {seed_int2str(seed)}")
    return gen    
    
    
def create_python_random_generator(seed: int) -> random.Random:
    assert isinstance(seed, int), f"seed must be an integer, got {type(seed)}"
    assert seed >= 0, f"seed must be >= 0, got {seed_int2str(seed)}"
    gen = random.Random(seed)
    print(f"random generator generator {gen} ({type(gen)}) instantiaed from the provided seed: {seed_int2str(seed)}")
    return gen


def create_torch_random_generator(seed: int) -> torch.Generator:
    assert isinstance(seed, int), f"seed must be an integer, got {type(seed)}"
    assert seed >= 0, f"seed must be >= 0, got {seed_int2str(seed)}"
    gen = torch.Generator()
    gen.manual_seed(seed)
    print(f"random generator {gen} ({type(gen)}) instantiaed from the provided seed: {seed_int2str(seed)}")
    return gen


def hashify_config(config_dict: dict, keys=None) -> str:
    """
    put the configs specified by `keys` in a string and hash it with 6 bytes 
    in order to get a redable 12-character that can be used to more easily
    group things in wandb 
    
    if `keys` not given, then all the keys in the dict are used
    """
    HEXDIGEST_N_CHARACTERS = 12
    if keys is None:
        keys = set(config_dict.keys())
    tobehashed = {k: v for k, v in config_dict.items() if k in keys}
    # sort by key
    tobehashed = sorted(tobehashed.items(), key=lambda kv: kv[0])
    tobehashed = json.dumps(tobehashed).encode("utf-8")
    return  hashlib.blake2b(tobehashed, digest_size = HEXDIGEST_N_CHARACTERS // 2).hexdigest()

