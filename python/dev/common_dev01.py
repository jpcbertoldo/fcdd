#!/usr/bin/env python

import hashlib
import json
import random
from typing import Optional, Tuple
import warnings

import numpy as np
import torch
from torch import Tensor
import warnings
from functools import partial
from scipy import signal 


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
    ss = np.random.SeedSequence()
    print(f"random seed generated from the system entropy using SeedSequence: {seed_int2str(ss.entropy)}")
    # torch's int is 64 bit most, so i reduce the seed to 64 bit here
    return int(ss.entropy % (2**64))

    
def create_numpy_random_generator(seed: int) -> np.random.Generator:
    assert isinstance(seed, int), f"seed must be an integer, got {type(seed)}"
    assert seed >= 0, f"seed must be >= 0, got {seed_int2str(seed)}"
    gen = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
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


class AdaptiveClipError(Exception):
    pass


@torch.no_grad()
def find_scores_clip_values_from_empircal_cdf(scores_normal: Tensor, scores_anomalous: Tensor, nquantiles: int = 301, cutfunc_threshold: float = 0.05) -> Optional[Tuple[float, float]]:
        
    MIN_SCORE_SAMPLE_SIZE_WARNING = 3000
    MIN_NQUANTILES = 101

    assert nquantiles >= MIN_NQUANTILES, f"nquantiles={nquantiles}, MIN_NQUANTILES={MIN_NQUANTILES}"
    assert 0 < cutfunc_threshold < 1, f"cutfunc_threshold={cutfunc_threshold}, must be between 0 and 1 (both excluded)"
        
    quantiles = torch.linspace(0, 1, nquantiles, device=scores_normal.device)

    def get_cdf(scores: Tensor) -> Tuple[Tensor, Tuple[float, float, float]]:
        """return a np-compatible CDF of the scores and (min, median, max) of the scores"""
        
        scores = scores.view(-1)
        
        if scores.numel() < MIN_SCORE_SAMPLE_SIZE_WARNING:
            # reminder: stacklevel=2 means we're in the function that called this one
            warnings.warn(f"WARNING: scores.numel()={scores.numel()} < {MIN_SCORE_SAMPLE_SIZE_WARNING}", stacklevel=2)
        
        # these are (nquantiles, 2) tensors so it's nota big deal to put them back in the cpu
        # it is necessary because we want to use np's interp() function
        points_cdf = torch.stack([torch.quantile(scores_normal, quantiles), quantiles], dim=1).cpu().detach().numpy()

        min_score = points_cdf[0, 0]
        max_score = points_cdf[-1, 0]
        
        if nquantiles % 2 == 1:
            median_score = points_cdf[(nquantiles - 1) // 2, 0]
            
        else:
            # in this case it means that there are the quantiles .5 - \epsilon and .5 + \epsilon, but not .5
            # instead of computing the real median we'll just approximate it as the average of their respective scores
            idx2 = nquantiles // 2    
            idx1 = idx2 - 1
            median_score = (points_cdf[idx1, 0] + points_cdf[idx2, 0]) / 2
        
        # [:, 0] is the scores, [:, 1] is the quantiles
        cdf_func = partial(np.interp, xp=points_cdf[:, 0], fp=points_cdf[:, 1], left=0, right=1)
        
        return cdf_func, (min_score, median_score, max_score)

    cdf_normal, (min_score_normal, median_score_normal, max_score_normal) = get_cdf(scores_normal)
    cdf_anomalous, (min_score_anomalous, median_score_anomalous, max_score_anomalous) = get_cdf(scores_anomalous)

    def cutfunc(scores, normalize=1.0):
        return np.stack([1 - cdf_normal(scores), cdf_anomalous(scores)], axis=1).min(axis=1).clip(0, .5) / normalize

    # we dont need to search the maximum in the entier range cuz we know its somewhere in the middle
    cutfunc_maxsearch_range = (min(median_score_normal, median_score_anomalous), max(median_score_normal, median_score_anomalous))
    cutfunc_max = np.max(cutfunc(np.linspace(*cutfunc_maxsearch_range, num=1000)))
    cutfunc_normalized = partial(cutfunc, normalize=cutfunc_max)

    cutfunc_range = (
        min(min_score_normal, min_score_anomalous),
        max(max_score_normal, max_score_anomalous),
    )
    score_search_space = np.linspace(*cutfunc_range, num=3000)
    solutions_args = signal.argrelmin(np.abs(cutfunc_normalized(score_search_space) - cutfunc_threshold))[0]

    if len(solutions_args) != 2:
        warnings.warn(f"WARNING: len(solutions_args)={len(solutions_args)}, expected 2, returning None (no clipping should be applied)", stacklevel=2)
        raise AdaptiveClipError(f"should have found exactly 2 solutions, but found {len(solutions_args)}")

    solutions = score_search_space[solutions_args]
    clipmin, clipmax = solutions

    if clipmin > clipmax:
        clipmin, clipmax = clipmax, clipmin

    return clipmin, clipmax
