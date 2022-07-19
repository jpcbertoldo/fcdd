#!/usr/bin/env python

import hashlib
import json
from pathlib import Path
import random
import time
from typing import List, Optional, Tuple
import warnings

import numpy as np
import torch
from torch import Tensor
import warnings
from functools import partial
from scipy import signal 
import os
from argparse import ArgumentParser, Namespace
import functools


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


class CudaVisibleDevices:
    
    @staticmethod
    def add_arguments(parser: ArgumentParser):
        parser.add_argument(
            "--cuda_visible_devices", 
            type=int, 
            nargs='*', 
            default=None,
        )
        
    @staticmethod
    def consume_arguments(args: Namespace):
        
        if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
            
            assert args.cuda_visible_devices is None, "cannot specify both --cuda_visible_devices (cli argument) and CUDA_VISIBLE_DEVICES (enviroment variable)"
        
        if args.cuda_visible_devices is not None:
            
            print(f"cli argument --cuda_visible_devices: {args.cuda_visible_devices}")
            
            env_var_str = ",".join(map(str, args.cuda_visible_devices))
            
            print(f"cli option '--cuda_visible_devices' specified, setting CUDA_VISIBLE_DEVICES to '{env_var_str}'")

            os.environ["CUDA_VISIBLE_DEVICES"] = env_var_str
        
        del vars(args)['cuda_visible_devices']



class LogdirBaserundir:
    """
    `logdir` depends on arguments, `base_rundir` is subfolder of `logdir` and contains the start time of script. 
    `rundir` is subfolder of `base_rundir` and is the directory where the log files will be stored for a single run.
    """
    
    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            '--logdir', type=Path,
            help='Where log data is to be stored. The runs will be in subfolders and the names will be {base_rundir_prefix}run_{starttime}{base_rundir_suffix}. Default: ../../data/results.'
        )
        parser.add_argument('--base_rundir_suffix', type=str, default='',)
        parser.add_argument('--base_rundir_prefix', type=str, default='',)
        
    @staticmethod
    def consume_arguments(
        args: Namespace, 
        start_time: int,
        subfolders: tuple = (),
    ) -> Path:
        
        logdir: Path = args.logdir
        del vars(args)['logdir']
        
        print(f"logdir: cli arg: {logdir}")
        logdir = logdir.resolve().absolute()
        
        print(f"logdir: resolved: {logdir}")
                
        # this allows you to subfolder by dataset, wandb project, etc
        for value in subfolders:
            assert value is not None, f"{value} is not set"
            assert isinstance(value, str), f"{value} is not a string"
            assert value, f"{value} is empty"
            logdir = logdir / value
            
        print(f"logdir: subfolder-ed: {logdir}")
        
        base_rundir_prefix: str = args.base_rundir_prefix
        del vars(args)['base_rundir_prefix']
        
        base_rundir_suffix: str = args.base_rundir_suffix
        del vars(args)['base_rundir_suffix']
        
        # add '_' before/after the suffix/prefix
        base_rundir_prefix += '_' if base_rundir_prefix else ''
        base_rundir_suffix = f"_{base_rundir_suffix}" if base_rundir_suffix else ''
        
        base_rundir_name = f"{base_rundir_prefix}run_{start_time}{base_rundir_suffix}"
        
        print(f"logdir: base_rundir_name: {base_rundir_name}")
        
        base_rundir = logdir / base_rundir_name
        
        print(f"logdir: base_rundir: {base_rundir}")
        
        return base_rundir


class Seeds:
    
    @staticmethod
    def add_arguments(parser: ArgumentParser):
        seeds_group = parser.add_mutually_exclusive_group(required=True)
        seeds_group.add_argument(
            '--n_seeds', type=int, 
            help='Number of (randomly generated) seeds per class. If seeds is specified this is unnecessary.'
        )
        seeds_group.add_argument(
            "--seeds", type=seed_str2int, nargs='*',
            help="If set, the model will be trained with the given seeds."
                "Otherwise it can they can be autogenerated with -n-seeds."
                "The seeds must be passed in hexadecimal format, e.g. 0x1234."
        )
        
    @staticmethod
    def consume_arguments(args: Namespace):
    
        def validate_seeds(seeds):
            for s in seeds:
                assert type(s) == int, f"seed must be an int, got {type(s)}"
                assert s >= 0, f"seed must be >= 0, got {s}"
            assert len(set(seeds)) == len(seeds), f"seeds must be unique, got {s}"
            return tuple(seeds)
        
        def generate_n_seeds(n_seeds):
            seeds = []
            for _ in range(n_seeds):
                seeds.append(create_seed())
                time.sleep(1/3)  # let the system state change
            return tuple(seeds)

        # seeds and n_seeds are in a mutually exclusive group in the parser
        # so the case below must have n_seeds 
        if args.seeds is not None:
            seeds = validate_seeds(args.seeds)
            
        else:
            seeds = generate_n_seeds(args.n_seeds)
            
        del vars(args)['n_seeds']
        del vars(args)['seeds']

        return seeds
    

class WandbOffline:
    
    @staticmethod
    def add_arguments(parser: ArgumentParser):
        parser.add_argument("--wandb_offline", action="store_true", help="If set, will not sync with the webserver.",)
    
    @staticmethod
    def consume_arguments(args: Namespace):
        wandb_offline = args.wandb_offline
                
        if wandb_offline:
            print(f"wandb_offline={wandb_offline} --> setting enviroment variable WANDB_MODE=offline")
            os.environ["WANDB_MODE"] = "offline"

    
class WandbTags:
    
    @staticmethod
    def add_arguments(parser: ArgumentParser):
        parser.add_argument("--wandb_tags", type=str, nargs='*', action='extend',)
        
    @staticmethod
    def consume_arguments(args: Namespace):

        tags = args.wandb_tags
        del vars(args)['wandb_tags']
        
        monovalue_tags, kv_tags = [], {}
        
        for tag in tags:
        
            ncolons = tag.count(":")
        
            if ncolons == 0:
                monovalue_tags.append(tag)
        
            elif ncolons == 1:
                key, value = tag.split(":")
                kv_tags[key] = value
                monovalue_tags.append(key)
        
            else:
                raise ValueError(f"Tag `{tag}` has too many colons.")
        
        return monovalue_tags, kv_tags
    
    
class CliConfigHash:
    
    @staticmethod
    def add_arguments(parser: ArgumentParser):
        parser.add_argument("--cli_config_hash", type=str, nargs=2, action="append", default=None,)
        
    @staticmethod
    def consume_arguments(args: Namespace):
        
        cli_config_hashes = args.cli_config_hash
        del vars(args)['cli_config_hash']
        
        if cli_config_hashes is None:
            return dict()
        
        def validate(name_keys_pairs: List[List[str, str]]):
            for name_keys in name_keys_pairs:
                
                assert len(name_keys) == 2, f"cli_config_hash must be a list of 2-tuples, got {name_keys}"
                
                name, keys = name_keys
                assert isinstance(name, str), f"cli_config_hash name must be a string, got {name}"
                assert isinstance(keys, str), f"cli_config_hash key must be a string, got {keys}"
                assert name, f"cli_config_hash name must not be empty, got {name}"
                assert keys, f"cli_config_hash key must not be empty, got {keys}"
                
                keys = keys.split(",")
                assert len(keys) > 0, f"cli_config_hash key must not be empty, got {keys}"
                
                for k in keys:
                    assert k, f"cli_config_hash key must not be empty, got {k}"
            
            return name_keys_pairs
        
        validate(cli_config_hashes)
        
        return dict((name, tuple(keys.split(","))) for name, keys in cli_config_hashes)


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
