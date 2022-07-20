#!/usr/bin/env python
# coding: utf-8

from argparse import ArgumentParser, Namespace
from functools import partialmethod
import os
from pathlib import Path
import time

from pytorch_lightning.trainer.states import RunningStage

import model_dev01
import mvtec_dataset_dev01
import train_dev01_bis
import wandb
from callbacks_dev01_bis import (
    HEATMAP_NORMALIZATION_MINMAX_IN_EPOCH,
    HEATMAP_NORMALIZATION_PERCENTILES_ADAPTIVE_CDF_BASED_IN_EPOCH,
    HEATMAP_NORMALIZATION_PERCENTILES_IN_EPOCH, LOG_HISTOGRAM_MODE_LOG,
    LOG_HISTOGRAM_MODES, DataloaderPreviewCallback, LearningRateLoggerCallback,
    LogPrcurveCallback, LogHistogramCallback,
    LogHistogramsSuperposedCallback, LogImageHeatmapTableCallback,
    LogPercentilesPerClassCallback, LogPerInstanceValueCallback,
    LogRocCallback,)
from common_dev01 import (LogdirBaserundir, Seeds, WandbOffline, WandbTags, CudaVisibleDevices, CliConfigHash, create_python_random_generator)


import sys

start_time = int(time.time())
print(f"start_time: {start_time}")

# ========================================================================= BUILD PARSER run()

parser_run = train_dev01_bis.parser_add_arguments_run(ArgumentParser())
parser_run.set_defaults(
    wandb_entity="mines-paristech-cmm",
    wandb_project="mvtec-debug",
    wandb_offline=False,
    classes=None,
)

CudaVisibleDevices.add_arguments(parser_run)
WandbOffline.add_arguments(parser_run)

LogdirBaserundir.add_arguments(parser_run)
parser_run.set_defaults(logdir=Path("../../data/results"))

WandbTags.add_arguments(parser_run)
parser_run.set_defaults(wandb_tags=["dev01_bis",])

Seeds.add_arguments(parser_run)
CliConfigHash.add_arguments(parser_run)

# ========================================================================= BUILD PARSER run_one()

parser_run_one = train_dev01_bis.parser_add_arguments_run_one(ArgumentParser())
parser_run_one.set_defaults(
    # training
    # epochs=500,  # before each epoch was doing 10 cycles over the data 
    epochs=3,
    learning_rate=1e-3,
    weight_decay=1e-4, 
    test=True,
    # model 
    model=model_dev01.MODEL_FCDD_CNN224_VGG_F,
    loss=model_dev01.LOSS_PIXELWISE_BATCH_AVG,
    optimizer=model_dev01.OPTIMIZER_SGD,
    scheduler=model_dev01.SCHEDULER_LAMBDA,
    scheduler_parameters=[0.985],
    # dataset
    dataset=mvtec_dataset_dev01.DATASET_NAME,
    raw_shape=(240, 240),
    net_shape=(224, 224),
    batch_size=64,  # it was 128, i'm accumulating batches to simulate the same size
    nworkers=1,
    pin_memory=False,
    preprocessing=mvtec_dataset_dev01.PREPROCESSING_LCNAUG1,
    supervise_mode=mvtec_dataset_dev01.SUPERVISE_MODE_REAL_ANOMALY,
    real_anomaly_limit=1,
    datadir=Path("../../data/datasets"),
    # pytorch lightning 
    lightning_accelerator=train_dev01_bis.LIGHTNING_ACCELERATOR_GPU,
    lightning_ndevices=1,
    lightning_strategy=None,
    lightning_precision=train_dev01_bis.LIGHTNING_PRECISION_32,
    lightning_model_summary_max_depth=4,
    lightning_check_val_every_n_epoch=10,
    lightning_accumulate_grad_batches=2,
    lightning_profiler=train_dev01_bis.LIGHTNING_PROFILER_SIMPLE,
    lightning_gradient_clip_val=0,
    lightning_gradient_clip_algorithm=train_dev01_bis.LIGHTNING_GRADIENT_CLIP_ALGORITHM_NORM,
    lightning_deterministic=False,
)

# ========================================================================= BUILD PARSERS of callbacks[]

callbacks_class_parser_pairs = []

# ROC 
parser = LogRocCallback.add_arguments(ArgumentParser(), stage="validate")
parser.set_defaults(
    scores_key="score_maps",
    gt_key="gtmaps",
    log_curve=False,
    limit_points=9000,  # 9k
)
callbacks_class_parser_pairs.append((LogRocCallback, parser))

parser = LogRocCallback.add_arguments(ArgumentParser(), stage="test")
parser.set_defaults(
    scores_key="score_maps",
    gt_key="gtmaps",
    log_curve=True,
    limit_points=None,  # 9k
)
callbacks_class_parser_pairs.append((LogRocCallback, parser))

# PRCURVE
parser = LogPrcurveCallback.add_arguments(ArgumentParser(), stage="validate")
parser.set_defaults(
    scores_key="score_maps",
    gt_key="gtmaps",
    log_curve=False,
    limit_points=9000,  # 9k
)
callbacks_class_parser_pairs.append((LogPrcurveCallback, parser))

parser = LogPrcurveCallback.add_arguments(ArgumentParser(), stage="test")
parser.set_defaults(
    scores_key="score_maps",
    gt_key="gtmaps",
    log_curve=True,
    limit_points=None,  # 9k
)
callbacks_class_parser_pairs.append((LogPrcurveCallback, parser))

# HISTOGRAM - score
parser = LogHistogramCallback.add_arguments(ArgumentParser(), histogram_of="score", stage="test")
parser.set_defaults(
    key="score_maps",
    mode=LOG_HISTOGRAM_MODE_LOG,
)
callbacks_class_parser_pairs.append((LogHistogramCallback, parser))

# HISTOGRAM - loss
parser = LogHistogramCallback.add_arguments(ArgumentParser(), histogram_of="loss", stage="test")
parser.set_defaults(
    key="loss_maps",
    mode=LOG_HISTOGRAM_MODE_LOG,
)
callbacks_class_parser_pairs.append((LogHistogramCallback, parser))

# SUPPERPOSED HISTOGRAMS - score
parser = LogHistogramsSuperposedCallback.add_arguments(ArgumentParser(), histogram_of="score", stage="test")
parser.set_defaults(
    values_key="score_maps",
    gt_key="gtmaps",
    mode=LOG_HISTOGRAM_MODE_LOG,
)
callbacks_class_parser_pairs.append((LogHistogramsSuperposedCallback, parser))

# SUPPERPOSED HISTOGRAMS - loss
parser = LogHistogramsSuperposedCallback.add_arguments(ArgumentParser(), histogram_of="loss", stage="test")
parser.set_defaults(
    values_key="loss_maps",
    gt_key="gtmaps",
    mode=LOG_HISTOGRAM_MODE_LOG,
)
callbacks_class_parser_pairs.append((LogHistogramsSuperposedCallback, parser))


# >>>>>>>>>>>>>>>>>>> argv <<<<<<<<<<<<<<<<<<<<<<

argv = sys.argv[1:]
print(f"argv: {argv}")

# >>>>>>>>>>>>>>>>>>> parse and process args of callbacks[] <<<<<<<<<<<<<<<<<<<<<<

# start by parsing callback stuff because they last parser must do parse_args() instead of parse_known_args()
callbacks = []

for klass, parser in callbacks_class_parser_pairs:
    
    print(f"callback: {klass.__name__}")
    print(f"parser: {parser.description}")
    
    args, argv = parser.parse_known_args(argv)
    
    print(f"args: {args}")
    print(f"remaining argv: {argv}")
    
    callback = klass(**vars(args))
    
    print(f"callback: {callback}")
    
    callbacks.append(callback)


# >>>>>>>>>>>>>>>>>>> parse run() and run_one() <<<<<<<<<<<<<<<<<<<<<<

args_run, argv = parser_run.parse_known_args(argv)

print('after parser_run')
print(f"args_run: {args_run}")
print(f"argv: {argv}")

args_run_one = parser_run_one.parse_args(argv)
print(f"args_run_one: parsed from cli: {args_run_one}")

# >>>>>>>>>>>>>>>>>>> process args of run() <<<<<<<<<<<<<<<<<<<<<<

CudaVisibleDevices.consume_arguments(args_run)
WandbOffline.consume_arguments(args_run)

base_rundir = LogdirBaserundir.consume_arguments(
    args_run, 
    start_time, 
    subfolders=(args_run.wandb_project, args_run_one.dataset)
)
print(f"base_rundir: {base_rundir}")
base_rundir.mkdir(parents=True, exist_ok=True)

wandb_tags, key_value_tags = WandbTags.consume_arguments(args_run)
print(f"wandb_tags: {wandb_tags}")
print(f"key_value_tags: {key_value_tags}")

seeds = Seeds.consume_arguments(args_run)
print(f"seeds: {seeds}")

cli_config_hashes = CliConfigHash.consume_arguments(args_run)
print(f"cli_config_hashes: {cli_config_hashes}")

# this one must come before confighashes so the hashes can use these values
# these are added to wandb.init config arg but not passed to run_one()
# while the kwargs at the end ar passed
wandb_init_config_extra=dict(
    pid=os.getpid(),
    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    slurm_cluster_name=os.environ.get("SLURM_CLUSTER_NAME"),
    slurmd_nodename=os.environ.get("SLURMD_NODENAME"),
    slurm_job_partition=os.environ.get("SLURM_JOB_PARTITION"),
    slurm_submit_host=os.environ.get("SLURM_SUBMIT_HOST"),
    slurm_job_user=os.environ.get("SLURM_JOB_USER"),
    slurm_task_pid=os.environ.get("SLURM_TASK_PID"),
    slurm_job_name=os.environ.get("SLURM_JOB_NAME"),
    slurm_array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
    slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
    slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    slurm_job_gpus=os.environ.get("SLURM_JOB_GPUS"),
    **key_value_tags,
)
print(f"wandb_init_config_extra: {wandb_init_config_extra}")

# a collection of things from wandb's config that will be hashed together
# and the value is appended to the the config
confighashes_keys=dict(
    confighash_full=(
        "epochs", "learning_rate", "weight_decay", "model", "loss", "optimizer", "scheduler", "scheduler_parameters", "dataset", "raw_shape", "net_shape", "batch_size", "preprocessing", "supervise_mode", "real_anomaly_limit", "normal_class",                         
    ),
    confighash_dcs=("datset", "normal_class", "supervise_mode"),
    confighash_dcsl=("datset", "normal_class", "supervise_mode", "loss"),
    confighash_slurm=(
        # the info here should be very redundant but it's ok
        "slurm_job_id", "slurm_array_job_id", "slurm_array_task_id", "slurm_task_pid", 
        "slurm_job_user", "slurm_job_name", "slurm_submit_host", "slurm_cluster_name", 
        "slurmd_nodename", "slurm_job_partition",
    ),
    **cli_config_hashes,
)
print(f"confighashes_keys: {confighashes_keys}")

setattr(args_run, "start_time", start_time)
setattr(args_run, "base_rundir", base_rundir)
setattr(args_run, "seeds", seeds)
setattr(args_run, "wandb_tags", wandb_tags)
setattr(args_run, "wandb_init_config_extra", wandb_init_config_extra)
setattr(args_run, "confighashes_keys", confighashes_keys)

# >>>>>>>>>>>>>>>>>>> process args of run_one() <<<<<<<<<<<<<<<<<<<<<<

train_dev01_bis.args_validate_dataset_specific_choices(args_run_one)
train_dev01_bis.args_validate_model_specific_choices(args_run_one)

# ========================================================================= LAUNCH

results = train_dev01_bis.run(
    **vars(args_run),
    **vars(args_run_one),
    callbacks=callbacks,
)

print('end')



# # ========================================================================= heatmaps

# def add_callbacks_log_image_heatmap(
#     nsamples_train: int, nsamples_validation: int, nsamples_test: int, 
#     resolution_train: Optional[int], resolution_validation: Optional[int], resolution_test: Optional[int],
# ):

#     if nsamples_train > 0:
#         callbacks.extend([     
#             LogImageHeatmapTableCallback(
#                 stage=RunningStage.TRAINING,
#                 imgs_key="inputs",
#                 scores_key="score_maps",
#                 masks_key="gtmaps",
#                 labels_key="labels",
#                 nsamples_each_class=nsamples_train,
#                 resolution=resolution_train,
#                 # heatmap_normalization=HEATMAP_NORMALIZATION_PERCENTILES_IN_EPOCH,
#                 # min_max_percentiles=wandb_log_image_heatmap_contrast_percentiles,
#                 heatmap_normalization=HEATMAP_NORMALIZATION_PERCENTILES_ADAPTIVE_CDF_BASED_IN_EPOCH,
#                 python_generator=create_python_random_generator(seed),
#             ),
#         ])
    
#     if nsamples_validation > 0:
#         callbacks.extend([     
#             LogImageHeatmapTableCallback(
#                 stage=RunningStage.VALIDATING,
#                 imgs_key="inputs",
#                 scores_key="score_maps",
#                 masks_key="gtmaps",
#                 labels_key="labels",
#                 nsamples_each_class=nsamples_validation,
#                 resolution=resolution_validation,
#                 # heatmap_normalization=HEATMAP_NORMALIZATION_PERCENTILES_IN_EPOCH,
#                 # min_max_percentiles=wandb_log_image_heatmap_contrast_percentiles,
#                 heatmap_normalization=HEATMAP_NORMALIZATION_PERCENTILES_ADAPTIVE_CDF_BASED_IN_EPOCH,
#                 python_generator=create_python_random_generator(seed),
#             ),
#         ])
    
#     if nsamples_test > 0:
#         callbacks.extend([     
#             LogImageHeatmapTableCallback(
#                 stage=RunningStage.TESTING,
#                 imgs_key="inputs",
#                 scores_key="score_maps",
#                 masks_key="gtmaps",
#                 labels_key="labels",
#                 nsamples_each_class=nsamples_test,
#                 resolution=resolution_test,
#                 # heatmap_normalization=HEATMAP_NORMALIZATION_PERCENTILES_IN_EPOCH,
#                 # min_max_percentiles=wandb_log_image_heatmap_contrast_percentiles,
#                 heatmap_normalization=HEATMAP_NORMALIZATION_PERCENTILES_ADAPTIVE_CDF_BASED_IN_EPOCH,
#                 python_generator=create_python_random_generator(seed),
#             ),
#         ])

# # wandb_log_image_heatmap_contrast_percentiles=(3., 97.),  # (contrast_min, contrast_max)
# wandb_log_image_heatmap_nsamples=(0, 0, 30),
# wandb_log_image_heatmap_resolution=(None, None, None),

# group_log_image_heatmap = parser.add_argument_group("log-image-heatmap")
# # group_log_image_heatmap.add_argument(
# #     "--wandb_log_image_heatmap_contrast_percentiles", type=float, nargs=2,
# #     help="Percentile values for the contrast of the heatmap: min/max"
# # )
# group_log_image_heatmap.add_argument(
#     "--wandb_log_image_heatmap_nsamples", nargs=3, type=int,
#     help="how many of each class (normal/anomalous) per epoch?"
#          "alwyas log the same ones assuming the order of images doesnt' change"
# )
# group_log_image_heatmap.add_argument(
#     "--wandb_log_image_heatmap_resolution", nargs=3, type=none_or_int,
#     help="size of the image (width=height), and if 'None' then keep the original size"
# )

# # wandb_log_image_heatmap_contrast_percentiles: Tuple[float, float],
# # wandb_log_image_heatmap_nsamples: Tuple[int, int, int],
# # wandb_log_image_heatmap_resolution: Tuple[int, int, int],

# # add_callbacks_log_image_heatmap(
# #     *wandb_log_image_heatmap_nsamples, *wandb_log_image_heatmap_resolution,
# # )










# preview_nimages=0,

# parser.add_argument(
#     "--preview_nimages", type=int, help="Number of images to preview per class (normal/anomalous)."
# )
# preview_nimages: int,


# if preview_nimages > 0:
#     datamodule.setup("fit")
#     callbacks.append(
#         DataloaderPreviewCallback(
#             dataloader=datamodule.train_dataloader(embed_preprocessing=True), 
#             n_samples=preview_nimages, logkey_prefix="train/preview",
#         )
#     )












# # ================================ PERCENTILES ================================

# def add_callbacks_log_percentiles_score(train: Tuple[float, ...], validation: Tuple[float, ...]):

#     if len(train) > 0:
#         callbacks.append(
#             LogPercentilesPerClassCallback(
#                 stage=RunningStage.TRAINING, 
#                 values_key="score_maps",
#                 gt_key="gtmaps", 
#                 percentiles=train,
#             )
#         )
                    
#     if len(validation) > 0:
#         callbacks.append(
#             LogPercentilesPerClassCallback(
#                 stage=RunningStage.VALIDATING, 
#                 values_key="score_maps",
#                 gt_key="gtmaps", 
#                 percentiles=validation,
#             )
#         )           

# # wandb_log_percentiles_score_train=(0., 1., 2., 5., 10., 90., 95., 98., 99., 100.,),
# wandb_log_percentiles_score_train=(),
# # wandb_log_percentiles_score_validation=(0., 1., 2., 5., 10., 90., 95., 98., 99., 100.,),
# wandb_log_percentiles_score_validation=(),

# parser.add_argument(
#     "--wandb_log_percentiles_score_train", type=float, nargs="*",
#     help="If set, the score will be logged at the given percentiles for train (normal and anomalous scores separatedly)."
# )
# parser.add_argument(
#     "--wandb_log_percentiles_score_validation", type=float, nargs="*",
#     help="If set, the score will be logged at the given percentiles for train (normal and anomalous scores separatedly)."
# )
# args_.wandb_log_percentiles_score = (
#     args_.wandb_log_percentiles_score_train,
#     args_.wandb_log_percentiles_score_validation,
# ) 
# del vars(args_)['wandb_log_percentiles_score_train']
# del vars(args_)['wandb_log_percentiles_score_validation']
    
# wandb_log_percentiles_score: Tuple[Tuple[float, ...], Tuple[float, ...]],
# add_callbacks_log_percentiles_score(*wandb_log_percentiles_score)











# # ================================ PER-INSTANCE ================================

# # score
# def add_callbacks_log_perinstance_mean_score(train: bool, validation: bool, test: bool):

#     if train:
#         callbacks.append(
#             LogPerInstanceValueCallback(
#                 stage=RunningStage.TRAINING,
#                 values_key="score_maps",
#                 labels_key="labels",
#             )
#         )
                    
#     if validation:
#         callbacks.append(
#             LogPerInstanceValueCallback(
#                 stage=RunningStage.VALIDATING,
#                 values_key="score_maps",
#                 labels_key="labels",
#             )
#         )    
    
#     if test:
#         callbacks.append(
#             LogPerInstanceValueCallback(
#                 stage=RunningStage.TESTING,
#                 values_key="score_maps",
#                 labels_key="labels",
#             )
#         )
# # loss            
# def add_callbacks_log_perinstance_mean_loss(train: bool, validation: bool, test: bool):

#     if train:
#         callbacks.append(
#             LogPerInstanceValueCallback(
#                 stage=RunningStage.TRAINING,
#                 values_key="loss_maps",
#                 labels_key="labels",
#             )
#         )
                    
#     if validation:
#         callbacks.append(
#             LogPerInstanceValueCallback(
#                 stage=RunningStage.VALIDATING,
#                 values_key="loss_maps",
#                 labels_key="labels",
#             )
#         )    
    
#     if test:
#         callbacks.append(
#             LogPerInstanceValueCallback(
#                 stage=RunningStage.TESTING,
#                 values_key="loss_maps",
#                 labels_key="labels",
#             )
#         )
            

# wandb_log_perinstance_mean_score = (False, False, True),
# wandb_log_perinstance_mean_loss = (False, False, True),

# parser.add_argument(
#     "--wandb_log_perinstance_mean_score", type=bool, nargs=3,
#     help="if true then the pixel score will be averaged per instance and logged in a table"
# )
# parser.add_argument(
#     "--wandb_log_perinstance_mean_loss", type=bool, nargs=3,
#     help="if true then the loss will be averaged per instance and logged in a table; you should give 3 values, respectively for the train/validation/test hooks"
# )

# wandb_log_perinstance_mean_score: Tuple[bool, bool, bool],
# wandb_log_perinstance_mean_loss: Tuple[bool, bool, bool], 
# add_callbacks_log_perinstance_mean_score(*wandb_log_perinstance_mean_score)
# add_callbacks_log_perinstance_mean_loss(*wandb_log_perinstance_mean_loss)






# WANDB_WATCH_GRADIENTS = "gradients"
# WANDB_WATCH_ALL = "all"
# WANDB_WATCH_PARAMETERS = "parameters"
# WANDB_WATCH_CHOICES = (
#     None,
#     WANDB_WATCH_GRADIENTS,
#     WANDB_WATCH_ALL,
#     WANDB_WATCH_PARAMETERS,
# )
# print(f"WANDB_WATCH_CHOICES={WANDB_WATCH_CHOICES}")

# parser.add_argument(
#     # choices taken from wandb/sdk/wandb_watch.py => watch()
#     "--wandb_watch", type=none_or_str, choices=WANDB_WATCH_CHOICES, 
#     help="Argument for wandb_logger.watch(..., log=WANDB_WATCH).",
# )
# parser.add_argument(
#     "--wandb_watch_log_freq", type=int, default=100,
#     help="Log frequency of gradients and parameters. Argument for wandb_logger.watch(..., log_freq=WANDB_WATCH_LOG_FREQ). ",
# )
    
# wandb_watch: Optional[str],
# wandb_watch_log_freq: int,

# begin of train hook
# if wandb_watch is not None:
#     wandb_logger.watch(model, log=wandb_watch, log_freq=wandb_watch_log_freq)
        
# end of train hook
# if wandb_watch is not None:
#     wandb_logger.experiment.unwatch(model)

    # wandb_watch=None,
    # wandb_watch_log_freq=100,  # wandb's default
    
    
    
    
    # checkpoint model weights
    # parser.add_argument(        
    #     "--wandb_checkpoint_mode", type=none_or_str, choices=WANDB_CHECKPOINT_MODES,
    # )
    # wandb_checkpoint_mode=train_dev01_bis.WANDB_CHECKPOINT_MODE_LAST,