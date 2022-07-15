#!/usr/bin/env python
# coding: utf-8

from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
import time

from pytorch_lightning.trainer.states import RunningStage

import model_dev01
import mvtec_dataset_dev01
import train_dev01_bis
import wandb
from callbacks_dev01 import (
    HEATMAP_NORMALIZATION_MINMAX_IN_EPOCH,
    HEATMAP_NORMALIZATION_PERCENTILES_ADAPTIVE_CDF_BASED_IN_EPOCH,
    HEATMAP_NORMALIZATION_PERCENTILES_IN_EPOCH, LOG_HISTOGRAM_MODE_LOG,
    LOG_HISTOGRAM_MODES, DataloaderPreviewCallback, LearningRateLoggerCallback,
    LogAveragePrecisionCallback, LogHistogramCallback,
    LogHistogramsSuperposedPerClassCallback, LogImageHeatmapTableCallback,
    LogPercentilesPerClassCallback, LogPerInstanceValueCallback,
    LogRocCallback, TorchTensorboardProfilerCallback)
from common_dev01 import (LogdirBaserundir, Seeds, WandbOffline, WandbTags, CudaVisibleDevices, CliConfigHash)


import sys

start_time = int(time.time())
print(f"start_time: {start_time}")

argv = sys.argv[1:]
print(f"argv: {argv}")

# ========================================================================= PARSER run()

parser_run = train_dev01_bis.parser_add_arguments_run(ArgumentParser())
parser_run.set_defaults(
    wandb_entity="mines-paristech-cmm",
    wandb_project="mvtec-debug",
    wandb_offline=False,
    wandb_checkpoint_mode=train_dev01_bis.WANDB_CHECKPOINT_MODE_LAST,
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


# >>>>>>>>>>>>>>>>>>> parse <<<<<<<<<<<<<<<<<<<<<<
args_run, argv = parser_run.parse_known_args(argv)
print('after parser_run')
print(f"args_run: {args_run}")
print(f"argv: {argv}")

CudaVisibleDevices.consume_arguments(args_run)
WandbOffline.consume_arguments(args_run)

base_rundir = LogdirBaserundir.consume_arguments(args_run, start_time, subfolder_args=['wandb_project', 'dataset',])
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


# ========================================================================= PARSER run_one()

parser_run_one = train_dev01_bis.parser_add_arguments_run_one(ArgumentParser())
parser_run_one.set_defaults(
    # training
    epochs=500,  # before each epoch was doing 10 cycles over the data 
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

args_run_one = parser_run_one.parse_args(argv)
print(f"args_run_one: parsed from cli: {args_run_one}")

train_dev01_bis.args_validate_dataset_specific_choices(args_run_one)
train_dev01_bis.args_validate_model_specific_choices(args_run_one)

# ========================================================================= CALLBACKS

callbacks = []

# ========================================================================= LAUNCH

results = train_dev01_bis.run(
    **vars(args_run),
    **vars(args_run_one),
    callbacks=callbacks,
)

print('end')

# # when computing ROC on the pixels, extract a sample of this many pixels
# # used in the ROC callbacks for train and validation
# PIXELWISE_ROC_LIMIT_POINTS = 3000  # 3k

# # when computing precision-recall curve (and avg precision) on the pixels, extract a sample of this many pixels
# # used in the ROC callbacks for train and validation
# PIXELWISE_PR_LIMIT_POINTS = 3000  # 3k

# # when logging histograms, log the histograms of this many random values (not all in the array/tensor)
# PIXELWISE_HISTOGRAMS_LIMIT_POINTS_TRAIN = 3000  # 3k
# PIXELWISE_HISTOGRAMS_LIMIT_POINTS_VALIDATION = 3000  # 3k
# PIXELWISE_HISTOGRAMS_LIMIT_POINTS_TEST = 30000  # 30k 


# def add_callbacks_log_roc(train: bool, validation: bool, test: bool):

#     if train:
#         callbacks.append(
#             LogRocCallback(
#                 stage=RunningStage.TRAINING,
#                 scores_key="score_maps",
#                 gt_key="gtmaps",
#                 log_curve=False,
#                 limit_points=PIXELWISE_ROC_LIMIT_POINTS,
#                 python_generator=create_python_random_generator(seed),
#             )
#         )
        
#     if validation:
#         callbacks.append(
#             LogRocCallback(
#                 stage=RunningStage.VALIDATING,
#                 scores_key="score_maps",
#                 gt_key="gtmaps",
#                 log_curve=False,
#                 limit_points=PIXELWISE_ROC_LIMIT_POINTS, 
#                 python_generator=create_python_random_generator(seed),
#             )
#         )
        
#     if test:
#         callbacks.append(
#             LogRocCallback(
#                 stage=RunningStage.TESTING,
#                 scores_key="score_maps",
#                 gt_key="gtmaps",
#                 log_curve=True,
#                 limit_points=None,
#                 python_generator=create_python_random_generator(seed),
#             )
#         )
        

# wandb_log_roc=(False, True, True),
# parser.add_argument(
#     "--wandb_log_roc", type=bool, nargs=3,
#     help="If set, the ROC AUC curve will be logged, respectively, for train/validation/test. On test the curve is also logged."
# )
# wandb_log_roc: Tuple[bool, bool, bool],
# assert len(wandb_log_roc) == 3, f"wandb_log_roc should have 3 bools, for train/validation/test, but got {wandb_log_roc}" 
# assert all(isinstance(obj, bool) for obj in wandb_log_roc), f"wandb_log_roc should only have bool, got {wandb_log_roc}"
# add_callbacks_log_roc(*args.wandb_log_roc)

        



# # ========================================================================= PR

# def add_callbacks_log_pr(train: bool, validation: bool, test: bool):

#     if train:
#         callbacks.append(
#             LogAveragePrecisionCallback(
#                 stage=RunningStage.TRAINING,
#                 scores_key="score_maps",
#                 gt_key="gtmaps",
#                 log_curve=False,
#                 limit_points=PIXELWISE_PR_LIMIT_POINTS, 
#                 python_generator=create_python_random_generator(seed),
#             )
#         )
#     if validation:
#         callbacks.append(
#             LogAveragePrecisionCallback(
#                 stage=RunningStage.VALIDATING,
#                 scores_key="score_maps",
#                 gt_key="gtmaps",
#                 log_curve=False,
#                 limit_points=PIXELWISE_PR_LIMIT_POINTS,  
#                 python_generator=create_python_random_generator(seed),
#             )
#         )
#     if test:
#         callbacks.append(
#             LogAveragePrecisionCallback(
#                 stage=RunningStage.TESTING,
#                 scores_key="score_maps",
#                 gt_key="gtmaps",
#                 log_curve=True,
#                 limit_points=None,
#                 python_generator=create_python_random_generator(seed),
#             )
#         )
        
            

# wandb_log_pr=(False, True, True),
# parser.add_argument(
#     "--wandb_log_pr", type=bool, nargs=3,
#     help="If set, the average precision curve will be logged, respectively, for train/validation/test. On test the PR curve is also logged."
# )
# wandb_log_pr: Tuple[bool, bool, bool],
# assert len(wandb_log_pr) == 3, f"wandb_log_pr should have 3 bools, for train/validation/test, but got {wandb_log_pr}" 
# assert all(isinstance(obj, bool) for obj in wandb_log_pr), f"wandb_log_pr should only have bool, got {wandb_log_pr}"
# add_callbacks_log_pr(*wandb_log_pr)        














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









# # ========================================================================= histograms

# def add_callbacks_log_histogram_score(train_mode: str, validation_mode: str, test_mode: str):
    
#     if train_mode is not None:
#         callbacks.extend([
#             LogHistogramCallback(stage=RunningStage.TRAINING, key="score_maps", mode=train_mode,), 
#             LogHistogramsSuperposedPerClassCallback(
#                 stage=RunningStage.TRAINING, 
#                 mode=train_mode,
#                 limit_points=PIXELWISE_HISTOGRAMS_LIMIT_POINTS_TRAIN,  # 3k
#                 # same everywhere 
#                 values_key="score_maps", 
#                 gt_key="gtmaps", 
#                 python_generator=create_python_random_generator(seed),
#             ),
#         ])

#     if validation_mode is not None:
#         callbacks.extend([
#             LogHistogramCallback(stage=RunningStage.VALIDATING, key="score_maps", mode=validation_mode,), 
#             LogHistogramsSuperposedPerClassCallback(
#                 stage=RunningStage.VALIDATING, 
#                 mode=validation_mode,
#                 limit_points=PIXELWISE_HISTOGRAMS_LIMIT_POINTS_VALIDATION,  # 3k
#                 # same everywhere 
#                 values_key="score_maps", 
#                 gt_key="gtmaps", 
#                 python_generator=create_python_random_generator(seed),
#             ),
#         ])
    
#     if test_mode is not None:
#         callbacks.extend([
#             LogHistogramCallback(stage=RunningStage.TESTING, key="score_maps", mode=test_mode,), 
#             LogHistogramsSuperposedPerClassCallback(
#                 stage=RunningStage.TESTING, 
#                 mode=test_mode,
#                 limit_points=PIXELWISE_HISTOGRAMS_LIMIT_POINTS_TEST,  # 30k
#                 # same everywhere 
#                 values_key="score_maps", 
#                 gt_key="gtmaps", 
#                 python_generator=create_python_random_generator(seed),
#             ),
#         ])


# def add_callbacks_log_histogram_loss(train_mode: str, validation_mode: str, test_mode: str):
    
#     if train_mode is not None:
#         callbacks.extend([
#             LogHistogramCallback(stage=RunningStage.TRAINING, key="loss_maps", mode=train_mode,), 
#             LogHistogramsSuperposedPerClassCallback(
#                 stage=RunningStage.TRAINING, 
#                 mode=train_mode,
#                 limit_points=PIXELWISE_HISTOGRAMS_LIMIT_POINTS_TRAIN,  # 3k
#                 # same everywhere 
#                 values_key="loss_maps", 
#                 gt_key="gtmaps", 
#                 python_generator=create_python_random_generator(seed),
#             ),
#         ])

#     if validation_mode is not None:
#         callbacks.extend([
#             LogHistogramCallback(stage=RunningStage.VALIDATING, key="loss_maps", mode=validation_mode,), 
#             LogHistogramsSuperposedPerClassCallback(
#                 stage=RunningStage.VALIDATING, 
#                 mode=validation_mode,
#                 limit_points=PIXELWISE_HISTOGRAMS_LIMIT_POINTS_VALIDATION,  # 3k
#                 # same everywhere 
#                 values_key="loss_maps", 
#                 gt_key="gtmaps", 
#                 python_generator=create_python_random_generator(seed),
#             ),
#         ])
    
#     if test_mode is not None:
#         callbacks.extend([
#             LogHistogramCallback(stage=RunningStage.TESTING, key="loss_maps", mode=test_mode,), 
#             LogHistogramsSuperposedPerClassCallback(
#                 stage=RunningStage.TESTING, 
#                 mode=test_mode,
#                 limit_points=PIXELWISE_HISTOGRAMS_LIMIT_POINTS_TEST,  # 30k
#                 # same everywhere 
#                 values_key="loss_maps", 
#                 gt_key="gtmaps", 
#                 python_generator=create_python_random_generator(seed),
#             ),
#         ])
    

# wandb_log_histogram_score=(None, None, LOG_HISTOGRAM_MODE_LOG),
# wandb_log_histogram_loss=(None, None, LOG_HISTOGRAM_MODE_LOG),
# parser.add_argument(
#     "--wandb_log_histogram_score", type=none_or_str, nargs=3, choices=LOG_HISTOGRAM_MODES,
#     help="if and how to log  score values; you should give 3 values, respectively for the train/validation/test hooks"
# )
# assert len(wandb_log_histogram_score) == 3, f"wandb_log_histogram_score should have 3 bools, for train/validation/test, but got {len(wandb_log_histogram_score)} things"
# assert all(val in LOG_HISTOGRAM_MODES for val in wandb_log_histogram_score), f"wandb_log_histogram_score values should be in {LOG_HISTOGRAM_MODES}, got {wandb_log_histogram_score}"        
# wandb_log_histogram_score: Tuple[bool, bool, bool],
# add_callbacks_log_histogram_score(*wandb_log_histogram_score)

# parser.add_argument(
#     "--wandb_log_histogram_loss", type=none_or_str, nargs=3, choices=LOG_HISTOGRAM_MODES,
#     help="if and how to log loss values; you should give 3 values, respectively for the train/validation/test hooks"
# )
# assert len(wandb_log_histogram_loss) == 3, f"wandb_log_histogram_loss should have 3 bools, for train/validation/test, but got {len(wandb_log_histogram_loss)} things"
# assert all(val in LOG_HISTOGRAM_MODES for val in wandb_log_histogram_loss), f"wandb_log_histogram_loss values should be in {LOG_HISTOGRAM_MODES}, got {wandb_log_histogram_loss}"
# wandb_log_histogram_loss: Tuple[bool, bool, bool],
# add_callbacks_log_histogram_loss(*wandb_log_histogram_loss)











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