#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import model_dev01_bis
import mvtec_dataset_dev01
import train_dev01_bis
from callbacks_dev01_bis import (
    HEATMAP_NORMALIZATION_PERCENTILES_ADAPTIVE_CDF_BASED_IN_EPOCH,
    LOG_HISTOGRAM_MODE_LOG, LogHistogramCallback,
    LogHistogramsSuperposedCallback, LogImageHeatmapTableCallback,
    LogPerInstanceMeanCallback, LogPrcurveCallback, LogRocCallback)
from common_dev01_bis import (CliConfigHash, CudaVisibleDevices,
                              LogdirBaserundir, Seeds, WandbOffline, WandbTags,)

start_time = int(time.time())
print(f"start_time: {start_time}")

parser = ArgumentParser()

# ========================================================================= BUILD PARSER run()

run_args_group = parser.add_argument_group("run")

train_dev01_bis.cli_add_arguments_run(run_args_group)
run_args_group.set_defaults(
    wandb_entity="mines-paristech-cmm",
    wandb_project="mvtec-debug",
    wandb_offline=False,
    classes=None,
)

CudaVisibleDevices.cli_add_arguments(run_args_group)
WandbOffline.cli_add_arguments(run_args_group)

LogdirBaserundir.cli_add_arguments(run_args_group)
run_args_group.set_defaults(logdir=Path("../../data/results"))

WandbTags.cli_add_arguments(run_args_group)
run_args_group.set_defaults(
    wandb_tags=[
        "dev01_bis",
    ]
)

Seeds.cli_add_arguments(run_args_group)
CliConfigHash.cli_add_arguments(run_args_group)

# ========================================================================= BUILD PARSER run_one()

runone_args_group = parser.add_argument_group("runone")

train_dev01_bis.cli_add_arguments_run_one(runone_args_group)
runone_args_group.set_defaults(
    # training
    # epochs=500,  # before each epoch was doing 10 cycles over the data
    epochs=3,
    learning_rate=1e-3,
    weight_decay=1e-4,
    test=True,
    # model
    model=model_dev01_bis.MODEL_FCDD_CNN224_VGG_F,
    loss=model_dev01_bis.LOSS_PIXELWISE_BATCH_AVG,
    optimizer=model_dev01_bis.OPTIMIZER_SGD,
    scheduler=model_dev01_bis.SCHEDULER_LAMBDA,
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

callbacks_class_and_namempa_by_argsgroup = {}

# ROC
group = parser.add_argument_group("roc_validate")
cli_arg_name_map = LogRocCallback.cli_add_arguments(
    group,
    stage="validate",
    # defaults
    scores_key="score_maps",
    gt_key="gtmaps",
    log_curve=False,
    limit_points=9000,  # 9k
)
callbacks_class_and_namempa_by_argsgroup[group.title] = (
    LogRocCallback,
    cli_arg_name_map,
)

group = parser.add_argument_group("roc_test")
cli_arg_name_map = LogRocCallback.cli_add_arguments(
    group,
    stage="test",
    # defaults
    scores_key="score_maps",
    gt_key="gtmaps",
    log_curve=True,
    limit_points=None,  # 9k
)
callbacks_class_and_namempa_by_argsgroup[group.title] = (
    LogRocCallback,
    cli_arg_name_map,
)

# PRCURVE
group = parser.add_argument_group("prcurve_validate")
cli_arg_name_map = LogPrcurveCallback.cli_add_arguments(
    group,
    stage="validate",
    # defaults
    scores_key="score_maps",
    gt_key="gtmaps",
    log_curve=False,
    limit_points=9000,  # 9k
)
callbacks_class_and_namempa_by_argsgroup[group.title] = (
    LogPrcurveCallback,
    cli_arg_name_map,
)

group = parser.add_argument_group("prcurve_test")
cli_arg_name_map = LogPrcurveCallback.cli_add_arguments(
    group,
    stage="test",
    # defaults
    scores_key="score_maps",
    gt_key="gtmaps",
    log_curve=True,
    limit_points=None,  # 9k
)
callbacks_class_and_namempa_by_argsgroup[group.title] = (
    LogPrcurveCallback,
    cli_arg_name_map,
)

# HISTOGRAM - score
group = parser.add_argument_group("histogram_score_test")
cli_arg_name_map = LogHistogramCallback.cli_add_arguments(
    group,
    histogram_of="score",
    stage="test",
    # defaults
    key="score_maps",
    mode=LOG_HISTOGRAM_MODE_LOG,
)
callbacks_class_and_namempa_by_argsgroup[group.title] = (
    LogHistogramCallback,
    cli_arg_name_map,
)

# HISTOGRAM - loss
group = parser.add_argument_group("histogram_loss_test")
cli_arg_name_map = LogHistogramCallback.cli_add_arguments(
    group,
    histogram_of="loss",
    stage="test",
    # defaults
    key="loss_maps",
    mode=LOG_HISTOGRAM_MODE_LOG,
)
callbacks_class_and_namempa_by_argsgroup[group.title] = (
    LogHistogramCallback,
    cli_arg_name_map,
)

# SUPPERPOSED HISTOGRAMS - score
group = parser.add_argument_group("supperposed_histograms_score_test")
cli_arg_name_map = LogHistogramsSuperposedCallback.cli_add_arguments(
    group,
    histogram_of="score",
    stage="test",
    # defaults
    values_key="score_maps",
    gt_key="gtmaps",
    mode=LOG_HISTOGRAM_MODE_LOG,
    limit_points=9000,  # 9k
)
callbacks_class_and_namempa_by_argsgroup[group.title] = (
    LogHistogramsSuperposedCallback,
    cli_arg_name_map,
)

# SUPPERPOSED HISTOGRAMS - loss
group = parser.add_argument_group("supperposed_histograms_loss_test")
cli_arg_name_map = LogHistogramsSuperposedCallback.cli_add_arguments(
    group,
    histogram_of="loss",
    stage="test",
    # defaults
    values_key="loss_maps",
    gt_key="gtmaps",
    mode=LOG_HISTOGRAM_MODE_LOG,
    limit_points=9000,  # 9k
)
callbacks_class_and_namempa_by_argsgroup[group.title] = (
    LogHistogramsSuperposedCallback,
    cli_arg_name_map,
)

# IMAGE/HEATMAP TABLE
group = parser.add_argument_group("imageheatmap_table_validate")
cli_arg_name_map = LogImageHeatmapTableCallback.cli_add_arguments(
    group,
    stage="validate",
    # defaults
    imgs_key="inputs",
    scores_key="score_maps",
    masks_key="gtmaps",
    labels_key="labels",
    nsamples=0,
    resolution=None,
    heatmap_normalization=HEATMAP_NORMALIZATION_PERCENTILES_ADAPTIVE_CDF_BASED_IN_EPOCH,
    min_max_percentiles=None,
)
callbacks_class_and_namempa_by_argsgroup[group.title] = (
    LogImageHeatmapTableCallback,
    cli_arg_name_map,
)

group = parser.add_argument_group("imageheatmap_table_test")
cli_arg_name_map = LogImageHeatmapTableCallback.cli_add_arguments(
    group,
    stage="test",
    # defaults
    imgs_key="inputs",
    scores_key="score_maps",
    masks_key="gtmaps",
    labels_key="labels",
    nsamples=30,
    resolution=None,
    heatmap_normalization=HEATMAP_NORMALIZATION_PERCENTILES_ADAPTIVE_CDF_BASED_IN_EPOCH,
    min_max_percentiles=None,
)
callbacks_class_and_namempa_by_argsgroup[group.title] = (
    LogImageHeatmapTableCallback,
    cli_arg_name_map,
)

# PER-INSTANCE MEAN - score
group = parser.add_argument_group("perinstance_mean_score_test")
cli_arg_name_map = LogPerInstanceMeanCallback.cli_add_arguments(
    group,
    mean_of="score",
    stage="test",
    # defaults
    values_key="score_maps",
    labels_key="labels",
)
callbacks_class_and_namempa_by_argsgroup[group.title] = (
    LogPerInstanceMeanCallback,
    cli_arg_name_map,
)

# PER-INSTANCE MEAN - loss
group = parser.add_argument_group("perinstance_mean_loss_test")
cli_arg_name_map = LogPerInstanceMeanCallback.cli_add_arguments(
    group,
    mean_of="loss",
    stage="test",
    # defaults
    values_key="loss_maps",
    labels_key="labels",
)
callbacks_class_and_namempa_by_argsgroup[group.title] = (
    LogPerInstanceMeanCallback,
    cli_arg_name_map,
)

del group, cli_arg_name_map  # avoid bugs

# >>>>>>>>>>>>>>>>>>> parse <<<<<<<<<<<<<<<<<<<<<<

argv = sys.argv[1:]
print(f"argv: {argv}")

args = parser.parse_args(argv)
print(f"args: {args}")

# split the args in groups
# src: https://stackoverflow.com/a/46929320/9582881
args_bygroup = {
    group.title: Namespace(**{
        a.dest: getattr(args, a.dest, None) 
        for a in group._group_actions + [
            aa
            for subgroup in group._action_groups
            for aa in subgroup._group_actions
        ]
    })
    for group in parser._action_groups
}
print(f"args_bygroup: {args_bygroup}")

args_run = args_bygroup["run"]
args_runone = args_bygroup["runone"]

# >>>>>>>>>>>>>>>>>>> process args of run() <<<<<<<<<<<<<<<<<<<<<<

CudaVisibleDevices.consume_arguments(args_run)
WandbOffline.consume_arguments(args_run)

base_rundir = LogdirBaserundir.consume_arguments(
    args_run, start_time, subfolders=(args_run.wandb_project, args_runone.dataset)
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
wandb_init_config_extra = dict(
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
confighashes_keys = dict(
    confighash_full=(
        "epochs",
        "learning_rate",
        "weight_decay",
        "model",
        "loss",
        "optimizer",
        "scheduler",
        "scheduler_parameters",
        "dataset",
        "raw_shape",
        "net_shape",
        "batch_size",
        "preprocessing",
        "supervise_mode",
        "real_anomaly_limit",
        "normal_class",
    ),
    confighash_dcs=("datset", "normal_class", "supervise_mode"),
    confighash_dcsl=("datset", "normal_class", "supervise_mode", "loss"),
    confighash_slurm=(
        # the info here should be very redundant but it's ok
        "slurm_job_id",
        "slurm_array_job_id",
        "slurm_array_task_id",
        "slurm_task_pid",
        "slurm_job_user",
        "slurm_job_name",
        "slurm_submit_host",
        "slurm_cluster_name",
        "slurmd_nodename",
        "slurm_job_partition",
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

train_dev01_bis.args_validate_dataset_specific_choices(args_runone)
train_dev01_bis.args_validate_model_specific_choices(args_runone)


# >>>>>>>>>>>>>>>>>>> process args of callbacks[] <<<<<<<<<<<<<<<<<<<<<<

# start by parsing callback stuff because they last parser must do parse_args() instead of parse_known_args()
callbacks = []

for group_title, (
    klass,
    cli_arg_name_map,
) in callbacks_class_and_namempa_by_argsgroup.items():

    print(f"group_title: {group_title}")
    print(f"callback: {klass.__name__}")
    print(f"cli_arg_name_map: {cli_arg_name_map}")

    callback_args = args_bygroup[group_title]
    print(f"callback_args (BEFORE name ): {callback_args}")

    callback_args = {
        argname: getattr(callback_args, cliname)
        for cliname, argname in cli_arg_name_map.items()
    }
    print(f"callback_args (AFTER name ): {callback_args}")

    callbacks.append(klass(**callback_args))

# ========================================================================= LAUNCH

results = train_dev01_bis.run(
    **vars(args_run),
    runone_common_kwargs=vars(args_runone),
    callbacks=callbacks,
)

print("end")


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
