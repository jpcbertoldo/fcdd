#!/usr/bin/env python
# coding: utf-8

from argparse import ArgumentParser

import model_dev01
import mvtec_dataset_dev01
import train_dev01
from callbacks_dev01 import LOG_HISTOGRAM_MODE_NONE, LOG_HISTOGRAM_MODE_LOG

parser = train_dev01.parser_add_arguments(ArgumentParser())
parser.set_defaults(
    # training
    epochs=50, 
    learning_rate=1e-3,
    weight_decay=1e-4, 
    # model 
    model=model_dev01.FCDD_CNN224_VGG.__name__,
    loss=model_dev01.LOSS_PIXELWISE_BATCH_AVG,
    optimizer=model_dev01.OPTIMIZER_SGD,
    scheduler=model_dev01.SCHEDULER_LAMBDA,
    scheduler_paramaters=[0.985],
    # dataset
    dataset=mvtec_dataset_dev01.DATASET_NAME,
    raw_shape=(260, 260),
    net_shape=(224, 224),
    batch_size=128,
    nworkers=2,
    pin_memory=False,
    preprocessing=mvtec_dataset_dev01.PREPROCESSING_LCNAUG1,
    supervise_mode=mvtec_dataset_dev01.SUPERVISE_MODE_SYNTHETIC_ANOMALY_CONFETTI,
    real_anomaly_limit=1,
    # heatmap
    gauss_std=12, 
    # script
    test=True,
    preview_nimages=5,
    n_seeds=3,
    # seeds=None,
    classes=None,
    # files
    # wandb
    wandb_project="mvtec-debug",
    wandb_tags=["dev01",],
    wandb_profile=False,
    wandb_offline=False,
    wandb_watch=train_dev01.WANDB_WATCH_NONE,
    wandb_watch_log_freq=100,  # wandb's default
    wandb_checkpoint_mode=train_dev01.WANDB_CHECKPOINT_MODE_LAST,
    # train/validation/test
    wandb_log_roc=(False, True, True),
    wandb_log_pr=(False, True, True),
    wandb_log_score_histogram=(LOG_HISTOGRAM_MODE_NONE, LOG_HISTOGRAM_MODE_NONE, LOG_HISTOGRAM_MODE_LOG),
    wandb_log_loss_histogram=(LOG_HISTOGRAM_MODE_NONE, LOG_HISTOGRAM_MODE_NONE, LOG_HISTOGRAM_MODE_LOG),
    # pytorch lightning 
    lightning_accelerator=train_dev01.LIGHTNING_ACCELERATOR_GPU,
    lightning_ndevices=1,
    lightning_strategy=train_dev01.LIGHTNING_STRATEGY_NONE,
    lightning_precision=train_dev01.LIGHTNING_PRECISION_32,
    lightning_model_summary_max_depth=4,
    lightning_check_val_every_n_epoch=1,
    lightning_accumulate_grad_batches=1,
    lightning_profiler="simple",
)
args = parser.parse_args()
train_dev01.args_validate_dataset_specific_choices(args)
train_dev01.args_validate_model_specific_choices(args)
args = train_dev01.args_post_parse(args)
results = train_dev01.run(**vars(args))
