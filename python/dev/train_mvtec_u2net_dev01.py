#!/usr/bin/env python
# coding: utf-8

from argparse import ArgumentParser
from pickletools import optimize

import torch

import model_dev01
import mvtec_dataset_dev01
import train_dev01
from callbacks_dev01 import LOG_HISTOGRAM_MODE_LOG

# i had to do this because RSU (for the u2net) uses bilineal interpolation, which does not have a deterministic implementation
# there is also the option 
# lightning_deterministic=True,
# down here
torch.use_deterministic_algorithms(False)

parser = train_dev01.parser_add_arguments(ArgumentParser())
parser.set_defaults(
    # >>>>>>>>>>>>>>>>>>>>> from u2net
    # [training]
    epochs=500,
    optimizer=model_dev01.OPTIMIZER_ADAM,
    # optimizer parameters ----------------
    # also defaults in torch.optim.Adam
    learning_rate=1e-3,
    weight_decay=0,
    # these two are hardcoded
    # betas=(0.9, 0.999),
    # eps=1e-8,
    # -------------------------------------
    # original: 12 with 320**2 images
    # equivalent: 12 * (288/224)**2 ~= 12 * 1.7 = 20 --> 24 
    batch_size=24,
    # lightning_accumulate_grad_batches=2,
    # ===========================================================================
    # >>>>>>>>>>>>>>>>>>>>> from fcdd DEFAULTS
    # [training]
    # epochs=500,
    # learning_rate=1e-3,
    # weight_decay=1e-4, 
    # optimizer=model_dev01.OPTIMIZER_SGD,
    # scheduler=model_dev01.SCHEDULER_LAMBDA,
    # scheduler_paramaters=[0.98],
    # original: 128 wo/ grad accumulation
    # batch_size=32,  
    # lightning_accumulate_grad_batches=4,
    # ===========================================================================
    # >>>>>>>>>>>>>>>>>>>>> experimental
    # learning_rate=1e-2,
    # lightning_accumulate_grad_batches=5,  # half epoch for class 0
    # ===========================================================================
    # >>>>>>>>>>>>>>>>>>>>> common 
    # [model]  
    model=model_dev01.MODEL_U2NET_HEIGHT4_LITE,
    # model=model_dev01.MODEL_U2NET_HEIGHT6_FULL,
    loss=model_dev01.LOSS_U2NET_ALLSIDES_PIXELWISE_BATCH_AVG,
    # [dataset]
    dataset=mvtec_dataset_dev01.DATASET_NAME,
    raw_shape=(240, 240),
    net_shape=(224, 224),
    nworkers=1,
    pin_memory=False,
    preprocessing=mvtec_dataset_dev01.PREPROCESSING_LCNAUG1,
    supervise_mode=mvtec_dataset_dev01.SUPERVISE_MODE_REAL_ANOMALY,
    real_anomaly_limit=1,
    # [script]
    test=True,
    preview_nimages=0,
    # n_seeds=1,
    # seeds=None,
    classes=None,
    # [wandb]
    wandb_project="mvtec-debug",
    wandb_tags=["dev01", "u2net"],
    wandb_profile=False,
    wandb_offline=False,
    wandb_watch=None,
    wandb_watch_log_freq=100,  # wandb's default
    wandb_checkpoint_mode=train_dev01.WANDB_CHECKPOINT_MODE_LAST,
    # [wandb(train/validation/test)]
    wandb_log_roc=(False, False, True),
    wandb_log_pr=(False, False, True),
    wandb_log_image_heatmap_contrast_percentiles=(3., 97.),  # (contrast_min, contrast_max)
    wandb_log_image_heatmap_nsamples=(0, 0, 30),
    wandb_log_image_heatmap_resolution=(None, None, None),
    wandb_log_histogram_score=(None, None, None),
    wandb_log_histogram_loss=(None, None, None),
    # wandb_log_percentiles_score_train=(0., 1., 2., 5., 10., 90., 95., 98., 99., 100.,),
    wandb_log_percentiles_score_train=(),
    # wandb_log_percentiles_score_validation=(0., 1., 2., 5., 10., 90., 95., 98., 99., 100.,),
    wandb_log_percentiles_score_validation=(),
    wandb_log_perinstance_mean_score = (False, False, False),
    wandb_log_perinstance_mean_loss = (False, False, False),
    # [pytorch lightning] 
    lightning_accelerator=train_dev01.LIGHTNING_ACCELERATOR_GPU,
    lightning_ndevices=1,
    lightning_strategy=None,
    lightning_precision=train_dev01.LIGHTNING_PRECISION_32,
    lightning_model_summary_max_depth=1,
    lightning_check_val_every_n_epoch=10,
    lightning_profiler=train_dev01.LIGHTNING_PROFILER_SIMPLE,
    lightning_gradient_clip_val=1.0,
    lightning_gradient_clip_algorithm=train_dev01.LIGHTNING_GRADIENT_CLIP_ALGORITHM_VALUE,
    lightning_deterministic=False,
)
args = parser.parse_args()
train_dev01.args_validate_dataset_specific_choices(args)
train_dev01.args_validate_model_specific_choices(args)
args = train_dev01.args_post_parse(args)
results = train_dev01.run(**vars(args))
