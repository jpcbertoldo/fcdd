"""
copy and paste of the notebook with the same name
"""
from pathlib import Path
dev_dir = Path("../python/dev").resolve()
dev_dir

WANDB_ENTITY = "mines-paristech-cmm"
WANDB_PROJECT = "fcdd-mvtec-dev00-checkpoint02"
WANDB_ENTITY_PROJECT = f"{WANDB_ENTITY}/{WANDB_PROJECT}"

WANDB_SWEEP_ID = None
WANDB_SWEEP_PATH = f"{WANDB_ENTITY_PROJECT}/{WANDB_SWEEP_ID}" if WANDB_SWEEP_ID else None

import wandb
api = wandb.Api()

sweep = api.sweep(WANDB_SWEEP_PATH) if WANDB_SWEEP_PATH else None
runs = api.runs(WANDB_ENTITY_PROJECT) if sweep is None else sweep.runs

from collections import defaultdict
lists = defaultdict(list)

def append_to_list(key, value):
    lists[key].append(value)

for run in runs: 
    append_to_list("summary", run.summary._json_dict)
    append_to_list("config", {k: v for k,v in run.config.items() if not k.startswith('_')})
    append_to_list("name", run.name)
    append_to_list("tags", run.tags)
    append_to_list("id", run.id)
    append_to_list("state", run.state)
    
import pandas as pd
runs_df = pd.DataFrame.from_dict(data=lists)

runs_df.shape
runs_df.columns


runs_df_filtered = runs_df

is_report_bmvc_02 = runs_df['tags'].apply(lambda x: "report-bmvc-02" in x)
runs_df_filtered = runs_df[is_report_bmvc_02]

runs_df_filtered.shape


import numpy as np

def extract(df, from_column, key):
    return df[from_column].apply(lambda x: x.get(key, None))

df = runs_df_filtered
config_keys = [
    "loss_mode",
    "noise_mode",
    "normal_class",
    "normal_class_label",
    "logdir",
    "datadir",
    "preproc",
    "batch_size",
]
summary_keys = [
    "test_rocauc",
]
for key in config_keys:
    df[key] = extract(df, "config", key)
for key in summary_keys:
    df[key] = extract(df, "summary", key)
del df
runs_df_filtered.columns
runs_df_filtered[config_keys].head(5)


from mvtec_dataset_dev01_bis import MVTecAnomalyDetectionDataModule, DATAMODULE_PREPROCESS_MOMENT_BEFORE_BATCH_TRANSFER, SUPERVISE_MODE_SYNTHETIC_ANOMALY_CONFETTI
import torch
from pytorch_lightning.loggers import WandbLogger
import wandb
from model_dev01_bis import FCDD
from pytorch_lightning import Trainer
from callbacks_dev01_bis import LogPrcurveCallback
from common_dev01_bis import create_python_random_generator


for idx, (rowidx, row) in enumerate(runs_df_filtered.iterrows()):
        
    runid = row["id"]
    run = api.run(f"{WANDB_ENTITY_PROJECT}/{runid}")
    
    # if "test/avg-precision" in run.summary.keys():
    #     print(f"skipping {runid}")
    #     continue
    
    logdir = row["logdir"]
    batch_size = int(row["batch_size"])
    normal_class = row["normal_class"]
    preproc = row["preproc"]
    datadir = row["datadir"]
        
    net = FCDD(
        in_shape=(224, 224),
        model_name="FCDD_CNN224_VGG_F",
        # these values dont matter
        optimizer_name="sgd",  
        lr=1e-3,
        weight_decay=1e-5,
        scheduler_name="lambda",
        scheduler_parameters=[.999],
        loss_name="old-fcdd",
        dropout_mode=None,
        dropout_parameters=[],
    )
    
    snapshot_fpath = (dev_dir / logdir / "snapshot.pt").resolve()

    if torch.cuda.is_available():
        snapshot = torch.load(snapshot_fpath)

    else:
        snapshot = torch.load(snapshot_fpath, map_location=torch.device('cpu'))

    net.load_state_dict(snapshot.pop('net', None))
    net.eval();
    
    wandb_logger = WandbLogger(id=runid, project=WANDB_PROJECT, entity=WANDB_ENTITY,)
    trainer = Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        gpus=1, 
        logger=wandb_logger,  
        callbacks=[
            LogPrcurveCallback(
                scores_key="score_maps", gt_key="gtmaps", log_curve=False, limit_points=None,
                # doesnt matter because limit_points is None
                python_generator=create_python_random_generator(0), stage="test",
            )
        ], 
    )
    
    trainer.test(
        model=net, 
        datamodule=MVTecAnomalyDetectionDataModule(
            root=(dev_dir / datadir).resolve(),
            normal_class=normal_class,
            preprocessing=preproc,
            preprocess_moment=DATAMODULE_PREPROCESS_MOMENT_BEFORE_BATCH_TRANSFER,
            supervise_mode=SUPERVISE_MODE_SYNTHETIC_ANOMALY_CONFETTI,
            batch_size=batch_size, 
            nworkers=0,
            pin_memory=False,
            seed=0,
            raw_shape=(240, 240),
            net_shape=(224, 224),
            real_anomaly_limit=0,
        )
    ) 
    
    wandb_logger.close()
    wandb.finish()