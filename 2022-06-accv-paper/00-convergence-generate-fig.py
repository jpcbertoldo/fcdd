# execute with genv

#In[]
# make a cell print all the outputs instead of just the last one
from IPython.core.interactiveshell import InteractiveShell
from pyparsing import col
InteractiveShell.ast_node_interactivity = "all"

#%%
from pathlib import Path
import pandas as pd
import numpy as np

# %%

data_dirpath = (Path.home() / "fcdd/2022-06-accv-paper").resolve().absolute()
f"{data_dirpath=}"

# i filtered the runs that i want from wandb and downloaded the csv
# so i can get the run names then use it to filter when getting the data from the api
runs_fpath = data_dirpath / "00-selected-runs.csv"
assert runs_fpath.exists()

run_names = set(pd.read_csv(runs_fpath)["Name"].unique())
f"{len(run_names)=}"

# %%

data_fpath = data_dirpath / "00-data.csv"

if data_fpath.exists():
    all_df = pd.read_csv(data_fpath)
    raise Exception(f"{data_fpath} already exists")

import wandb
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("mines-paristech-cmm/fcdd-mvtec-dev00-checkpoint02")
summary_list = [] 
config_list = [] 
name_list = []
runid_list = []
for run in runs: 
    # run.summary are the output key/values like accuracy.
    # We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict) 

    # run.config is the input metrics.
    # We remove special values that start with _.
    config = {k:v for k,v in run.config.items() if not k.startswith('_')}
    config_list.append(config) 

    name_list.append(run.name)
    runid_list.append(run.id) 

import pandas as pd 
summary_df = pd.DataFrame.from_records(summary_list) 
config_df = pd.DataFrame.from_records(config_list) 
name_df = pd.DataFrame({'name': name_list}) 
runid_df = pd.DataFrame({'runid': runid_list}) 
all_df = pd.concat([runid_df, name_df, config_df,summary_df], axis=1)

all_df.to_csv(data_fpath)

# %%

runs_fpath = data_dirpath / "00-runs.csv"

if runs_fpath.exists():
    runs = pd.read_csv(runs_fpath)
    raise Exception(f"{runs_fpath} already exists")

runs = all_df[all_df.name.isin(run_names)]
f"{runs.shape=}"

selectec_columns = ["runid", "name", "normal_class", "loss_mode", "noise_mode"]
runs = runs[selectec_columns]
 
runs.to_csv(runs_fpath)

# %%

hists = []
for runid in runs["runid"]:
    run = api.run(f"mines-paristech-cmm/fcdd-mvtec-dev00-checkpoint02/{runid}")
    hists.append([row["test_rocauc"] for row in run.scan_history()])
   
# %%

nepochs = len(hists[0])
epoch_colnames = list(range(nepochs)) 

rocauc = pd.DataFrame(
    hists, 
    columns=epoch_colnames, 
    index=runs.set_index(["noise_mode", "normal_class", "loss_mode"]).index
).T
rocauc.index.names = ["epoch"]
rocauc = rocauc.T.groupby(axis=0, level=(0, 1, 2)).agg([("mean", np.mean), ("min", np.min), ("max", np.max), ("std", np.std)]).T
# rocauc.index.names = rocauc.index.names[:-1] + ["metric"]
rocauc = rocauc.swaplevel(0, 1, axis=0)

# %%

CLASSES_LABELS = (
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
    'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
    'wood', 'zipper'
)


# %%
from matplotlib import pyplot as plt
import itertools

nclasses = 15
nepochs = 50
noise_mode = "mvtec_gt"
# noise_mode = "confetti"

fig, axs = plt.subplots(
    nrows := 4, ncols := 4, 
    dpi=100, figsize=((sz := 1.3) * ncols * 3, sz * nrows * 1.5),
    sharex=True, sharey=True,
)
fig.set_tight_layout(True)

twins = np.empty_like(axs)
for i, j in itertools.product(range(nrows), range(ncols)):
    twins[i, j] = axs[i, j].twinx()

for rowidx, twinrow in enumerate(twins):
    twinrow[0].get_shared_y_axes().join(*twinrow.tolist())
    # this is because the last row has an empty ax
    lastcol = -1 if rowidx < nrows -1  else -2
    for tw in twinrow[:lastcol]:
        tw.yaxis.set_ticklabels([])

loss0 = "pixel-level-balanced"
loss1 = "pixel-wise-averages-per-image"

for ax, twin, classidx in zip(axs.ravel(), twins.ravel(), range(nclasses)):
    ax.set_title(f"class {classidx}: {CLASSES_LABELS[classidx]}")
    ax.set_ylim(70, 100)
    
    ax_df = 100 * rocauc[noise_mode][classidx]    
    twin.set_ylim(0, 40)
        
    def get_mean_var(l: str):
        return ax_df[l].loc["mean"], ax_df[l].loc["max"] - ax_df[l].loc["min"]
    
    mean, var = get_mean_var(loss0)
    ax.plot(range(nepochs), mean, "-", label=f"{loss0} mean", color="blue")
    twin.plot(range(nepochs), var, "-", label=f"{loss0} max-min", color="red")

    mean, var = get_mean_var(loss1)
    ax.plot(range(nepochs), mean, "--", label=f"{loss0} mean", color="blue")
    twin.plot(range(nepochs), var, "--", label=f"{loss0} max-min", color="red")

axs[-1, -1].axis("off")
twins[-1, -1].axis("off")

# %%
from matplotlib import pyplot as plt
import itertools

nclasses = 15
nepochs = 50
# noise_mode = "mvtec_gt"
noise_mode = "confetti"

suptitle_translation = {
    "mvtec_gt": "semi-supervised (1 real anomaly / anomaly type)",
    "confetti": "unsupervised (synthetic confetti noise)",
}

fig, axs = plt.subplots(
    nrows := 4, ncols := 4, 
    dpi=100, figsize=((sz := 1.3) * ncols * 3, sz * nrows * 1.5),
    sharex=True, sharey=True,
)

twins = np.empty_like(axs)
for i, j in itertools.product(range(nrows), range(ncols)):
    twins[i, j] = axs[i, j].twinx()

for rowidx, (twinrow, axrow) in enumerate(zip(twins, axs)):
    axrow[0].set_ylabel("Mean test AUROC (%)")
    twinrow[0].get_shared_y_axes().join(*twinrow.tolist())
    # this is because the last row has an empty ax
    lastcol = -1 if rowidx < nrows -1  else -2
    twinrow[lastcol].set_ylabel("Std test AUROC (%)", rotation=270, labelpad=10)
    for tw in twinrow[:lastcol]:
        tw.yaxis.set_ticklabels([])

loss0 = "pixel-level-balanced"
loss1 = "pixel-wise-averages-per-image"

loss_translation = {
    loss0: "per-batch average",
    loss1: "per-image average",
}

for ax, twin, classidx in zip(axs.ravel(), twins.ravel(), range(nclasses)):
    ax.set_title(f"class {classidx}: {CLASSES_LABELS[classidx]}")
    ax.set_ylim(70, 100)
    
    ax_df = 100 * rocauc[noise_mode][classidx]    
    twin.set_ylim(0, 10)
        
    def get_mean_var(l: str):
        return ax_df[l].loc["mean"], ax_df[l].loc["std"]
    
    mean, var = get_mean_var(loss0)
    ax.plot(range(nepochs), mean, "-", label=f"{loss_translation[loss0]} mean", color="blue")
    twin.plot(range(nepochs), var, "-", label=f"{loss_translation[loss0]} max-min", color="red")

    mean, var = get_mean_var(loss1)
    ax.plot(range(nepochs), mean, "--", label=f"{loss_translation[loss1]} std", color="blue")
    twin.plot(range(nepochs), var, "--", label=f"{loss_translation[loss1]} std", color="red")

axs[-1, -1].axis("off")
twins[-1, -1].axis("off")
fig.set_tight_layout(True)

fig.suptitle(f"Test AUROC: mean and std at each epoch\n{suptitle_translation[noise_mode]}")

handles, labels = axs[0, 0].get_legend_handles_labels()
thandles, tlabels = twins[0, 0].get_legend_handles_labels()
handles, labels = handles + thandles, labels + tlabels
axs[-1, -1].legend(handles, labels, loc="center")