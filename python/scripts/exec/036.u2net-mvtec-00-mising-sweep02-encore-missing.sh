#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# ==============================================================================

SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/fcdd/python/scripts/sbatch/07-nparallel-arrayargs-01-1gpu-2cpuptask.slurm"
echo "SBATCH_SCRIPT_FPATH=${SBATCH_SCRIPT_FPATH}"

export SBATCH_SCRIPT_ARG_WORKDIR="${HOME}/fcdd/python/dev"
export SBATCH_SCRIPT_ARG_CONDAENV="fcdd_rc21"

# NOT THE USUAL TRAIN SCRIPT!!!
export SBATCH_SCRIPT_ARG_SCRIPT_FNAME="unetdd_mvtec_00_sweep_02_train_missingfromsweep.py"

export SBATCH_SCRIPT_ARG_ARRAY_ARGS_FPATH="null"
# export SBATCH_SCRIPT_ARG_ARRAY_ARGS_FPATH="${SCRIPT_DIR}/035.u2net-mvtec-00-arrayargs"
export SBATCH_SCRIPT_ARG_NPARALLEL_RUNS=1

ARGS=""
ARGS="--wandb_tags run-n-parallel nparallel:${SBATCH_SCRIPT_ARG_NPARALLEL_RUNS}"
ARGS="${ARGS} --wandb_tags fix-missing-sweep-02"
echo "ARGS=${ARGS}"


DEBUG_ARGS=""
# DEBUG_ARGS="${DEBUG_ARGS} --epochs 1"
# DEBUG_ARGS="${DEBUG_ARGS} --wandb_offline --wandb_checkpoint_mode none"
# DEBUG_ARGS="${DEBUG_ARGS} --wandb_tags debug"
# DEBUG_ARGS="${DEBUG_ARGS} --wandb_tags deleteme"
# DEBUG_ARGS="${DEBUG_ARGS} --wandb_project mvtec-debug"
echo "DEBUG_ARGS=${DEBUG_ARGS}"

sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS} ${DEBUG_ARGS} --supervise_mode real-anomaly --classes 1 --seeds 0x2395859d21efbbd6
sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS} ${DEBUG_ARGS} --supervise_mode real-anomaly --classes 1 --seeds 0xeb33f3d140c7b5f4
sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS} ${DEBUG_ARGS} --supervise_mode real-anomaly --classes 1 --seeds 0x40452d6f87068d81
sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS} ${DEBUG_ARGS} --supervise_mode real-anomaly --classes 1 --seeds 0x22cd096d5f8a9c21
sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS} ${DEBUG_ARGS} --supervise_mode real-anomaly --classes 1 --seeds 0x526f289f74732498





