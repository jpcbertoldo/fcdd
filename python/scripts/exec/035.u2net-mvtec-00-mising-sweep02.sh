#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# ==============================================================================

SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/fcdd/python/scripts/sbatch/06-nparallel-arrayargs-01-1gpu-2cpuptask.slurm"

export SBATCH_SCRIPT_ARG_WORKDIR="${HOME}/fcdd/python/dev"
export SBATCH_SCRIPT_ARG_CONDAENV="fcdd_rc21"

# NOT THE USUAL TRAIN SCRIPT!!!
export SBATCH_SCRIPT_ARG_SCRIPT_FNAME="unetdd_mvtec_00_sweep_02_train_missingfromsweep.py"

export SBATCH_SCRIPT_ARG_ARRAY_ARGS_FPATH="${SCRIPT_DIR}/035.u2net-mvtec-00-arrayargs"
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

sbatch ${SBATCH_SCRIPT_FPATH} ${ARGS} ${DEBUG_ARGS}
