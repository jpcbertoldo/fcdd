#! /bin/bash
 
# Personalisation de la tache

#SBATCH --job-name   03-fcdd-mvtec-train-dev01-run-n-parallel
#SBATCH --partition  cmm-gpu
#SBATCH --gres       gpu:1
#SBATCH --cpus-per-task 2

#SBATCH --array=0-5%1   # Id of job start-end % number of simultaneous running job

# variable meanings
# %x = job-name
# %N = node-name
# %A = SLURM_ARRAY_JOB_ID (same for all jobs in the same array)
# %a = SLURM_ARRAY_TASK_ID (the job index in the array)
# %j = SLURM_JOBID (unique for each job)

# !!!!! THIS ONE !!!!!! 
# OUTPUT FOR ARRAY JOB
#SBATCH --output     /cluster/CMM/home/jcasagrandebertoldo/log/fcdd/mvtec/run-n-parallel/%x-%N-%A_%a-%j.log
 
# OUTPUT FOR SINGLE JOB
##SBATCH --output     /cluster/CMM/home/jcasagrandebertoldo/log/fcdd/mvtec/run-n-parallel/%x-%N-%j.log

# tous les evenements pertinents seront envoyes par email a cette adresse
#SBATCH --mail-type  ALL
#SBATCH --mail-user joaopcbertoldo@gmail.com


# obs : la ligne suivante est necessaire pour forces l'exécution 
. $HOME/.bashrc

# ==============================================================================
echo "================================== ARGS =================================="
# ==============================================================================

# src: https://www.delftstack.com/howto/linux/split-string-into-array-in-bash/#:~:text=IFS%3D%27%2C%27%20read%20%2Dra%20ip_array%20%3C%3C%3C%20%22%24addrs%22
# src: https://phoenixnap.com/kb/bash-read#:~:text=variable%20by%20default.-,Bash%20read%20Options,-The%20Bash%20read
# -r disable \ to escape characters
# -a <array> puts the array into <array>
# convert $* to array of strings
read -ra ARGS <<< "$*"

# capture only the number of parallel runs from the args and rebuild it without it
# it is supposed to be the first pair of key-value
ARG0=${ARGS[0]}
ARG1=${ARGS[1]}
ARGS=(${ARGS[@]:2})

# rebuild ARGS as a string (not array)
TMP="${ARGS[0]}"
for arg in "${ARGS[@]:1}"
do
	TMP="${TMP} ${arg}"
done
unset ARGS
ARGS=$TMP

# check that the nb of parallel runs is valid 
ARGNAME_NPARALLEL_RUNS="--slurm-n-parallel-runs"
IS_INT_REGEX='^[0-9]+$'

if [[ ${ARG0} != ${ARGNAME_NPARALLEL_RUNS} ]] || [[ ! ${ARG1} =~ $IS_INT_REGEX ]]
then
	echo "the first argument must be '${ARGNAME_NPARALLEL_RUNS}' followed by an integer > 0. Got '${ARG0} ${ARG1}'"
	exit 1
else
	NPARALLEL_RUNS=${ARG1}
fi

# without formatting
INDICES_=$(seq 0 $((${NPARALLEL_RUNS}-1)))

# formated values
INDICES=()
for idx_ in ${INDICES_[@]}
do
    INDICES+=( $(printf "%02d" "$idx_") )
done

echo "number of parrallel runs: ${NPARALLEL_RUNS}"
echo "INDICES=${INDICES[@]}"
echo "ARGS='${ARGS}'"

# SLURM_ARRAY_TASK_ID \in {}
# echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

# if (( ${SLURM_ARRAY_TASK_ID} >= 15 ))
# then
#     echo "SLURM_ARRAY_TASK_ID bad"
#     echo "exiting"
#     exit
# fi

echo "setting common args in slurm job" 

COMMON_ARGS=""
COMMON_ARGS="${COMMON_ARGS} --nworkers 1"
COMMON_ARGS="${COMMON_ARGS} --wandb-tags dev01 run-n-parallel-${NPARALLEL_RUNS}"
echo "COMMON_ARGS=${COMMON_ARGS}"

ARGS="${COMMON_ARGS} ${ARGS}"
echo "ARGS=${ARGS}"


# ==============================================================================
echo "=============================== VARIABLES ================================"
# ==============================================================================

SLEEP_BETWEEN_LAUNCHS_SECS=7

Node=$(hostname)

# exec
WorkDir=$HOME/fcdd/python/dev
CONDA_ENV_NAME="fcdd_rc21"
SCRIPT_FNAME="train_mvtec_dev01.py"

# etc
MYTMPDIR="/cluster/CMM/data1/jcasagrandebertoldo/tmp"

# logging
LOGDIR="${HOME}/log/fcdd/mvtec/run-n-parallel"

# OUTPUT FOR ARRAY JOB
JNAME_BASE="${SLURM_JOB_NAME}-${Node}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}-${SLURM_JOB_ID}.python-"

# OUTPUT FOR SINGLE JOB
# JNAME_BASE="${SLURM_JOB_NAME}-${Node}-${SLURM_JOB_ID}.python-"

JPATH_BASE=${LOGDIR}/$JNAME_BASE
JPATH_EXT=".log"

JPATHS=()

for idx in ${INDICES[@]}
do
    JPATH="${JPATH_BASE}${idx}${JPATH_EXT}"
    echo "idx=${idx}"
    echo "JPATH=${JPATH}"
    JPATHS+=($JPATH)
done

echo "Node=$Node"
echo "WorkDir=$WorkDir"
echo "CONDA_ENV_NAME=$CONDA_ENV_NAME"
echo "SCRIPT_FNAME=$SCRIPT_FNAME"
echo "MYTMPDIR = ${MYTMPDIR}"
echo "LOGDIR=${LOGDIR}"
echo "JNAME_BASE=${JNAME_BASE}"
echo "JPATH_BASE=${JPATH_BASE}"
echo "JPATH_EXT=${JPATH_EXT}"   
echo "JPATHS=${JPATHS[@]}"

# ==============================================================================
echo "================================= SETUP =================================="
# ==============================================================================

echo "going to WorkDir"
cd $WorkDir

echo "loading conda"
source ${HOME}/init-conda-bash

echo "activating conda env"
conda activate $CONDA_ENV_NAME

echo "creating log dir"
mkdir -p ${LOGDIR}

echo "changing tmp dir"
export TMPDIR=$MYTMPDIR

# ==============================================================================
echo "========================= HEALTH CHECKS =================================="
# ==============================================================================

echo "health checks"
echo "pwd = $(pwd)"
echo "which conda = $(which conda)"
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo "which python = $(which python)"
echo "TMPDIR=${TMPDIR}"
echo "nvcc --version = $(nvcc --version)"
echo "nvidia-smi"
nvidia-smi


# ==============================================================================
echo "================================ LAUNCH =================================="
# ==============================================================================

echo "launching processes"
PIDS=()

for idx in ${INDICES[@]}
do
    JPATH=${JPATHS[$idx]}
    
    echo "launching idx=${idx}"
    echo "JPATH=${JPATH}"
    
    # & will put it in the background
    python ${SCRIPT_FNAME} ${ARGS} > "${JPATH}" 2>&1 & 
    
    PID=$!
    echo "PID=${PID}"
    
    PIDS+=($PID)

    echo "sleeping for ${SLEEP_BETWEEN_LAUNCHS_SECS} seconds"
    sleep ${SLEEP_BETWEEN_LAUNCHS_SECS}

done

echo "PIDS=${PIDS[@]}"

# ==============================================================================
echo "================================== WAIT =================================="
# ==============================================================================

for pid in ${PIDS[@]}
do

    echo "waiting for pid=${pid}"
    wait ${pid}
    echo "pid=${pid} finished"

done

echo "all processes finished"
