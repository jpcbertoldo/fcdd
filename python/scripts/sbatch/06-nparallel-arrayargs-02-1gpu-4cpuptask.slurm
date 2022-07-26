#! /bin/bash
 
# Personalisation de la tache

#SBATCH --job-name   06-nparallel-arrayargs
#SBATCH --partition  cmm-gpu
#SBATCH --gres       gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --array=0-100%3   # Id of job start-end % number of simultaneous running job

# variable meanings
# %x = job-name
# %N = node-name
# %A = SLURM_ARRAY_JOB_ID (same for all jobs in the same array)
# %a = SLURM_ARRAY_TASK_ID (the job index in the array)
# %j = SLURM_JOBID (unique for each job)

# OUTPUT FOR ARRAY JOB
#SBATCH --output     /cluster/CMM/home/jcasagrandebertoldo/log/nparallel-arrayargs/%x-%A_%a-%j-%N.log
 
#SBATCH --mail-type  ALL
#SBATCH --mail-user joaopcbertoldo@gmail.com

# obs : la ligne suivante est necessaire pour forces l'exécution 
. $HOME/.bashrc

# constants
ARRAY_ARGS_FPATH_NULL="null"
END_OF_JOB_CONTINUE_ARRAY="continue-array"
END_OF_JOB_END_ARRAY="end-array"

CANCEL_AND_EXIT_MSG="canceling the array job SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID} and exiting"
END_AND_EXIT_MSG="ending the array job SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID} (cancel pending jobs) and exiting"

function cancel_array_and_exit() {
    scancel ${SLURM_ARRAY_JOB_ID}
	exit 1
}

function end_array_and_exit() {
    # src: https://stackoverflow.com/a/47325966/9582881
    # this will cancel the array job and all pending jobs in the array
    scancel -t PD ${SLURM_ARRAY_JOB_ID}
	exit 0
}

function check_env_var() {
    if [ -z "${!1}" ]; then
        echo "ERROR: ${1} is not set"
        echo "${CANCEL_AND_EXIT_MSG}"
        cancel_array_and_exit
    else
        echo "${1}=${!1}"
    fi
}

# ==============================================================================
echo "==================== SBATCH ARGUMENTS (FROM ENV VARS) ===================="
# ==============================================================================

check_env_var "SBATCH_SCRIPT_ARG_NPARALLEL_RUNS"
check_env_var "SBATCH_SCRIPT_ARG_ARRAY_ARGS_FPATH"

check_env_var "SBATCH_SCRIPT_ARG_WORKDIR"
check_env_var "SBATCH_SCRIPT_ARG_SCRIPT_FNAME"
check_env_var "SBATCH_SCRIPT_ARG_CONDAENV"

# exec
NPARALLEL_RUNS="${SBATCH_SCRIPT_ARG_NPARALLEL_RUNS}"
ARRAY_ARGS_FPATH="${SBATCH_SCRIPT_ARG_ARRAY_ARGS_FPATH}"

WorkDir="${SBATCH_SCRIPT_ARG_WORKDIR}"
SCRIPT_FNAME="${SBATCH_SCRIPT_ARG_SCRIPT_FNAME}"
CONDA_ENV_NAME="${SBATCH_SCRIPT_ARG_CONDAENV}"

# ==============================================================================
echo "=============================== VARIABLES ================================"
# ==============================================================================

INIT_CONDA_BASH_FPATH="${HOME}/init-conda-bash"
SLEEP_BETWEEN_LAUNCHS_SECS=7
Node=$(hostname)
MYTMPDIR="/cluster/CMM/data1/jcasagrandebertoldo/tmp"
# as in the SBATCH output
JOB_ARRAY_IDENTIFIER="${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}"
JOB_FULL_IDENTIFIER="${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}-${SLURM_JOB_ID}-${Node}"

# logging

BASE_LOGDIR="${HOME}/log/nparallel-arrayargs"
LOGDIR="${BASE_LOGDIR}/${JOB_ARRAY_IDENTIFIER}"

echo "creating log dir"
echo "LOGDIR=${LOGDIR}"
mkdir -p ${LOGDIR}

LOGFILE_FNAME="${JOB_FULL_IDENTIFIER}.log"

function move_logfile_to_logdir() {
    echo "moving logfile to logdir"
    mv "${BASE_LOGDIR}/${LOGFILE_FNAME}" "${LOGDIR}"
}

JNAME_BASE="${JOB_FULL_IDENTIFIER}.python-"
JPATH_BASE="${LOGDIR}/${JNAME_BASE}"
JPATH_EXT=".log"

echo "Node=$Node"
echo "WorkDir=$WorkDir"
echo "CONDA_ENV_NAME=$CONDA_ENV_NAME"
echo "SCRIPT_FNAME=$SCRIPT_FNAME"
echo "MYTMPDIR = ${MYTMPDIR}"
echo "LOGDIR=${LOGDIR}"
echo "JNAME_BASE=${JNAME_BASE}"
echo "JPATH_BASE=${JPATH_BASE}"
echo "JPATH_EXT=${JPATH_EXT}"   


# ==============================================================================
echo "========================== N PARALLEL RUNS STUFF ========================="
# ==============================================================================

IS_INT_REGEX='^[0-9]+$'
if [[ ! ${NPARALLEL_RUNS} =~ $IS_INT_REGEX ]] || [ ${NPARALLEL_RUNS} -lt 1 ]
then
	echo "ERROR: NPARALLEL_RUNS is not a valid integer >= 1"
    echo "NPARALLEL_RUNS=${NPARALLEL_RUNS}"
    echo "${CANCEL_AND_EXIT_MSG}"
    move_logfile_to_logdir
    cancel_array_and_exit
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

JPATHS=()

for idx in ${INDICES[@]}
do
    JPATH="${JPATH_BASE}${idx}${JPATH_EXT}"
    echo "idx=${idx}"
    echo "JPATH=${JPATH}"
    JPATHS+=($JPATH)
done

# ==============================================================================
echo "============================= ARRAY TASK ARGS ============================"
# ==============================================================================

if [ "${ARRAY_ARGS_FPATH}" == "${ARRAY_ARGS_FPATH_NULL}" ]
then
    echo "ARRAY_ARGS_FPATH=${ARRAY_ARGS_FPATH} is null"
    echo "there are no array task args, array will be canceled at the end of this job"

    ARRAY_TASK_ARGS=""
    END_OF_JOB="${END_OF_JOB_END_ARRAY}"

else
    echo "reading array task args from file"
    echo "ARRAY_ARGS_FPATH=${ARRAY_ARGS_FPATH}"

    # check if file exists
    if [ ! -f "${ARRAY_ARGS_FPATH}" ]
    then
        echo "ERROR: ARRAY_ARGS_FPATH=${ARRAY_ARGS_FPATH} does not exist"
        echo "${CANCEL_AND_EXIT_MSG}"
       move_logfile_to_logdir
        cancel_array_and_exit
    fi

    readarray -t ARRAY_ARGS_LINES < "${ARRAY_ARGS_FPATH}"

    N_LINES=${#ARRAY_ARGS_LINES[@]}
    echo "number of lines in array task args file: ${N_LINES}"

    # check if file is empty
    if [ "${N_LINES}" -eq 0 ]
    then
        echo "ERROR: the file is empty, it should have one set of arguments per line; if you want to run without array args then pass the value <<${ARRAY_ARGS_FPATH_NULL}>> for ARRAY_ARGS_FPATH"
        echo "${CANCEL_AND_EXIT_MSG}"
        move_logfile_to_logdir
        cancel_array_and_exit
    fi

    # check if it is the end of args
    if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_LINES}" ]
    then
        echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} does not have any array task args anymore"
        echo "${END_AND_EXIT_MSG}"
        move_logfile_to_logdir
        end_array_and_exit
    fi

    echo "reading array task args from line SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    ARRAY_TASK_ARGS="${ARRAY_ARGS_LINES[${SLURM_ARRAY_TASK_ID}]}"
    END_OF_JOB="${END_OF_JOB_CONTINUE_ARRAY}"

fi

echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "ARRAY_TASK_ARGS=${ARRAY_TASK_ARGS}"
echo "END_OF_JOB=${END_OF_JOB}"

# ==============================================================================
echo "================================== ARGS =================================="
# ==============================================================================

ARGS=$*
echo "ARGS=${ARGS}"
echo "joining ARGS and ARRAY_TASK_ARGS"
ARGS="${ARGS} ${ARRAY_TASK_ARGS}"
echo "ARGS=${ARGS}"

# ==============================================================================
echo "================================= SETUP =================================="
# ==============================================================================

echo "going to WorkDir"
cd $WorkDir

echo "loading conda"
source ${INIT_CONDA_BASH_FPATH}

echo "activating conda env"
conda activate $CONDA_ENV_NAME

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

echo "END_OF_JOB=${END_OF_JOB}"

if [ "${END_OF_JOB}" == "${END_OF_JOB_CONTINUE_ARRAY}" ]
then
    echo "continuing the array, exiting this job"
    move_logfile_to_logdir
    exit 0

elif [ "${END_OF_JOB}" == "${END_OF_JOB_END_ARRAY}" ]
then
    echo "${END_AND_EXIT_MSG}"
    move_logfile_to_logdir
    end_array_and_exit

else
    echo "unknown END_OF_JOB=${END_OF_JOB}"
    echo "${CANCEL_AND_EXIT_MSG}"
    move_logfile_to_logdir
    cancel_array_and_exit
fi