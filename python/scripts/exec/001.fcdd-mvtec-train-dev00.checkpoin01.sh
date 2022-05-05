#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 

THIS_SCRIPT_DIR=$( cd ${0%/*} && pwd -P )
THIS_SCRIPT_DIR=$( realpath ${THIS_SCRIPT_DIR} )

cd $HOME

SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/repos/fcdd/python/scripts/sbatch/fcdd-mvtec-train-dev00-checkpoint01.sh"

# debug params
# --cls-restrictions 0 --it 1 --epochs 1 --no-test

# // C) SUPervised, MASK, old loss
# // "--supervise-mode", "noise", "--noise-mode", "mvtec_gt",
sbatch ${SBATCH_SCRIPT_FPATH} --supervise-mode noise --noise-mode mvtec_gt
sleep 3  # avoid the same directory name

# // D) unsupervised (confetti), MASK, old loss
# // "--supervise-mode", "malformed_normal_gt", "--noise-mode", "confetti",
sbatch ${SBATCH_SCRIPT_FPATH} --supervise-mode malformed_normal_gt --noise-mode confetti
sleep 3  # avoid the same directory name

# // E) SUPervised, MASK, pixel level loss
# // "--supervise-mode", "noise", "--noise-mode", "mvtec_gt", "--pixel-level-loss",
sbatch ${SBATCH_SCRIPT_FPATH} --supervise-mode noise --noise-mode mvtec_gt --pixel-level-loss
sleep 3  # avoid the same directory name

# // F) unsupervised (confetti), MASK, pixel level loss
# // "--supervise-mode", "malformed_normal_gt", "--noise-mode", "confetti", "--pixel-level-loss",
sbatch ${SBATCH_SCRIPT_FPATH} --supervise-mode malformed_normal_gt --noise-mode confetti --pixel-level-loss
sleep 3  # avoid the same directory name
