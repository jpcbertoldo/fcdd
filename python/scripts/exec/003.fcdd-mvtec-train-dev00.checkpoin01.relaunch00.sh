#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an 

THIS_SCRIPT_DIR=$( cd ${0%/*} && pwd -P )
THIS_SCRIPT_DIR=$( realpath ${THIS_SCRIPT_DIR} )

cd $HOME

SBATCH_SCRIPT_FPATH="/cluster/CMM/home/jcasagrandebertoldo/repos/fcdd/python/scripts/sbatch/fcdd-mvtec-train-dev00-checkpoint01.sh"

# debug params
# --cls-restrictions 0 --it 1 --epochs 1 --no-test

# // A) SUPervised, MASK, old loss
sbatch ${SBATCH_SCRIPT_FPATH} \
--supervise-mode noise --noise-mode mvtec_gt \
--cls-restrictions 4 5 6 7 8 9 10 11 12 13 14 
sleep 3  # avoid the same directory name

# // C) SUPervised, MASK, pixel wise loss (fixed)
sbatch ${SBATCH_SCRIPT_FPATH} \
--supervise-mode noise --noise-mode mvtec_gt \
--cls-restrictions 4 5 6 7 8 9 10 11 12 13 14 \
--pixel-loss-fix 
sleep 3  # avoid the same directory name

# ============================================================ later ============================================================

# # // B) unsupervised (confetti), MASK, old loss
# # sbatch ${SBATCH_SCRIPT_FPATH} \
# --supervise-mode malformed_normal_gt --noise-mode confetti \
# --cls-restrictions 4 5 6 7 8 9 10 11 12 13 14 
# sleep 3  # avoid the same directory name

# ============================================================ nope ============================================================

# already running
# # // D) unsupervised (confetti), MASK, pixel wise loss (fixed)
# sbatch ${SBATCH_SCRIPT_FPATH} \
# --supervise-mode malformed_normal_gt --noise-mode confetti \
# --cls-restrictions 4 5 6 7 8 9 10 11 12 13 14 \
# --pixel-loss-fix 
# sleep 3  # avoid the same directory name
