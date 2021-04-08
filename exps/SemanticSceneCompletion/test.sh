#!/usr/bin/env bash
#export OMP_NUM_THREADS=4

GPUNUM=8
NODENUM=1
JOBNAME=test
PART=vi_irdc

TOOLS="srun --partition=$PART --gres=gpu:${GPUNUM} -n$NODENUM --ntasks-per-node=1"

$TOOLS --job-name=$JOBNAME python eval.py -e 249-250 -d 0-1
