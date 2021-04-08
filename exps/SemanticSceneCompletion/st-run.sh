#!/usr/bin/env bash
#export OMP_NUM_THREADS=4

GPUNUM=16
NODENUM=1
JOBNAME=demo

PART=vi_irdc

TOOLS="srun --partition=$PART --gres=gpu:${GPUNUM} -n$NODENUM --ntasks-per-node=1"

$TOOLS --job-name=$JOBNAME sh -c "python -m torch.distributed.launch --nnodes=$NODENUM --nproc_per_node=$GPUNUM --node_rank \$SLURM_PROCID --master_addr=\$(sinfo -Nh -n \$SLURM_NODELIST | head -n 1 | cut -d ' ' -f 1) --master_port 29058 train.py"

