#!/bin/bash
#SBATCH --job-name=kek2     # create a short name for your job
#SBATCH --partition=rnd
#SBATCH --exclude=99dgx-09,99dgx-08
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
##SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --output=slurm_outputs/%j.out  # to write outputs
#SBATCH -e slurm_outputs/%j.err  # to write errs


export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


# to add conda stuff
# source /home/${USER}/.bashrc
conda activate asr_env

unset TMPDIR 

srun \
python test.py \
    --batch-size 32 \
    --jobs 8 \
    -c default_test_model/config-other.json \
    #-c default_test_model/config-clean.json \
