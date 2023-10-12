#!/bin/bash
#SBATCH --job-name=kek     # create a short name for your job
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
python train.py \
    -c hw_asr/configs/conformer_13.json
#    -c hw_asr/configs/conformer_11.json
#    -c hw_asr/configs/conformer_10.json
#    -c hw_asr/configs/conformer_9.json -r saved/models/conformer_v9/1009_175335/model_best.pth
#    -c hw_asr/configs/conformer_8.json
#    -c hw_asr/configs/conformer_6.json
#    -c hw_asr/configs/conformer_5.json
#    -c hw_asr/configs/conformer_4.json
#    -c hw_asr/configs/conformer_2.json
#    -c hw_asr/configs/conformer_3.json
#    -c hw_asr/configs/conformer_1.json
#    -c hw_asr/configs/one_batch_conformer.json
#    -c hw_asr/configs/one_batch_test.json 