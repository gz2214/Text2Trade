#!/bin/bash -e
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=model_train_2-2-2-16-100-1
#SBATCH --mail-type=END
##SBATCH --mail-user=sh7008@nyu.edu
#SBATCH --output=/scratch/sh7008/Text2Trade/progress.out

module purge

singularity exec --nv \
--overlay /scratch/sh7008/overlay-256GB-500K.ext3:rw \
--overlay /scratch/sh7008/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; conda activate 2590-hw3; python3 main.py true"
