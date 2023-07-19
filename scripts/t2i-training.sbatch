#!/bin/bash
#
#SBATCH --job-name=testDiT-XL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=30:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:a100:4
#SBATCH --output=outlog/DiT_cfg1.5_a1004_%j.out

module purge

export OMP_NUM_THREADS=12

singularity exec --nv \
	    --overlay /scratch/nm3607/container/DiT_container.ext3:ro \
		--overlay /vast/work/public/ml-datasets/coco/coco-2017.sqf:ro \
	    		  /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	    		  /bin/bash -c \
		"source /ext3/env.sh; conda activate DiT; \
		torchrun --nnodes=1 --nproc_per_node=4 train.py \
		--model DiT-XL/2 \
		--num-workers 4 \
		--data-path /scratch/nm3607/datasets/coco/ \
		--ckpt /scratch/nm3607/DiT/t2i-results/022-DiT-XL-2/checkpoints/0150000.pt \
		--global-batch-size 128 \
		--cfg-scale 1.5"