#!/bin/bash
#
#SBATCH --job-name=sampleDiT
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=40:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:4
#SBATCH --out=outlog/t2i-sample_%j.out

module purge

singularity exec --nv \
	    --overlay /scratch/nm3607/container/DiT_sample.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	    /bin/bash -c "source /ext3/env.sh; conda activate DiT; \
		torchrun --nnodes=1 --nproc_per_node=4 sample_ddp.py \
		--model DiT-XL/2 \
		--num-fid-samples 25000 \
		--ckpt /scratch/nm3607/DiT/t2i-results/023-DiT-XL-2/checkpoints/0250000.pt \
		--data-path /scratch/nm3607/datasets/coco \
		--cfg-scale 1.5"