#!/bin/bash
#SBATCH --output /gpfs/radev/home/meb266/scratch/AdaptiveGorilla/env.d/spaths/slurm/%A_%a.out
#SBATCH --array 0-5
#SBATCH --job-name gorillas
#SBATCH --partition=day --cpus-per-task=8 --mem=4GB --time=60 --chdir=/gpfs/radev/home/meb266/scratch/AdaptiveGorilla

# DO NOT EDIT LINE BELOW
/gpfs/radev/apps/avx512/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/radev/scratch/yildirim/meb266/AdaptiveGorilla/scripts/target_ensemble/AC-joblist.txt --status-dir /gpfs/radev/home/meb266/scratch/AdaptiveGorilla/env.d/spaths/slurm

