#!/bin/bash
#SBATCH --output /gpfs/radev/home/meb266/project/AdaptiveGorilla/env.d/spaths/slurm/%A_%a.out
#SBATCH --array 0-39
#SBATCH --job-name gorillas
#SBATCH --mem=4GB --cpus-per-task=8 --partition=day --time=30 --chdir=/gpfs/radev/home/meb266/project/AdaptiveGorilla

# DO NOT EDIT LINE BELOW
/gpfs/radev/apps/avx512/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/radev/project/yildirim/meb266/AdaptiveGorilla/scripts/most/joblist.txt --status-dir /gpfs/radev/home/meb266/project/AdaptiveGorilla/env.d/spaths/slurm

