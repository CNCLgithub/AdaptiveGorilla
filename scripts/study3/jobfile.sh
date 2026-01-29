#!/bin/bash
#SBATCH --output /gpfs/radev/home/meb266/project/AdaptiveGorilla/env.d/spaths/slurm/%A_%a.out
#SBATCH --array 0-39
#SBATCH --job-name gorillas
#SBATCH --mem=2GB --cpus-per-task=1 --partition=day --time=45 --chdir=/gpfs/radev/home/meb266/project/AdaptiveGorilla

# DO NOT EDIT LINE BELOW
/gpfs/radev/apps/avx512/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/radev/project/yildirim/meb266/AdaptiveGorilla/scripts/study3/joblist.txt --status-dir /gpfs/radev/home/meb266/project/AdaptiveGorilla/env.d/spaths/slurm

