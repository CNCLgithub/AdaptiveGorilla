# Study 1

These scripts correspond to Study 1, demonstrating the first model of SIB, based off [Most et. al 2001](https://doi.org/10.1111/1467-9280.00303).


## Scripts

- `dataset.jl`: Generates a series of trials. The exact copy of the script used can be found under `env.d/spaths/datasets/study1/script.jl`
- `run_model.jl`: Runs a model variant on the given dataset, the results will be located at `env.d/spaths/experiments/study1`
- `aggregate_runs.jl`: Combines all of the runs across models in to a csv file: `env.d/spaths/experiments/study1/aggregate.csv`.
`

## Auxillary scripts

- `gen_joblist.sh`: A helper script to create a joblist for Yale's [dSQ](https://github.com/ycrc/dsq) program.
- `dsq-jobfile.sh`: The file produced by dSQ that can be submitted to `sbatch`. (Note: the file should be regenerated on your own slurm cluster, see below)


## Running on a cluster

This code ran on Yale's HPC managed by YCRC, using SLURM. This code can be easily run on a local machine via the command line or Julia REPL (see individual scripts).

However, to reproducibly run the entire setup, the details for submitting `run_model.jl` to a SLURM cluster are described below. 

1. Create a joblist, where each line defines a single call to `run_model.jl` (one for each scene). This can be done with `gen_joblist.sh`

2. Use `dsq` to create a SLURM batch file. 
We used: 
```sh
dsq -J gorillas --status-dir "${PWD}/env.d/spaths/slurm" --batch-file scripts/sib/dsq-jobfile.sh --job-file scripts/sib/joblist.txt --partition=day --cpus-per-task=8 --mem=4GB --time=60 --chdir="${PWD}" --output="${PWD}/env.d/spaths/slurm/%A_%a.out"

```

Yielding:
``` sh
#SBATCH --output /gpfs/radev/home/meb266/scratch/AdaptiveGorilla/env.d/spaths/slurm/%A_%a.out
#SBATCH --array 0-5
#SBATCH --job-name gorillas
#SBATCH --partition=day --cpus-per-task=8 --mem=4GB --time=60 --chdir=/gpfs/radev/home/meb266/scratch/AdaptiveGorilla

# DO NOT EDIT LINE BELOW
/gpfs/radev/apps/avx512/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/radev/scratch/yildirim/meb266/AdaptiveGorilla/scripts/study1/joblist.txt --status-dir /gpfs/radev/home/meb266/scratch/AdaptiveGorilla/env.d/spaths/slurm
```

3. Submit the batch job to slurm: `sbatch scripts/study1/dsq-jobfile.sh`


The results should appear under `env.d/spaths/experiments/study1`.


## Replicating Analyses
