# AdaptiveGorilla

> Did you see it?

Implementation of "Multigranular Abstractions Drive Human Visual Awareness"

**Mario Belledonne and Ilker Yildirim.** Department of Psychology, Yale University.

> [!NOTE]
> The experiment and behavioral-analysis code lives in a separate repository and is included here as the [`experiment`](https://github.com/CNCLgithub/ib-jspsych) git submodule under `/experiment`.

## Citation

TBD!

> Note: This is a work in progress.

## News

- `2026/03/10`: A version of this work will be presented at MODVIS+VSS 2026!
- `2025/06/12`: Presented a checkpoint at CCN 2025.

## Overview

This repository implements the **Multigranular Optimization (MO)** model, an algorithmic theory of goal-driven visual awareness. Rather than a fixed representational frame, MO dynamically constructs **multigranular abstractions** over incoming sensory inputs: task-relevant objects are represented in fine detail as individuals, while task-irrelevant objects are coarsened into group summaries. Group representations can "explain away" the sensory detections of unexpected objects that are consistent with them, which grounds inattentional blindness in the model.

MO consists of three components:

1. **Multigranular world models** — a reversible, information-conserving space of abstractions ("granularity frames") over individual and group representations, connected by the *reframe kernel* `κ` (merge/split moves).
2. **Granularity efficiency** `℧_G` — an online-computable measure of the task-efficiency of a frame, the ratio of aggregate task-relevance `E_Δ` to the proportion of task-irrelevant representations `A_Δ`.
3. **Granularity optimization** — a bilevel (hierarchical particle-filter) genetic-style search over frames that runs in near real time (< 1% overhead for reframing/resampling) on a single CPU core.

Across three studies MO (i) achieves substantial gains in runtime, accuracy, and memory use relative to resource-matched fixed-granularity and non-goal-conditioned controls, (ii) recapitulates the classic appearance-dependent awareness patterns of sustained inattentional blindness, and (iii) predicts — and a preregistered study confirms — a novel *functional irrelevance* effect at both the condition and trial level.

## Model variants

The code implements MO alongside three ablation controls (matched in total computational resources):

- `mo` — the full Multigranular Optimization model.
- `ja` — *Just Attention*: fixed granularity (all individuals) with adaptive computation, no reframing.
- `ta` — *Task Agnostic*: reframing without adaptive computation, optimizing description length.
- `fr` — *Fixed Resource*: a standard particle filter at fixed granularity with uniform processing.

Variant parameters live under `scripts/params/*.toml`.

## Organization

- `experiment`: git submodule ([`ib-jspsych`](https://github.com/CNCLgithub/ib-jspsych)) containing the jsPsych experiment implementation and behavioral analysis code.
- `scripts`: Each of the three studies corresponds to a sub-directory under `scripts/study<n>`, with their own `README`s. Additional `scripts/sensitivity-<n>` directories hold parameter-sensitivity analyses.
- `src`: implements the model under a Julia package (`AdaptiveGorilla`).
- `tests`: various test scripts (not complete)
- `env.d`: computing environment.


## Installation

This project runs on [Apptainer](https://apptainer.org/) for a reproducible environment. To setup from scratch, simply clone the repo (with submodules) and download the container packets (detailed instructions below).

```sh
git clone --recurse-submodules git@github.com:CNCLgithub/MultigranularOptimization.git
cd MultigranularOptimization
./env.d/setup.sh env_pull    # container and datasets
./env.d/setup.sh julia       # julia environment
./env.d/run.sh julia --project=. -e 'using Pkg; Pkg.precompile()'
```

If you already cloned the repository, fetch the `experiment` submodule with:

```sh
git submodule update --init --recursive
```


### Details

The command `./env.d/setup.sh` downloads the container, relevant Julia dependencies, and datasets for this project. The Julia dependencies are bundled to ensure exact reproduciblity. Typically, these dependencies would be included in the Apptainer container directly, however, due to Julia's JIT behavior, this would require several additional layers of complexity (JIT would require write permissions which are not allowed in a `.sif` container). I found it more straightforward to simply point Julia (see `env.d/default.conf`) to a folder on the host machine.

## Study name mapping

Study numbering differs between the manuscript and this repository's `scripts/` directories:

| Manuscript | Repository | Description |
| --- | --- | --- |
| Study 1 | `scripts/study3` | Load curve (tractability and performance under increasing load) |
| Study 2 | `scripts/study1` | Sustained inattentional blindness, appearance effect (Most et al., 2001) |
| Study 3 | `scripts/study2` | Functional irrelevance effect (target-ensemble) |

## Simulation studies

The script directory names do not correspond one-to-one to the manuscript study numbers:

| Manuscript study | Script directory | Content |
| --- | --- | --- |
| Study 1 (tractability and performance under load) | `scripts/study3` | Load curve |
| Study 2 (appearance effect in inattentional blindness) | `scripts/study1` | Sustained inattentional blindness (Most et al., 2001) |
| Study 3 (functional irrelevance) | `scripts/study2` | Target-ensemble / irrelevant targets |

Each study under `scripts/study<n>` follows the same pattern:

- `dataset.jl`: generates the trials for that study.
- `run_model.jl`: runs a model variant on the dataset; results are written to `env.d/spaths/experiments/study<n>`.
- `aggregate_runs.jl`: combines all runs across model variants into `env.d/spaths/experiments/study<n>/aggregate.csv`.

Parameter-sensitivity analyses are under `scripts/sensitivity-<n>`.

### Mapping to manuscript studies

| Script directory | Manuscript study |
| --- | --- |
| `scripts/study1` | Study 1: Tractability and performance benefits under load |
| `scripts/study2` | Study 2: Recapitulating the "appearance" effect in inattentional blindness |
| `scripts/study3` | Study 3: Confirming a novel prediction on human awareness (functional irrelevance) |
| `scripts/sensitivity-1` | Supplementary: parameter sensitivity analysis 1 |
| `scripts/sensitivity-2` | Supplementary: parameter sensitivity analysis 2 |

## Running on a cluster

The studies were run on Yale's HPC managed by YCRC, using SLURM. The code can also be run on a local machine via the command line or the Julia REPL (see the per-study `README`s).

To reproducibly run an entire study on a SLURM cluster:

1. Create a joblist, where each line defines a single call to `run_model.jl` (one for each scene), using `gen_joblist.sh`.
2. Use Yale's [dSQ](https://github.com/ycrc/dsq) to create and submit a SLURM batch file:

```sh
dsq -J gorillas \
  --status-dir "${PWD}/env.d/spaths/slurm" \
  --batch-file scripts/study<n>/dsq-jobfile.sh \
  --job-file scripts/study<n>/joblist.txt \
  --partition=day --cpus-per-task=8 --mem=4GB --time=60 \
  --chdir="${PWD}" \
  --output="${PWD}/env.d/spaths/slurm/%A_%a.out"
```

## Contribution

Pull requests welcome!

In general please

1. Fork the repo
2. Make necessary commits to the relevant branch
3. Submit a PR, thanks!

