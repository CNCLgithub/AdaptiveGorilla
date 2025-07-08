# AdaptiveGorilla

> Did you see it?

Implementation of "It's a feature not a bug: Multi-granular world models explain inattentional blindness"

## Citation

```bib
@article{belledonne_feature,
  author   = "Belledonne, Mario and Yildirim, Ilker",
  title    = "It's a feature not a bug: Multi-granular world models explain inattentional blindness",
  journal  = "TBD",
  year     = TBD,
  volume   = "TBD",
  number   = "TBD",
  pages    = "TBD",
}
```

> Note: This is a work in progress.

## News

- `2025/06/12`: A version of this work will be presented at CCN 2025!

## Installation


This project runs on [Apptainer](https://apptainer.org/) for a reproducible environment. To setup from scratch, simply clone the repo and download the container packets (detailed instructions below).

1. Clone the repo: `git clone git@github.com:CNCLgithub/AdaptiveGorilla.git`
2. Navigate to the repo directory: `cd AdaptiveGorilla`
3. Download container and datasets: `./env.d/setup.sh env_pull`
4. Setup the julia environment with `./env.d/setup.sh julia`
5. Initialize the julia project `./env.d/run.sh julia --project=.` followed by `using Pkg; Pkg.precompile()`


### Details

The command `./env.d/setup.sh` downloads the container, relevant Julia dependencies, and datasets for this project. The Julia dependencies are bundled to ensure exact reproduciblity. Typically, these dependencies would be included in the Apptainer container directly, however, due to Julia's JIT behavior, this would require several additional layers of complexity (JIT would require write permissions which are not allowed in a `.sif` container). I found it more straightforward to simply point Julia (see `env.d/default.conf`) to a folder on the host machine. 


## Contribution

Pull requests welcome!

In general please

1. Fork the repo
2. Make necessary commits to the relevant branch
3. Submit a PR, thanks!

