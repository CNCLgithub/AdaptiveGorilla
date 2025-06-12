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

1. Clone the repo `git clone git@github.com:CNCLgithub/AdaptiveGorilla.git`
2. Navigate to the repo directory `cd AdaptiveGorilla`
3. Setup the docker environment with `./env.d/setup.sh cont_build`
4. Setup the julia environment with `./env.d/setup.sh julia`
5. Initialize the julia project `./env.d/run.sh julia --project=.` followed by `using Pkg; Pkg.initialize()`

If you run into an issue with one of the unregistered repos, you can simply add them again with `Pkg.add(<THE GIT URL>)`

### Docker

- May need to configure gpu driver: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#rootless-mode

## Contribution

Pull requests welcome!

In general please

1. Fork the repo
2. Make necessary commits to the relevant branch
3. Submit a PR, thanks!

