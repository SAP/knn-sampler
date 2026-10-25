[![REUSE status](https://api.reuse.software/badge/github.com/SAP/knn-sampler)](https://api.reuse.software/info/github.com/SAP/knn-sampler)

# Missing-value imputation

This project is a Python application for missing-value imputation and for reproducing the experiments from the publication [kNNSampler: Stochastic Imputations for Recovering Missing Value Distributions](https://arxiv.org/abs/2509.08366).
It's a joint work by [SAP SE](https://www.sap.com/) and [Eurecom](https://www.eurecom.fr/) with funding support from the [ANRT](https://www.anrt.asso.fr/).

## knnSampler imputation algorithm

**knnSampler** is a kNN-based method for missing-value imputation with support for multiple imputation and uncertainty quantification.
It aims to preserve the underlying data distribution when imputing missing values (see the [publication](https://arxiv.org/abs/2509.08366) for more details).

## How to cite

If you use knnSampler, please cite the original [publication](https://arxiv.org/abs/2509.08366):

```bibtex
@article{pashmchi2025knnsampler,
  title={k{NNS}ampler: Stochastic Imputations for Recovering Missing Value Distributions},
  author={Parastoo PASHMCHI and J{\'e}r{\^o}me Benoit and Motonobu Kanagawa},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=4CDnIACCQG},
  note={}
}
```

## Running the project

### Prerequisites

This project requires Python 3.12+ and [Poetry](https://python-poetry.org/) 2+.

1. Clone the repository

```shell
git clone <repository_url>
```

2. Navigate to the project directory

```shell
cd <repository_directory>
```

3. Install dependencies

```shell
poetry install --no-root
```

### Run algorithms

The project uses a self-documenting configuration file [`assets/config.conf`](./assets/config.conf).

```shell
poetry run task main
```

Runs the main imputation pipeline using [`assets/config.conf`](./assets/config.conf).

For end-to-end reproducible stochastic runs, set the dataset `random_state`
in the dataset section and `execution_random_state` in the `[execution]`
section of [`assets/config.conf`](./assets/config.conf). The dataset seed
controls data preparation and missing-value selection; the execution seed
controls stochastic imputation. When set, each
`do_imputation(..., iterations=N)` iteration receives an independent derived
seed, while `task main` performs one execution per imputer. Leave the execution
seed blank to keep stochastic strategies unseeded. `KNNxKDEImputer` still uses
global randomness internally and is not fully reproducible.

For use from Python, load settings with `assets.config.load_config()`.

K-fold selection uses `optimal_k_random_state` (default `42`, `kfold` only,
integer in `0..4294967295`; `None` is Python-API-only). Its scaler is fitted
independently on each training fold; `loo_penalized` keeps the historical
global scaling. The historical module-override path remains supported: when
every required legacy field is set directly on `assets.config`, configuration
files are not read and `execution_random_state` defaults to `None` unless set
explicitly.

### Benchmark algorithms

Note: the benchmarking scripts do not have dedicated configuration files. To change benchmark settings, edit the top of [benchmark_all.py](./benchmark_all.py) and [benchmark_knnsampler.py](./benchmark_knnsampler.py).

#### Benchmark all algorithms

For comparing imputation algorithms with each other.

A fixed benchmark seed reproduces the prepared datasets. Stochastic imputations,
permutation tests and execution times may still vary.

```shell
poetry run task benchmark_all
```

#### Benchmark knnSampler

For detailed knnSampler results with different parameter ranges.

```shell
poetry run task benchmark_knnsampler
```

## Contributing

This project is open to feature requests and bug reports via [GitHub issues](https://github.com/SAP/knn-sampler/issues). Contributions and feedback are welcome. See [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

### Install development dependencies

```shell
poetry install --no-root --with dev
```

### Code formatting

```shell
poetry run task format
```

### Code linting

```shell
poetry run task lint
```

### Code testing

```shell
poetry run task test
```

### Static type checking

```shell
poetry run task typecheck
```

### Install code-quality Git hooks

```shell
poetry run pre-commit install
```

## Security / Disclosure

If you find a bug that may pose a security issue, follow our [security policy](https://github.com/SAP/knn-sampler/security/policy) instructions to report it. Do not open GitHub issues for security reports.

## Code of Conduct

By participating in this project, you agree to follow our [Code of Conduct](https://github.com/SAP/.github/blob/main/CODE_OF_CONDUCT.md).

## License

Copyright 2025 SAP SE or an SAP affiliate company and knnSampler contributors. See [LICENSE](./LICENSE) for details. Detailed third-party licensing information is available via the [REUSE tool](https://api.reuse.software/info/github.com/SAP/knn-sampler).
