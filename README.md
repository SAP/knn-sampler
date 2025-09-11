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
@misc{pashmchi2025knnsamplerstochasticimputationsrecovering,
      title={kNNSampler: Stochastic Imputations for Recovering Missing Value Distributions}, 
      author={Parastoo Pashmchi and Jerome Benoit and Motonobu Kanagawa},
      year={2025},
      eprint={2509.08366},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2509.08366}, 
}
```

## Running the project

### Prerequisites

This project requires Python 3.8+ and [Poetry](https://python-poetry.org/).

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

### Benchmark algorithms

Note: the benchmarking scripts do not have dedicated configuration files. To change benchmark settings, edit the top of [benchmark_all.py](./benchmark_all.py) and [benchmark_knnsampler.py](./benchmark_knnsampler.py).

#### Benchmark all algorithms

For comparing imputation algorithms with each other.

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
