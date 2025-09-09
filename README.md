[![REUSE status](https://api.reuse.software/badge/github.com/SAP/knn-sampler)](https://api.reuse.software/info/github.com/SAP/knn-sampler)

# Missing values handling

This project is a Python application for missing-value imputation and reproducing the experiments from the [publication](https://).

## knnSampler imputation algorithm

`knnSampler` is a kNN-based method for missing-value imputation with support for multiple imputation and uncertainty quantification (see the [publication](https://)).

## How to cite

If you use knnSampler, please cite the original [publication](https://).

## Running the project

### Prerequisites

This project requires Python 3.8+ and [poetry](https://python-poetry.org/).

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

The project uses a self-documented configuration file `assets/config.conf`.   
The default configuration reproduces the results from the [publication](https://) section(s) ... .

```shell
poetry run task main
```

Runs the main imputation pipeline using `assets/config.conf`.

### Benchmark algorithms

The benchmark reproduces results from the [publication](https://) section(s) ... .  
Note: the benchmarking scripts have no dedicated configuration files. To change benchmark settings, edit the top of [benchmark_all.py](./benchmark_all.py) and [benchmark_knnsampler.py](./benchmark_knnsampler.py).

#### Benchmark all algorithms

For comparing imputation algorithms with each other.

```shell
poetry run task benchmark_all
```

#### Benchmark knnSampler

For detailed knnSampler results with different parameter intervals.

```shell
poetry run task benchmark_knnsampler
```

## Contributing

This project is open to feature requests and bug reports via [GitHub issues](https://github.com/SAP/knn-sampler/issues). Contribution and feedback are welcome. See [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

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

### Install code quality git hooks

```shell
poetry run pre-commit install
```

## Security / Disclosure

If you find a bug that may pose a security issue, follow our [security policy](https://github.com/SAP/knn-sampler/security/policy) instructions to report it. Do not open GitHub issues for security reports.

## Code of Conduct

By participating in this project, you agree to follow our [Code of Conduct](https://github.com/SAP/.github/blob/main/CODE_OF_CONDUCT.md).

## License

Copyright 2025 SAP SE or an SAP affiliate company and knnSampler contributors. See [LICENSE](./LICENSE) for details. Detailed third-party licensing information is available via the [REUSE tool](https://api.reuse.software/info/github.com/SAP/knn-sampler).
