# Random Fourier Features based SLAM

This repository contains python implementation of method from the paper [Random Fourier Features based SLAM](https://arxiv.org/pdf/2011.00594.pdf).


## Table of content
  - [Repository structure](#repository-structure)
  - [Installation](#installation)
  - [Usage](#usage)

## Repository structure

```bash
.
├── README.md
├── requirements.txt
├── setup.py
├── run_experiment.sh
├── dump
├── data
├── figs
├── notebooks
|   ├── plot_results.ipynb
|   └── generate_data.ipynb
└── src
    ├── __init__.py
    ├── main.py
    ├── model.py
    ├── random_features.py
    ├── data_utils.py
    ├── demo.ipynb
    ├── experiments.ipynb
    ├── observation.py
    └── utils.py
```
## Installation

in the home dir:

```bash
cd [VENV]
virtualenv rff-slam
source rff-slam/bin/activate
```

back in the project dir:

```bash
pip install -r requirements.txt
pip intsall -e .
chmod +x ./run_experiment.sh
```

## Usage

```bash
python src/main.py -v --observation_model range-bearing
```

```bash
./run_experiment.sh
```