## 607_Studio_7


### Overview 
This project was created to investigate how heavy-tailed error distributions affect the performance of linear regression estimators in high-dimensional settings.

We compare three estimators:

| Method | Description |
|---------|--------------|
| **OLS** | Ordinary Least Squares — sensitive to outliers and heavy tails |
| **LAD** | Least Absolute Deviation (Quantile Regression, τ=0.5) — robust to outliers |
| **Huber** | Huber Regression — interpolates between OLS and LAD |

All outputs are stored in the `results/` directory.

### Repository Structure
```
607_Studio_7/
├── src/
│ ├── estimators.py # OLS, LAD, Huber wrappers 
│ ├── evaluation.py # mean-squared error 
│ ├── setup.py # data generation (design, β, errors)
│ └── visualization.py # plotting 
├── run_experiment.py # main script (runs all simulations)
├── results/
│ ├── data/ # stores simulation_results.csv
│ └── figures/ # stores generated plots
├── tests/ # simple pytest sanity checks
├── requirements.txt
└── README.md # this file
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/lzkostina/607_Studio_7
cd 607_Studio_7
```
2. Create environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage
```bash
python run_experiment.py \
  --n 500 \
  --dfs 1 2 3 20 inf \
  --gammas 0.2 0.5 0.8 \
  --rhos 0.0 0.5 \
  --snrs 1 5 10 \
  --reps 30 \
  --center-X --standardize-X \
  --master-seed 20251014
```
This code allows to run full simulation study and generates the full results file results/data/simulation_results.csv

```bash
python -m src.visualization --input results/data/simulation_results.csv \
                        --out-main results/figures/mse_vs_df.png \
                        --small-multiples


```
By running the code above you will be able to recreate figures stored in results/figures
### Testing

This project includes basic tests to ensure pipeline correctness.

To run all tests:
```bash
pytest tests/
```
