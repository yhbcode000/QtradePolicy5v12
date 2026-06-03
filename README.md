# Qtrade Policy 5 v12

Portfolio strategy evaluator for comparing the Qtrade Policy 5 v12 policy set and generating CSV outputs, plots, and a Markdown report.

## Installation

Install the project in editable mode from the repository root:

```bash
python -m pip install -e .
```

The editable install uses the runtime dependencies declared in `pyproject.toml`:

- `matplotlib`
- `numpy`
- `pandas`
- `yfinance`

## Startup

You can run the engine directly from the Python module file:

```bash
python portfolio_policy_eval_v12_report.py engine
```

After installing the project, you can also use the console script entry point:

```bash
qtrade-policy5 engine
```

The `engine` command is the default command. The CLI also accepts compatibility flags for one-shot, non-browser, non-IG smoke runs.

## Smoke check

Use this documented smoke check after packaging changes or in a fresh virtual environment:

```bash
python -m pip install -e .
qtrade-policy5 engine --once --no-browser --no-ig
```

The command downloads market data through `yfinance`, so it requires internet access and may fail if Yahoo Finance data is unavailable for the configured date range.

## Outputs

A successful run writes CSV files in the repository root, chart images under `plots/`, downloaded data under `data/`, and a Markdown report under `report/`.
