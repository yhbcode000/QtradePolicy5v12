# Qtrade Policy 5 v12

Policy 5 started as the most successful ratio-guarded backtest policy in this repository. It is now also available as a CLI trading-engine monitor that:

- downloads the latest 60 days of 15 minute bars from Yahoo Finance on startup;
- optionally polls IG Markets' REST API and cumulates mapped realtime snapshots into the current 15 minute bar;
- evaluates the Policy 5 buy/sell/hold decision loop;
- starts a local web page showing the current market line, account state, positions, and where/what/when action log.

## Install

```bash
python -m pip install -r requirements.txt
```


## `.env` configuration

On startup the app reads `.env` from the repo root before it reads process-level environment variables. Non-empty values in `.env` override same-named system environment variables, so local trading defaults and IG settings can live in one file. Empty values are ignored so placeholder keys do not erase real system credentials.

This repo includes both `.env` with safe local defaults and `.env.example` with every supported key. To configure your own account, copy/edit `.env.example` or update `.env` locally:

```bash
cp .env.example .env
# edit .env with IG_SERVICE_USERNAME, IG_SERVICE_PASSWORD, IG_SERVICE_API_KEY, etc.
python portfolio_policy_eval_v12_report.py engine
```

Supported engine defaults include `ENGINE_YAHOO_PERIOD`, `ENGINE_INTERVAL`, `ENGINE_POLL_SECONDS`, `ENGINE_HOST`, `ENGINE_PORT`, `ENGINE_TICKER`, `ENGINE_NO_BROWSER`, `ENGINE_NO_IG`, and `ENGINE_ENABLE_LIVE_TRADING`. IG keys include `IG_SERVICE_USERNAME`, `IG_SERVICE_PASSWORD`, `IG_SERVICE_API_KEY`, `IG_SERVICE_ACC_TYPE`, `IG_SERVICE_ACC_NUMBER`, `IG_EPIC_MAP`, `IG_DEAL_SIZE_MAP`, `IG_CURRENCY_CODE`, `IG_EXPIRY`, `IG_FORCE_OPEN`, and `IG_NO_CONFIRM_DEALS`.

## Start the trading engine

```bash
python portfolio_policy_eval_v12_report.py engine
```

The default command opens `http://127.0.0.1:8765/` in a browser. Use `--no-browser` when running on a headless server.

Useful options:

```bash
python portfolio_policy_eval_v12_report.py engine \
  --yahoo-period 60d \
  --interval 15m \
  --poll-seconds 60 \
  --ticker AVGO
```

The page refreshes automatically and includes:

- a current market line for the selected ticker;
- the latest Policy 5 decision (`BUY`, `SELL`, `HOLD`, `WATCH_BUY`, or a guarded skip action);
- account equity, free margin, and margin usage;
- the action log showing when an action was evaluated, where it applied, what happened, and why.

## IG realtime polling

The engine reads the following IG service credentials from system environment variables:

```bash
IG_SERVICE_USERNAME
IG_SERVICE_PASSWORD
IG_SERVICE_API_KEY
IG_SERVICE_ACC_TYPE
IG_SERVICE_ACC_NUMBER
```

Set `IG_SERVICE_ACC_TYPE=DEMO` for demo accounts or another value for the live gateway. To poll market snapshots, also provide an IG epic map with either `IG_EPIC_MAP` or `--ig-epic-map`:

```bash
export IG_EPIC_MAP='{"AVGO":"YOUR_AVGO_EPIC","MU":"YOUR_MU_EPIC"}'
python portfolio_policy_eval_v12_report.py engine
```

You may also pass a JSON file path:

```bash
python portfolio_policy_eval_v12_report.py engine --ig-epic-map ./ig_epics.json
```

If IG credentials or epics are missing, the engine continues with Yahoo Finance startup data and periodic Policy 5 evaluation only. The implemented IG integration authenticates with IG REST and polls `/markets/{epic}` snapshots.

## Enable automatic IG trading

The engine can submit IG market orders when Policy 5 produces a `BUY` or `SELL` action. Live order submission is opt-in so the default engine remains a monitor/paper-trading mode. To enable automatic trading, provide IG credentials, an epic map, and `--enable-live-trading`:

```bash
export IG_EPIC_MAP='{"AVGO":"YOUR_AVGO_EPIC","MU":"YOUR_MU_EPIC"}'
python portfolio_policy_eval_v12_report.py engine --enable-live-trading
```

Order sizing defaults to the Policy 5 simulated unit amount for the signal. You can override the live IG ticket size per ticker with `IG_DEAL_SIZE_MAP` or `--ig-deal-size-map`:

```bash
export IG_DEAL_SIZE_MAP='{"AVGO":1,"MU":1}'
python portfolio_policy_eval_v12_report.py engine --enable-live-trading --currency-code USD --expiry -
```

When live trading is enabled, each submitted ticket is logged as `LIVE_ORDER_SUBMITTED` with the source Policy 5 action id, ticker, IG epic, direction, size, deal reference, and confirmation status. If a required epic or positive size is missing, the engine logs `LIVE_ORDER_SKIPPED` instead of submitting.


## IG platform test

The repository includes an opt-in IG integration test. It is skipped by default so normal test runs never place real orders. To verify login and Marvell market data without trading:

```bash
export RUN_IG_PLATFORM_TEST=1
export IG_TEST_MARVELL_EPIC="YOUR_MARVELL_IG_EPIC"   # optional; otherwise MRVL in IG_EPIC_MAP or IG market search is used
python -m unittest tests.test_ig_live_integration
```

To intentionally test automatic order placement, add `RUN_IG_LIVE_TRADE_TEST=1`. This submits a real IG market `BUY` for Marvell using `IG_LIVE_TEST_SIZE`, defaulting to `1`, then confirms the deal and checks open positions:

```bash
export RUN_IG_PLATFORM_TEST=1
export RUN_IG_LIVE_TRADE_TEST=1
export IG_TEST_MARVELL_EPIC="YOUR_MARVELL_IG_EPIC"
export IG_LIVE_TEST_SIZE=1
python -m unittest tests.test_ig_live_integration
```

## Run the original report/backtest workflow

```bash
python portfolio_policy_eval_v12_report.py backtest
```

This produces the v12 CSV outputs, plots, and Markdown report used by the original policy evaluator.
