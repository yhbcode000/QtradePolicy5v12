# Qtrade_Policy_5v12

## Trading system startup flow

When the trading engine starts, it should initialize and decide in this order:

1. Load `.env` so runtime configuration, credentials, position source, sizing, and live-trading safety flags are available before any market or broker action.
2. Fetch the latest 60 days of Yahoo 15m data to build the current market window used by the strategy.
3. Authenticate to IG if IG integration is enabled.
4. Read the current IG open positions before making a trading decision.
5. Compute the current notional exposure and target contract counts.
6. Compute the current momentum / Policy 5 decision.
7. Submit a live order only if `ENGINE_ENABLE_LIVE_TRADING=1`.

> **Warning:** Keep live trading disabled until the dashboard's current-position table has been verified against the broker account. Do not set `ENGINE_ENABLE_LIVE_TRADING=1` until the displayed open positions, notionals, and contract counts are confirmed to be correct.

### Starting from zero positions

To intentionally start from no existing positions, configure the engine to use a zero-position source and provide the starting capital plus either target contract ratios or a target allocation plan in `.env`:

```dotenv
ENGINE_POSITION_SOURCE=zero
ENGINE_STARTING_EQUITY=...

# Provide target contract ratios, for example:
ENGINE_TARGET_CONTRACT_RATIOS=...

# Or provide a target plan, for example:
ENGINE_TARGET_PLAN=...
```

Use this mode only when the account should be treated as flat. If the account already has IG open positions, use the IG/current-position source instead so the engine can read those positions before calculating target contract counts and any Policy 5 order decision.
