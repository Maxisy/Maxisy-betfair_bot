# Betfair Tennis Scalping Bot

Automated tennis scalping bot for Betfair Exchange. Monitors live matches using point-by-point score data from Goalserve, maintains a real-time probability model, and trades when market odds diverge from fair value.

The bot never predicts match outcomes. It identifies when the crowd has mispriced odds after a point is played and trades the reversion. In and out within seconds to minutes.

## How It Works

1. **Goalserve** provides live point-by-point scores every 5 seconds
2. A **four-layer Markov chain** (point > game > set > match) calculates the true win probability
3. **Betfair Streaming API** delivers real-time odds
4. When market odds diverge from model odds by >6%, the bot places a trade
5. Exits on: 3-tick profit target, 4-tick stop loss, 60-second timeout, or edge convergence

## Project Structure

```
betfair_bot/
├── pyproject.toml
├── .env.example
├── certs/                  # Betfair SSL certificates (gitignored)
├── data/                   # Serve stats cache
├── logs/                   # Daily summaries
└── bot/
    ├── config.py           # Central configuration
    ├── ticks.py            # Betfair tick ladder and price utilities
    ├── models.py           # Core data models (ScoreState, Position, etc.)
    ├── probability.py      # Four-layer Markov chain probability model
    ├── betfair_client.py   # Betfair APING REST client
    ├── stream.py           # Betfair Streaming API (SSL WebSocket)
    ├── goalserve.py        # Goalserve score poller
    ├── matcher.py          # Goalserve-to-Betfair match mapping
    ├── serve_stats.py      # Tennis Abstract serve % scraper
    ├── market_filter.py    # Market qualification filter
    ├── risk.py             # Risk manager and kill switch
    ├── positions.py        # Order lifecycle and position tracking
    ├── trading.py          # Core trading engine
    ├── logger.py           # Trade logger (trades.jsonl)
    ├── alerts.py           # Discord/Telegram webhook alerts
    └── main.py             # Asyncio orchestrator and entry point
```

## Prerequisites

| Item | Details |
|---|---|
| **Python** | 3.11+ |
| **Betfair account** | Register at betfair.com, verify with ID |
| **Betfair App Key** | Get at developer.betfair.com |
| **Betfair SSL Certificate** | Self-signed, uploaded to your Betfair account |
| **Goalserve API Key** | Tennis Package, $150/month at goalserve.com |
| **Discord/Telegram webhook** | Optional, for alerts |

## Setup

### 1. Install dependencies

```bash
pip install .
```

### 2. Generate Betfair SSL certificates

```bash
cd certs
openssl genrsa -out client-2048.key 2048
openssl req -new -x509 -days 3650 -key client-2048.key -out client-2048.crt
```

Upload `client-2048.crt` to your Betfair account at My Account > My Security > Automated Betting Program Access.

### 3. Configure environment

```bash
cp .env.example .env
```

Fill in `.env`:

```
BETFAIR_USERNAME=your_username
BETFAIR_PASSWORD=your_password
BETFAIR_APP_KEY=your_app_key
BETFAIR_CERTS_PATH=./certs
GOALSERVE_API_KEY=your_goalserve_key
ALERT_WEBHOOK_URL=https://discord.com/api/webhooks/...
ENV=paper
```

### 4. Run

```bash
python3 -m bot.main
```

## Trading Parameters

| Parameter | Value |
|---|---|
| Minimum edge | 6% |
| Target profit | 3 ticks |
| Stop loss | 4 ticks |
| Max hold time | 60 seconds |
| Odds range | 1.15 - 4.00 |
| Max market exposure | £200 |
| Max portfolio exposure | £500 |
| Daily loss limit | £150 (kill switch) |
| Commission | 5% on net winnings |
| Min net profit per trade | £0.38 |

## Target Markets

**Included:** ATP Challenger, WTA International, ITF, ATP/WTA 250, ATP/WTA 500

**Excluded:** Grand Slams, Masters 1000 (too efficient, too many professional traders)

## Paper Trading

Set `ENV=paper` in `.env`. All logic runs identically to live mode but no real orders are placed. Fills are simulated at signal price. Every trade is logged to `trades.jsonl`.

Run paper mode for a minimum of 48 hours before deploying live capital.

## Stake Phases

Only increase `phase_max_stake` in `config.py` after two consecutive profitable weeks at the current level:

| Phase | Max Stake |
|---|---|
| Paper trading | £2 (simulated) |
| Week 2 (micro-stakes) | £10 |
| Week 3-4 | £20 |
| Month 2+ | £50 |

## Risk Controls

- **Kill switch** — triggers on daily loss limit (£150), closes all positions, stops bot
- **Stake reduction** — 50% stakes when daily loss hits £75 or win rate drops below 40%
- **Rate limiting** — max 20 trades per minute
- **One position per market** — no pyramiding
- **LAPSE persistence** — all unmatched orders cancel automatically
- **Position reconciliation** — on every startup, fetches open orders and reconciles before trading

## Logging

Every trade is written to `trades.jsonl` with full details: entry/exit odds, model odds, edge, score state, hold time, exit reason, P&L.

Daily summaries are saved to `logs/daily_YYYY-MM-DD.json` at 00:00 UTC.

## Alerts

Sent via Discord or Telegram webhook:

- Bot started/stopped
- Stream disconnect/reconnect
- Daily loss limit hit
- Large single trade loss (>£30)
- Goalserve down >60 seconds
- Win rate below 40%
- Daily summary at midnight UTC

## Deployment

Recommended: London-based VPS (Hetzner CPX11 ~£4/month or AWS t3.small eu-west-2) to minimise latency to Betfair's Slough data centre.

Monthly running costs: ~£130-160 (Goalserve $150 + VPS £4-8).

## Non-Negotiable Rules

1. Never trade where expected net profit after commission is below £0.38
2. Never trade outside odds 1.15 - 4.00
3. Never trade with stale model state (>30 seconds old)
4. Never trade in the first 2 points of a new game
5. Never target Grand Slams or Masters 1000
6. Every signal passes through the risk manager
7. Always use LAPSE persistence on every order
8. One open position per market maximum
9. Paper trade minimum 48 hours before live capital
10. Log every trade without exception
