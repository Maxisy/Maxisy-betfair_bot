"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    # Betfair credentials
    betfair_username: str = field(default_factory=lambda: os.environ["BETFAIR_USERNAME"])
    betfair_password: str = field(default_factory=lambda: os.environ["BETFAIR_PASSWORD"])
    betfair_app_key: str = field(default_factory=lambda: os.environ["BETFAIR_APP_KEY"])
    betfair_certs_path: Path = field(
        default_factory=lambda: Path(os.environ.get("BETFAIR_CERTS_PATH", "./certs"))
    )

    # Goalserve
    goalserve_api_key: str = field(default_factory=lambda: os.environ["GOALSERVE_API_KEY"])

    # Alerts
    alert_webhook_url: str = field(
        default_factory=lambda: os.environ.get("ALERT_WEBHOOK_URL", "")
    )

    # Mode
    env: str = field(default_factory=lambda: os.environ.get("ENV", "paper"))

    # --- Trading constants ---
    min_edge: float = 0.06  # 6%
    min_net_profit: float = 0.38  # £0.38
    commission_rate: float = 0.05  # 5%
    min_odds: float = 1.15
    max_odds: float = 4.00
    stop_loss_ticks: int = 4
    target_profit_ticks: int = 3
    max_hold_seconds: int = 60
    edge_gone_threshold: float = 0.02  # 2%
    max_market_exposure: float = 200.0
    max_portfolio_exposure: float = 500.0
    daily_loss_limit: float = 150.0
    daily_loss_half_stake: float = 75.0
    max_trades_per_minute: int = 20
    model_staleness_seconds: int = 30
    new_game_skip_points: int = 2

    # Stake sizing
    stake_pct: float = 0.02  # 2% of float
    phase_max_stake: float = 2.0  # Start at paper-trade level

    # Market filter
    min_matched_volume: float = 5000.0
    min_back_liquidity: float = 200.0
    min_spread_ticks: int = 1
    max_spread_ticks: int = 8

    # Goalserve fallback
    goalserve_fallback_stake_pct: float = 0.50
    goalserve_alert_seconds: int = 60
    goalserve_pause_seconds: int = 300

    # Partial fill timeouts
    entry_fill_timeout: float = 5.0
    exit_fill_timeout: float = 10.0

    # Surface defaults for serve win % — conservative estimates
    # Blended ATP/WTA/Challenger/ITF averages (not top-ATP which is ~65%)
    surface_defaults: dict[str, float] = field(
        default_factory=lambda: {"hard": 0.58, "clay": 0.55, "grass": 0.60}
    )

    # Allowed tournament tiers (keywords to match in tournament name)
    # Per spec: Challenger, ITF, ATP 250/500, WTA 250/500 only
    # Exclude: Grand Slams, Masters 1000, doubles
    allowed_tournaments: list[str] = field(
        default_factory=lambda: [
            # Goalserve inplay feed names
            "ATP ",           # ATP 250/500 (e.g. "ATP Bucharest")
            "WTA ",           # WTA 250/500
            "Challenger",     # Challenger tour
            "M15", "M25",     # ITF Men
            "W15", "W25", "W35", "W50", "W60", "W75", "W100",  # ITF Women
        ]
    )
    excluded_tournaments: list[str] = field(
        default_factory=lambda: [
            # These override allowed_tournaments
            "Grand Slam",
            "Wimbledon",
            "US Open",
            "Australian Open",
            "Roland Garros",
            "French Open",
            "Masters 1000",
            "ATP Masters",
            "Indian Wells",
            "Miami Open",
            "Monte Carlo",
            "Madrid",
            "Rome",
            "Canada",
            "Cincinnati",
            "Shanghai",
            "Paris Masters",
            "WTA Premier Mandatory",
            "WTA Premier 5",
            "WTA 1000",
            " MD",            # Doubles (e.g. "ATP Marrakech MD")
            " WD",            # Women's Doubles
            "Doubles",        # Catch-all for doubles variants
            " XD",            # Mixed Doubles
        ]
    )

    @property
    def is_paper(self) -> bool:
        return self.env == "paper"

    @property
    def cert_file(self) -> Path:
        return self.betfair_certs_path / "client-2048.crt"

    @property
    def key_file(self) -> Path:
        return self.betfair_certs_path / "client-2048.key"
