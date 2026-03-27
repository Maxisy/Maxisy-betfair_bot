"""Core data models used across the bot."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Side(str, Enum):
    BACK = "BACK"
    LAY = "LAY"


class ExitReason(str, Enum):
    TARGET_REACHED = "target_reached"
    STOP_LOSS = "stop_loss"
    TIME_EXIT = "time_exit"
    EDGE_GONE = "edge_gone"
    MARKET_CLOSE = "market_close"
    RECONCILED = "reconciled"


class ScoreSource(str, Enum):
    API = "api"
    INFERENCE = "inference"


class MarketStatus(str, Enum):
    OPEN = "OPEN"
    SUSPENDED = "SUSPENDED"
    CLOSED = "CLOSED"


@dataclass
class ScoreState:
    match_id: str
    betfair_market_id: str = ""
    player1_name: str = ""
    player2_name: str = ""
    server: str = ""  # "player1" or "player2"
    point_score: tuple[int, int] = (0, 0)  # (server_points, receiver_points) 0-4
    game_score: tuple[int, int] = (0, 0)  # (server_games, receiver_games) current set
    set_score: tuple[int, int] = (0, 0)  # (player1_sets, player2_sets)
    best_of: int = 3
    surface: str = "hard"
    player1_serve_pct: float = 0.63
    player2_serve_pct: float = 0.63
    last_updated: float = 0.0
    source: ScoreSource = ScoreSource.API
    # Selection IDs for Betfair
    player1_selection_id: int = 0
    player2_selection_id: int = 0
    tournament: str = ""
    scheduled_start: float = 0.0
    # Track points played in current game for new-game filter
    points_in_current_game: int = 0

    @property
    def is_fresh(self) -> bool:
        return (time.time() - self.last_updated) < 30.0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.last_updated

    @property
    def server_serve_pct(self) -> float:
        if self.server == "player1":
            return self.player1_serve_pct
        return self.player2_serve_pct

    @property
    def receiver_serve_pct(self) -> float:
        if self.server == "player1":
            return self.player2_serve_pct
        return self.player1_serve_pct

    @property
    def server_selection_id(self) -> int:
        if self.server == "player1":
            return self.player1_selection_id
        return self.player2_selection_id


@dataclass
class MarketState:
    market_id: str = ""
    status: MarketStatus = MarketStatus.OPEN
    in_play: bool = False
    total_matched: float = 0.0
    # selection_id -> runner data
    runners: dict[int, RunnerState] = field(default_factory=dict)
    event_name: str = ""
    tournament: str = ""
    scheduled_start: float = 0.0


@dataclass
class RunnerState:
    selection_id: int = 0
    best_back_price: float = 0.0
    best_back_size: float = 0.0
    best_lay_price: float = 0.0
    best_lay_size: float = 0.0
    last_traded_price: float = 0.0


@dataclass
class Position:
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    market_id: str = ""
    selection_id: int = 0
    side: Side = Side.BACK
    entry_odds: float = 0.0
    stake: float = 0.0
    entry_time: float = field(default_factory=time.time)
    model_odds_at_entry: float = 0.0
    market_odds_at_entry: float = 0.0
    edge_at_entry: float = 0.0
    score_at_entry: str = ""
    model_state_age_at_entry: float = 0.0
    score_source_at_entry: ScoreSource = ScoreSource.API
    event_name: str = ""
    tournament: str = ""
    surface: str = ""
    # Exit tracking
    exit_order_id: Optional[str] = None
    exit_limit_placed_at: Optional[float] = None

    @property
    def hold_seconds(self) -> float:
        return time.time() - self.entry_time


@dataclass
class TradeLog:
    timestamp: str = ""
    trade_id: str = ""
    market_id: str = ""
    selection_id: int = 0
    event_name: str = ""
    tournament: str = ""
    surface: str = ""
    side: str = ""
    entry_odds: float = 0.0
    exit_odds: float = 0.0
    stake: float = 0.0
    gross_profit: float = 0.0
    commission: float = 0.0
    net_profit: float = 0.0
    hold_seconds: float = 0.0
    exit_reason: str = ""
    model_odds: float = 0.0
    market_odds: float = 0.0
    edge_pct: float = 0.0
    score_at_entry: str = ""
    model_state_age_seconds: float = 0.0
    score_source: str = ""
    paper_trade: bool = True
