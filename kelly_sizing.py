"""
kelly_sizing.py — Quarter-Kelly position sizing for the OKX reactive perp screener.

Kelly is a CEILING, not a target -> we use fractional (quarter) Kelly.
Leverage is NOT a free dial -> it falls out of the stop distance.
A hard per-trade risk cap protects a tiny account from correlated dip-buys.

Edge measured from Hyperliquid Growi HF's 6,273 realized closes:
    win rate 76.3%, avg win $205 vs avg loss $222 -> payoff b = 0.92
    full trade-level Kelly f* = (p*b - q)/b = 0.508

Not investment advice. Past vault performance does not guarantee results.
"""

from dataclasses import dataclass

GROWI_WIN_RATE = 0.763
GROWI_PAYOFF_B = 0.92


def full_kelly_fraction(p: float, b: float) -> float:
    """Trade-level Kelly fraction of bankroll to risk. f* = (p*b - q) / b."""
    q = 1.0 - p
    return max(0.0, (p * b - q) / b)


@dataclass
class SizingConfig:
    kelly_fraction: float = 0.25       # quarter-Kelly (estimation + correlation haircut)
    max_risk_per_trade: float = 0.02   # hard cap: never risk >2% of equity per trade
    max_leverage: float = 3.0          # strategy ceiling; the vaults top out ~3.2x
    exchange_max_leverage: float = 50.0
    min_notional_usd: float = 5.0      # OKX perp min order (approx; verify per symbol)
    max_concurrent: int = 3            # cap simultaneous correlated positions


def kelly_risk_fraction(cfg: SizingConfig,
                        p: float = GROWI_WIN_RATE,
                        b: float = GROWI_PAYOFF_B) -> float:
    """Effective per-trade risk fraction: fractional Kelly, capped by max_risk_per_trade."""
    return min(full_kelly_fraction(p, b) * cfg.kelly_fraction, cfg.max_risk_per_trade)


@dataclass
class Position:
    notional_usd: float
    leverage: float
    margin_usd: float
    risk_usd: float          # dollars lost if stop is hit
    contracts: float         # notional / (price * contract_value)
    reason: str              # which constraint bound the size


def size_position(equity_usd: float,
                  entry_price: float,
                  stop_price: float,
                  contract_value: float = 1.0,
                  open_positions: int = 0,
                  cfg: SizingConfig = SizingConfig(),
                  p: float = GROWI_WIN_RATE,
                  b: float = GROWI_PAYOFF_B):
    """
    Quarter-Kelly position for a single signal. Returns Position, or None if rejected
    (too many positions, invalid stop, or below exchange minimum).

    contract_value: USD value per contract per $1 price (1.0 for linear USDT perps
    sized in coin units; adjust for contract-multiplier markets).
    """
    if open_positions >= cfg.max_concurrent:
        return None

    stop_dist_frac = abs(entry_price - stop_price) / entry_price
    if stop_dist_frac <= 0:
        return None  # undefined risk without a real stop

    risk_frac = kelly_risk_fraction(cfg, p, b)
    risk_usd = equity_usd * risk_frac

    notional = risk_usd / stop_dist_frac
    leverage = notional / equity_usd
    reason = "kelly-risk-bound"

    lev_cap = min(cfg.max_leverage, cfg.exchange_max_leverage)
    if leverage > lev_cap:
        leverage = lev_cap
        notional = equity_usd * leverage
        risk_usd = notional * stop_dist_frac
        reason = "leverage-capped"

    if notional < cfg.min_notional_usd:
        return None

    return Position(
        notional_usd=round(notional, 2),
        leverage=round(leverage, 2),
        margin_usd=round(notional / leverage, 2),
        risk_usd=round(risk_usd, 2),
        contracts=round(notional / (entry_price * contract_value), 6),
        reason=reason,
    )
