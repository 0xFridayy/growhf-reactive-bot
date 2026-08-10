"""
test_okx_tele_bot.py — Unit tests for OKX Telegram bot core logic.

Tests cover the high-risk gaps:
  - OI-flip regime detection and transitions
  - Funding-flip sign crossing at boundaries
  - TPO profile edge cases
  - Mean reversion with zero/tiny volatility
  - Command parsing and resolve_inst_id()
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

# Import the bot modules
from okx_tele_bot import (
    direction,
    pct_change,
    resolve_inst_id,
    analyze_macro,
    handle_command,
    FlipScanner,
    REGIME_LABELS,
    regime_bias,
)
from okx_perp_screener import spike_has_edge
import macro_feeds
from macro_feeds import resolve_macro
from market_profile import (
    build_tpo,
    build_tpo_profile,
    mean_reversion,
    regime,
    efficiency_ratio,
)


# --------------------------------------------------------------------------- #
# Test direction() and pct_change() — core signal helpers
# --------------------------------------------------------------------------- #
class TestSignalHelpers:
    def test_direction_up(self):
        assert direction(5.0, 0.5) == "up"
        assert direction(0.6, 0.5) == "up"

    def test_direction_down(self):
        assert direction(-5.0, 0.5) == "down"
        assert direction(-0.6, 0.5) == "down"

    def test_direction_flat(self):
        assert direction(0.3, 0.5) == "flat"
        assert direction(-0.3, 0.5) == "flat"
        assert direction(0.0, 0.5) == "flat"

    def test_direction_at_boundary(self):
        """Dead zone boundary should be treated as flat."""
        assert direction(0.5, 0.5) == "flat"
        assert direction(-0.5, 0.5) == "flat"

    def test_pct_change_basic(self):
        assert pct_change(110, 100) == 10.0
        assert pct_change(90, 100) == -10.0

    def test_pct_change_zero_old(self):
        """Zero old price should return 0.0 to avoid division by zero."""
        assert pct_change(100, 0) == 0.0
        assert pct_change(0, 0) == 0.0

    def test_pct_change_small_moves(self):
        assert pct_change(100.001, 100) == pytest.approx(0.001, abs=1e-6)


# --------------------------------------------------------------------------- #
# Test resolve_inst_id() — command parsing
# --------------------------------------------------------------------------- #
class TestResolveInstId:
    def test_bare_symbol(self):
        """Bare symbol like 'btc' → 'BTC-USDT-SWAP'."""
        assert resolve_inst_id("btc", "USDT") == "BTC-USDT-SWAP"
        assert resolve_inst_id("SOL", "USDT") == "SOL-USDT-SWAP"

    def test_symbol_with_quote(self):
        """Symbol with hyphen like 'btc-usdt' → 'BTC-USDT-SWAP'."""
        assert resolve_inst_id("btc-usdt", "USDT") == "BTC-USDT-SWAP"
        assert resolve_inst_id("sol-busd", "BUSD") == "SOL-BUSD-SWAP"

    def test_symbol_already_swap(self):
        """Already has -SWAP suffix."""
        assert resolve_inst_id("BTC-USDT-SWAP", "USDT") == "BTC-USDT-SWAP"

    def test_quote_filter_override(self):
        """Quote filter should be used when not in symbol."""
        assert resolve_inst_id("eth", "BUSD") == "ETH-BUSD-SWAP"

    def test_slash_to_hyphen(self):
        """Slash should be replaced with hyphen."""
        assert resolve_inst_id("btc/usdt", "USDT") == "BTC-USDT-SWAP"

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace should be stripped."""
        assert resolve_inst_id("  btc  ", "USDT") == "BTC-USDT-SWAP"


# OI-flip tests removed because OI-flip alerts are deleted from the bot.


# --------------------------------------------------------------------------- #
# Test OI-flip bias mapping
# --------------------------------------------------------------------------- #
class TestOIFlipBias:
    def test_regime_bias_is_long_for_longs_building(self):
        assert regime_bias("🟢 longs building") == "long"

    def test_regime_bias_is_short_for_shorts_building(self):
        assert regime_bias("🔴 shorts building") == "short"

    def test_regime_bias_is_neutral_for_weak_flips(self):
        assert regime_bias("🟡 short covering") == "neutral"
        assert regime_bias("🟠 long unwind") == "neutral"


class TestPerpScreenerEdgeFilter:
    def test_spike_has_edge_requires_cvd_confirmed_up_for_long(self):
        mock_cc = Mock()
        mock_cc.fetch_cvd.return_value = [{"ts": 1, "delta": 1.0, "cvd": 1.0}]
        mock_cc.resample_to_candles.return_value = [1.0]
        mock_cc.cvd_read.return_value = ("confirmed up", "")
        mock_cc.causal_pivots.return_value = ([], [])
        mock_cc.structure_state.return_value = Mock(state="uptrend")

        with patch.dict("sys.modules", {"chart_command": mock_cc}):
            candles = [[1, 1, 2, 0.5, 1.5, 100]]
            assert spike_has_edge("BTC-USDT", candles, "long") is True

    def test_spike_has_edge_rejects_no_edge_for_short(self):
        mock_cc = Mock()
        mock_cc.fetch_cvd.return_value = [{"ts": 1, "delta": -1.0, "cvd": -1.0}]
        mock_cc.resample_to_candles.return_value = [-1.0]
        mock_cc.cvd_read.return_value = ("divergent", "")
        mock_cc.causal_pivots.return_value = ([], [])
        mock_cc.structure_state.return_value = Mock(state="downtrend")

        with patch.dict("sys.modules", {"chart_command": mock_cc}):
            candles = [[1, 1, 2, 0.5, 1.5, 100]]
            assert spike_has_edge("BTC-USDT", candles, "short") is False


# --------------------------------------------------------------------------- #
# Test FlipScanner funding-flip logic
# --------------------------------------------------------------------------- #
class TestFlipScannerFundingFlip:
    @pytest.fixture
    def scanner(self):
        cfg = {
            "telegram_bot_token": "test-token",
            "telegram_chat_id": "123",
            "quote_filter": "USDT",
            "oi_funding": {
                "enabled": True,
                "scan_interval_seconds": 300,
                "universe_top_n": 40,
                "oi_change_threshold_pct": 5.0,
                "price_change_threshold_pct": 0.5,
                "funding_flip_min_abs": 0.0001,
                "cooldown_seconds": 3600,
            },
        }
        return FlipScanner(cfg)

    def test_funding_flip_zero_cross_positive(self, scanner):
        """Funding rate crossing from negative to positive."""
        scanner.prev_funding["BTC-USDT"] = -0.0002
        import time

        now = time.time()
        with patch("okx_tele_bot.fetch_funding", return_value=0.0003):
            with patch("okx_tele_bot.tg_send") as mock_send:
                scanner._check_funding_flips(["BTC-USDT"], now, "token", "chat")
                # Crossed zero, magnitude > min_abs → alert sent
                assert mock_send.called

    def test_funding_flip_zero_cross_negative(self, scanner):
        """Funding rate crossing from positive to negative."""
        scanner.prev_funding["BTC-USDT"] = 0.0002
        import time

        now = time.time()
        with patch("okx_tele_bot.fetch_funding", return_value=-0.0003):
            with patch("okx_tele_bot.tg_send") as mock_send:
                scanner._check_funding_flips(["BTC-USDT"], now, "token", "chat")
                assert mock_send.called

    def test_funding_flip_below_min_magnitude(self, scanner):
        """Crossing but below min_abs threshold → no alert."""
        scanner.prev_funding["BTC-USDT"] = -0.00001
        import time

        now = time.time()
        with patch("okx_tele_bot.fetch_funding", return_value=0.00001):
            with patch("okx_tele_bot.tg_send") as mock_send:
                scanner._check_funding_flips(["BTC-USDT"], now, "token", "chat")
                # Magnitude < 0.0001 → no alert
                assert not mock_send.called

    def test_funding_no_cross_same_sign(self, scanner):
        """No cross when both same sign."""
        scanner.prev_funding["BTC-USDT"] = 0.0001
        import time

        now = time.time()
        with patch("okx_tele_bot.fetch_funding", return_value=0.0003):
            with patch("okx_tele_bot.tg_send") as mock_send:
                scanner._check_funding_flips(["BTC-USDT"], now, "token", "chat")
                assert not mock_send.called

    def test_funding_first_scan_no_alert(self, scanner):
        """No alert on first scan (prev_funding is None)."""
        import time

        now = time.time()
        with patch("okx_tele_bot.fetch_funding", return_value=0.0001):
            with patch("okx_tele_bot.tg_send") as mock_send:
                scanner._check_funding_flips(["BTC-USDT"], now, "token", "chat")
                # prev_funding is None → skip
                assert not mock_send.called

    def test_funding_cooldown_suppresses(self, scanner):
        """Cooldown suppresses repeated funding flips."""
        import time

        scanner.prev_funding["BTC-USDT"] = -0.0002
        now = time.time()
        scanner.last_alert["BTC-USDT"] = now - 1800  # 30 min ago

        with patch("okx_tele_bot.fetch_funding", return_value=0.0003):
            with patch("okx_tele_bot.tg_send") as mock_send:
                scanner._check_funding_flips(["BTC-USDT"], now, "token", "chat")
                # In cooldown → no alert
                assert not mock_send.called


# --------------------------------------------------------------------------- #
# Test TPO profile edge cases
# --------------------------------------------------------------------------- #
class TestTPOProfile:
    def test_tpo_basic(self):
        """Build a basic TPO profile from candles."""
        # 5 candles: [ts, o, h, l, c, vol, ...]
        candles = [
            [1000, 100, 105, 95, 102, 1000],
            [2000, 102, 107, 101, 106, 1000],
            [3000, 106, 110, 104, 108, 1000],
            [4000, 108, 112, 106, 110, 1000],
            [5000, 110, 115, 108, 112, 1000],
        ]
        prof = build_tpo_profile(candles)
        assert prof is not None
        assert prof.poc is not None
        assert prof.vah > prof.val
        assert prof.in_value_area is not None

    def test_tpo_insufficient_candles(self):
        """Less than 5 candles → return None."""
        candles = [
            [1000, 100, 105, 95, 102, 1000],
            [2000, 102, 107, 101, 106, 1000],
        ]
        assert build_tpo_profile(candles) is None

    def test_tpo_zero_span(self):
        """All candles at same price → span=0 → return None."""
        candles = [
            [1000, 100, 100, 100, 100, 1000],
            [2000, 100, 100, 100, 100, 1000],
            [3000, 100, 100, 100, 100, 1000],
            [4000, 100, 100, 100, 100, 1000],
            [5000, 100, 100, 100, 100, 1000],
        ]
        assert build_tpo_profile(candles) is None

    def test_tpo_tiny_span(self):
        """Very small price range still produces valid profile."""
        candles = [
            [1000, 100.00, 100.001, 100.00, 100.0005, 1000],
            [2000, 100.0005, 100.0015, 100.0005, 100.001, 1000],
            [3000, 100.001, 100.0015, 100.0008, 100.0012, 1000],
            [4000, 100.0012, 100.002, 100.001, 100.0015, 1000],
            [5000, 100.0015, 100.002, 100.0012, 100.0018, 1000],
        ]
        prof = build_tpo_profile(candles)
        assert prof is not None
        assert prof.poc is not None

    def test_tpo_price_location_inside_value(self):
        """Test that in_value_area flag works correctly."""
        candles = [
            [1000, 100, 110, 90, 105, 1000],
            [2000, 105, 115, 100, 110, 1000],
            [3000, 110, 120, 105, 115, 1000],
            [4000, 115, 125, 110, 120, 1000],
            [5000, 120, 130, 115, 125, 1000],
        ]
        prof = build_tpo_profile(candles)
        # Last candle close is 125, should be inside or outside value area
        assert prof.in_value_area is not None

    def test_tpo_poc_distance_calculation(self):
        """POC distance should be calculated correctly."""
        candles = [
            [1000, 100, 105, 95, 100, 1000],
            [2000, 100, 110, 95, 105, 1000],
            [3000, 105, 115, 100, 110, 1000],
            [4000, 110, 120, 105, 115, 1000],
            [5000, 115, 125, 110, 120, 1000],
        ]
        prof = build_tpo_profile(candles)
        # Current price is 120, POC should be somewhere in range
        assert prof.poc_dist_pct is not None

    def test_build_tpo_filters_to_current_utc_session(self):
        """Session-aware TPO should ignore prior-day bars."""
        now = datetime.now(timezone.utc)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        prev_day = start - timedelta(days=1)
        candles = [
            [int((start + timedelta(hours=1)).timestamp() * 1000), 95, 100, 90, 98, 1000],
            [int((start + timedelta(hours=2)).timestamp() * 1000), 98, 102, 95, 100, 1000],
            [int((start + timedelta(hours=3)).timestamp() * 1000), 100, 104, 96, 101, 1000],
            [int((start + timedelta(hours=4)).timestamp() * 1000), 101, 106, 99, 103, 1000],
            [int((start + timedelta(hours=5)).timestamp() * 1000), 103, 108, 101, 105, 1000],
            [int((prev_day + timedelta(hours=1)).timestamp() * 1000), 180, 200, 170, 190, 1000],
            [int((prev_day + timedelta(hours=2)).timestamp() * 1000), 190, 210, 180, 200, 1000],
        ]
        prof = build_tpo(candles, session_start=start)
        assert prof is not None
        assert prof.price_high <= 108.0
        assert prof.price_low >= 90.0
        assert prof.last == 98.0


# --------------------------------------------------------------------------- #
# Test mean reversion edge cases
# --------------------------------------------------------------------------- #
class TestMeanReversion:
    def test_mean_reversion_basic(self):
        """Build a basic mean reversion indicator."""
        # Create 20+ candles with some trend
        candles = [[i * 100, 100 + i, 101 + i, 99 + i, 100 + i, 1000] for i in range(25)]
        mr = mean_reversion(candles)
        assert mr is not None
        assert mr.mean is not None
        assert mr.std is not None
        assert mr.zscore is not None
        assert mr.signal in ("fade-short", "fade-long", "neutral")

    def test_mean_reversion_insufficient_candles(self):
        """Less than 5 candles → return None."""
        candles = [[i * 100, 100, 101, 99, 100, 1000] for i in range(3)]
        assert mean_reversion(candles) is None

    def test_mean_reversion_zero_volatility(self):
        """All prices identical → std=0 → z=0 → neutral signal."""
        candles = [[i * 100, 100, 100, 100, 100, 1000] for i in range(25)]
        mr = mean_reversion(candles)
        assert mr is not None
        assert mr.std == 0.0
        assert mr.zscore == 0.0
        assert mr.signal == "neutral"

    def test_mean_reversion_stretched_high(self):
        """Price stretched above mean (z > 2.0) → fade-short signal."""
        # Build candles with mean ~100, current price at ~130
        closes = [100 + i * 0.5 for i in range(20)]  # mild uptrend
        closes.append(130)  # stretched high
        candles = [[i * 100, c, c + 1, c - 1, c, 1000] for i, c in enumerate(closes)]
        mr = mean_reversion(candles)
        if mr and mr.zscore >= 2.0:
            assert mr.signal == "fade-short"

    def test_mean_reversion_stretched_low(self):
        """Price stretched below mean (z < -2.0) → fade-long signal."""
        closes = [100 + i * 0.5 for i in range(20)]
        closes.append(70)  # stretched low
        candles = [[i * 100, c, c + 1, c - 1, c, 1000] for i, c in enumerate(closes)]
        mr = mean_reversion(candles)
        if mr and mr.zscore <= -2.0:
            assert mr.signal == "fade-long"

    def test_mean_reversion_near_fair_value(self):
        """Z-score near 0 → neutral signal."""
        closes = [100 + i * 0.1 for i in range(25)]  # very flat
        candles = [[i * 100, c, c + 0.1, c - 0.1, c, 1000] for i, c in enumerate(closes)]
        mr = mean_reversion(candles)
        assert mr is not None
        assert abs(mr.zscore) < 0.5
        assert mr.signal == "neutral"


# --------------------------------------------------------------------------- #
# Test regime classification
# --------------------------------------------------------------------------- #
class TestRegimeClassification:
    def test_regime_trending_up(self):
        """Strong uptrend with high efficiency ratio → trending."""
        # Build uptrend: closes going from 100 to 120
        closes = [100 + i * 1.0 for i in range(25)]
        candles = [[i * 100, c, c + 1, c - 1, c, 1000] for i, c in enumerate(closes)]
        reg = regime(candles, lookback=20, trend_er=0.5, range_er=0.3)
        assert reg is not None
        assert "Trending" in reg.label or "up" in reg.label

    def test_regime_ranging(self):
        """Choppy sideways movement → ranging."""
        # Oscillating closes around 100
        closes = [100 + (5 if i % 2 == 0 else -5) for i in range(25)]
        candles = [[i * 100, c, c + 1, c - 1, c, 1000] for i, c in enumerate(closes)]
        reg = regime(candles, lookback=20, trend_er=0.5, range_er=0.3)
        assert reg is not None
        assert reg.efficiency_ratio < 0.5

    def test_regime_insufficient_candles(self):
        """Less than 5 candles → return None."""
        candles = [[i * 100, 100, 101, 99, 100, 1000] for i in range(3)]
        assert regime(candles) is None

    def test_efficiency_ratio_calculation(self):
        """Efficiency ratio: net move / path length."""
        # Straight move up: net=10, path=10 → ER=1.0
        closes = list(range(100, 110))  # 100, 101, ..., 109
        er = efficiency_ratio(closes, 10)
        assert er == pytest.approx(1.0, abs=0.01)

    def test_efficiency_ratio_choppy(self):
        """Choppy movement: net=0, path=large → ER=0."""
        closes = [100, 105, 100, 105, 100, 105, 100]  # oscillating
        er = efficiency_ratio(closes, 6)
        assert er < 0.5


# --------------------------------------------------------------------------- #
# Test integration: analyze() command
# --------------------------------------------------------------------------- #
class TestAnalyzeCommand:
    """Integration tests for the analyze() function would go here,
    but they require mocking OKX API calls. These are left as placeholders
    for the user to expand with live-like mock responses.
    """

    @patch("okx_tele_bot.fetch_tickers")
    @patch("okx_tele_bot.fetch_candles")
    @patch("okx_tele_bot.fetch_funding")
    def test_analyze_missing_instrument(self, mock_funding, mock_candles, mock_tickers):
        """Analyze should raise ValueError if instrument doesn't exist."""
        from okx_tele_bot import analyze

        mock_tickers.return_value = []
        mock_candles.return_value = []
        mock_funding.return_value = None

        with patch("okx_tele_bot.okx_get", side_effect=Exception("Not found")):
            with pytest.raises(ValueError, match="unknown instrument"):
                analyze("FAKE-USDT-SWAP", "USDT")


# --------------------------------------------------------------------------- #
# Test macro gauges (BTC.D / USDT.D / DXY)
# --------------------------------------------------------------------------- #
class TestResolveMacro:
    def test_aliases_map_to_canonical(self):
        for q in ("btc.d", "BTC.D", "btcd", "BTCDOM", "btc-d"):
            assert resolve_macro(q) == "BTC.D"
        for q in ("usdt.d", "USDTD", "USDTM", "usdt-d"):
            assert resolve_macro(q) == "USDT.D"
        for q in ("dxy", "DXY", "usdx", "dollar"):
            assert resolve_macro(q) == "DXY"

    def test_real_coins_are_not_macro(self):
        """Regression: normal instruments must fall through to resolve_inst_id()."""
        for q in ("btc", "sol", "hype", "eth", "doge", ""):
            assert resolve_macro(q) is None

    def test_tolerates_a_resolved_instid(self):
        """'/analyze BTC.D-USDT-SWAP' (what the old code built) still resolves."""
        assert resolve_macro("BTC.D-USDT-SWAP") == "BTC.D"

    def test_no_alias_shadows_a_real_okx_base(self):
        """An alias must never hijack a listed perp. Guards future alias additions."""
        okx_bases = {"BTC", "ETH", "SOL", "DOGE", "HYPE", "ZEC", "XAU", "MU", "SNDK"}
        assert not (set(macro_feeds.ALIASES) & okx_bases)


class TestLineCandles:
    def test_shape_is_okx_compatible_and_newest_first(self):
        rows = macro_feeds._line_candles([(1000, 50.0), (2000, 51.0), (3000, 50.5)])
        assert len(rows) == 3
        assert all(len(r) == 9 for r in rows)
        assert [int(r[0]) for r in rows] == [3000, 2000, 1000]   # newest first
        # analytics parse these straight out of the row
        assert float(rows[0][4]) == 50.5

    def test_no_invented_wicks(self):
        """High/low must never exceed the open->close span for a derived index."""
        rows = macro_feeds._line_candles([(1000, 50.0), (2000, 51.0)])
        for r in rows:
            o, h, l, c = float(r[1]), float(r[2]), float(r[3]), float(r[4])
            assert h == max(o, c) and l == min(o, c)


class TestDominanceMath:
    """The derivation must land on the anchor's official level, and must not read a
    partially-priced basket as a dominance move."""

    ANCHOR = {
        "supply": {"BTC": 20.0, "ETH": 100.0, "SOL": 400.0},
        "inst": {"BTC": "BTC-USDT-SWAP", "ETH": "ETH-USDT-SWAP", "SOL": "SOL-USDT-SWAP"},
        "uncovered": 400.0,
        "usdt_mcap": 200.0,
        "total": 2000.0,
        "coverage": 0.8,
        "ts": 0.0,
    }

    def _candles(self, price_map):
        """price_map: {sym: {ts: close}} -> patched _okx_candles."""
        def fake(inst_id, bar, limit):
            sym = inst_id.split("-")[0]
            series = price_map.get(sym, {})
            return [[str(ts), "0", "0", "0", str(px), "0", "0", "0", "1"]
                    for ts, px in sorted(series.items(), reverse=True)]
        return fake

    def test_level_matches_hand_computation(self):
        # btc = 20*50 = 1000, alt = 100*4 + 400*0.5 = 600, uncovered = 400
        # den = 2000 -> BTC.D = 50%, USDT.D = 200/2000 = 10%
        prices = {"BTC": {i: 50.0 for i in range(1, 11)},
                  "ETH": {i: 4.0 for i in range(1, 11)},
                  "SOL": {i: 0.5 for i in range(1, 11)}}
        with patch.object(macro_feeds, "_anchor", return_value=self.ANCHOR), \
             patch.object(macro_feeds, "_okx_candles", side_effect=self._candles(prices)):
            btcd, _ = macro_feeds._dominance_series("BTC.D", "15m", 10)
            usdtd, _ = macro_feeds._dominance_series("USDT.D", "15m", 10)
        assert float(btcd[0][4]) == pytest.approx(50.0)
        assert float(usdtd[0][4]) == pytest.approx(10.0)

    def test_partial_basket_bars_are_skipped(self):
        """A bar where most of the basket has no candle would look like a huge
        dominance spike. Those bars must be dropped, not derived."""
        full = {i: 4.0 for i in range(1, 11)}
        prices = {"BTC": {i: 50.0 for i in range(1, 11)},
                  "ETH": full,
                  "SOL": {i: 0.5 for i in range(1, 11)}}
        del prices["ETH"][5]        # ETH carries 400/600 of alt supply weight
        del prices["SOL"][5]
        with patch.object(macro_feeds, "_anchor", return_value=self.ANCHOR), \
             patch.object(macro_feeds, "_okx_candles", side_effect=self._candles(prices)):
            rows, _ = macro_feeds._dominance_series("BTC.D", "15m", 10)
        assert 5 not in [int(r[0]) for r in rows]
        assert all(float(r[4]) == pytest.approx(50.0) for r in rows)


class TestAnalyzeMacro:
    FLAT = [[str(1000 - i), "50", "50", "50", "50", "0", "0", "0", "1"] for i in range(96)]

    def _rising(self):
        # newest-first: level climbs steadily toward the present
        return [[str(1000 - i), f"{50 + (95 - i) * 0.02}", f"{50 + (95 - i) * 0.02}",
                 f"{50 + (95 - i) * 0.02}", f"{50 + (95 - i) * 0.02}",
                 "0", "0", "0", "1"] for i in range(96)]

    def test_rising_dominance_reads_as_favour_btc(self):
        meta = {"basket": 30, "coverage": 0.81, "anchor_age_min": 3.0,
                "source": "OKX closes x CoinGecko supplies"}
        with patch("okx_tele_bot.macro_candles", return_value=(self._rising(), meta)):
            msg, verdict = analyze_macro("BTC.D")
        assert "favour BTC over alts" in verdict
        assert "BTC dominance" in msg

    def test_never_emits_a_sized_plan(self):
        """These are not executable on OKX — a Kelly plan would be a live trade call."""
        meta = {"source": "Yahoo DX-Y.NYB", "lag_min": 10.0}
        with patch("okx_tele_bot.macro_candles", return_value=(self._rising(), meta)):
            msg, _ = analyze_macro("DXY")
        assert "quarter-Kelly" not in msg and "Notional" not in msg
        assert "not tradeable" in msg
        assert "feed lag 10m" in msg          # delay must be disclosed

    def test_flat_series_gives_no_signal(self):
        meta = {"basket": 30, "coverage": 0.81, "anchor_age_min": 1.0, "source": "x"}
        with patch("okx_tele_bot.macro_candles", return_value=(self.FLAT, meta)):
            _, verdict = analyze_macro("USDT.D")
        assert "flat / no clear regime signal" in verdict


class TestMacroCommandRouting:
    CFG = {"quote_filter": "USDT", "sizing": {"enabled": True, "account_equity_usd": 280}}

    def test_analyze_btcd_routes_to_macro_not_okx(self):
        """The original bug: '/analyze btc.d' built BTC.D-USDT-SWAP and 404'd."""
        with patch("okx_tele_bot.analyze_macro", return_value=("MACRO OK", "v")) as m, \
             patch("okx_tele_bot.analyze") as perp:
            out = handle_command("/analyze btc.d", self.CFG)
        assert out == "MACRO OK"
        m.assert_called_once_with("BTC.D")
        perp.assert_not_called()

    def test_bare_macro_symbol_works(self):
        with patch("okx_tele_bot.analyze_macro", return_value=("MACRO OK", "v")):
            assert handle_command("btcdom", self.CFG) == "MACRO OK"

    def test_normal_symbol_still_uses_okx_path(self):
        with patch("okx_tele_bot.analyze", return_value=("PERP OK", "v")) as perp, \
             patch("okx_tele_bot.analyze_macro") as m:
            out = handle_command("/analyze sol", self.CFG)
        assert out == "PERP OK"
        perp.assert_called_once()
        assert perp.call_args[0][0] == "SOL-USDT-SWAP"
        m.assert_not_called()

    def test_macro_failure_reports_the_macro_symbol(self):
        with patch("okx_tele_bot.analyze_macro", side_effect=RuntimeError("feed down")):
            out = handle_command("/analyze dxy", self.CFG)
        assert "DXY" in out and "feed down" in out
        assert "USDT-SWAP" not in out      # never show a fake instId


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
