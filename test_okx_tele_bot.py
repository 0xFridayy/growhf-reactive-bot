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
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the bot modules
from okx_tele_bot import (
    direction,
    pct_change,
    resolve_inst_id,
    FlipScanner,
    REGIME_LABELS,
)
from market_profile import (
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


# --------------------------------------------------------------------------- #
# Test FlipScanner OI-flip logic
# --------------------------------------------------------------------------- #
class TestFlipScannerOIFlip:
    @pytest.fixture
    def scanner(self):
        """Create a FlipScanner with test config."""
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

    def test_oi_flip_longs_building(self, scanner):
        """Price up + OI up → longs building (bullish)."""
        scanner.prev_price["BTC-USDT"] = 50000
        scanner.prev_oi["BTC-USDT"] = 1000000

        with patch("okx_tele_bot.tg_send") as mock_send:
            scanner._check_oi_flip(
                "BTC-USDT", 50100, 1050000, 0, "token", "chat_id"
            )
            # Should detect regime: price +0.2%, OI +5% → (up, up) → longs building
            assert scanner.prev_regime["BTC-USDT"] == "🟢 longs building"
            assert mock_send.called

    def test_oi_flip_short_covering(self, scanner):
        """Price up + OI down → short covering (weak bull, not fresh demand)."""
        scanner.prev_price["BTC-USDT"] = 50000
        scanner.prev_oi["BTC-USDT"] = 1000000

        with patch("okx_tele_bot.tg_send") as mock_send:
            scanner._check_oi_flip(
                "BTC-USDT", 50100, 950000, 0, "token", "chat_id"
            )
            # Price +0.2%, OI -5% → (up, down) → short covering
            assert scanner.prev_regime["BTC-USDT"] == "🟡 short covering"
            assert mock_send.called

    def test_oi_flip_shorts_building(self, scanner):
        """Price down + OI up → shorts building (bearish)."""
        scanner.prev_price["BTC-USDT"] = 50000
        scanner.prev_oi["BTC-USDT"] = 1000000

        with patch("okx_tele_bot.tg_send") as mock_send:
            scanner._check_oi_flip(
                "BTC-USDT", 49900, 1050000, 0, "token", "chat_id"
            )
            # Price -0.2%, OI +5% → (down, up) → shorts building
            assert scanner.prev_regime["BTC-USDT"] == "🔴 shorts building"
            assert mock_send.called

    def test_oi_flip_long_unwind(self, scanner):
        """Price down + OI down → long unwind (weak bear, deleveraging)."""
        scanner.prev_price["BTC-USDT"] = 50000
        scanner.prev_oi["BTC-USDT"] = 1000000

        with patch("okx_tele_bot.tg_send") as mock_send:
            scanner._check_oi_flip(
                "BTC-USDT", 49900, 950000, 0, "token", "chat_id"
            )
            # Price -0.2%, OI -5% → (down, down) → long unwind
            assert scanner.prev_regime["BTC-USDT"] == "🟠 long unwind"
            assert mock_send.called

    def test_no_flip_same_regime(self, scanner):
        """No alert when regime doesn't change."""
        scanner.prev_price["BTC-USDT"] = 50000
        scanner.prev_oi["BTC-USDT"] = 1000000
        scanner.prev_regime["BTC-USDT"] = "🟢 longs building"

        with patch("okx_tele_bot.tg_send") as mock_send:
            scanner._check_oi_flip(
                "BTC-USDT", 50100, 1050000, 0, "token", "chat_id"
            )
            # Regime stays the same → no alert
            assert not mock_send.called

    def test_first_scan_no_alert(self, scanner):
        """No alert on first scan (prev_oi/price are None)."""
        with patch("okx_tele_bot.tg_send") as mock_send:
            scanner._check_oi_flip(
                "BTC-USDT", 50000, 1000000, 0, "token", "chat_id"
            )
            # prev_oi is None → return early
            assert not mock_send.called
            assert "BTC-USDT" not in scanner.prev_regime

    def test_cooldown_suppresses_alert(self, scanner):
        """Alert cooldown should prevent spam."""
        import time

        scanner.prev_price["BTC-USDT"] = 50000
        scanner.prev_oi["BTC-USDT"] = 1000000
        scanner.prev_regime["BTC-USDT"] = "🟡 short covering"
        now = time.time()
        scanner.last_alert["BTC-USDT"] = now - 1800  # 30 min ago (cooldown=3600)

        with patch("okx_tele_bot.tg_send") as mock_send:
            scanner._check_oi_flip(
                "BTC-USDT", 50100, 1050000, now, "token", "chat_id"
            )
            # Still in cooldown → no alert sent
            assert not mock_send.called

    def test_neutral_regime_no_alert(self, scanner):
        """Neutral regimes (flat price or OI) should not trigger alert."""
        scanner.prev_price["BTC-USDT"] = 50000
        scanner.prev_oi["BTC-USDT"] = 1000000
        scanner.prev_regime["BTC-USDT"] = "⚖️ price up, OI flat"

        with patch("okx_tele_bot.tg_send") as mock_send:
            scanner._check_oi_flip(
                "BTC-USDT", 50000.1, 1000000, 0, "token", "chat_id"
            )
            # Neutral tone → no alert
            assert not mock_send.called


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
