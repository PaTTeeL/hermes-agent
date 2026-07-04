"""Tests for 429 per-model rate-limit TTL core functionality."""

import pytest
from unittest.mock import patch

from agent.credential_pool import CredentialPool, PooledCredential, RateLimitEntry


class TestPerModelRateLimitTTL:
    """Test 429 per-model TTL escalation, model isolation, and expiry recovery."""

    @pytest.fixture
    def pool_with_entry(self):
        """Create a pool with a single entry, with _persist mocked out."""
        entry = PooledCredential(
            id="test1",
            provider="test",
            label="Test Key",
            auth_type="api_key",
            priority=0,
            source="manual",
            access_token="",
        )
        with patch.object(CredentialPool, "_persist", return_value=None):
            pool = CredentialPool("test", [entry])
        return pool, entry

    @patch("agent.credential_pool.time.monotonic")
    def test_ttl_escalation(self, mock_monotonic, pool_with_entry):
        """Consecutive 429s escalate TTL: 5min -> 10min -> 15min -> ... (1h cap)."""
        pool, entry = pool_with_entry

        # 1st 429: now=1000, TTL = 300s (5min), reset_at = 1300
        mock_monotonic.return_value = 1000.0
        updated = pool.mark_rate_limited(entry, "gpt-4", 429)
        assert updated.rate_limited["gpt-4"].consecutive_count == 1
        assert abs(updated.rate_limited["gpt-4"].reset_at - 1300.0) < 1.0

        # 2nd 429: now=1001, TTL = 300 + 300*1 = 600s (10min), reset_at = 1601
        mock_monotonic.return_value = 1001.0
        updated = pool.mark_rate_limited(updated, "gpt-4", 429)
        assert updated.rate_limited["gpt-4"].consecutive_count == 2
        assert abs(updated.rate_limited["gpt-4"].reset_at - 1601.0) < 1.0

        # 3rd 429: now=1002, TTL = 300 + 300*2 = 900s (15min), reset_at = 1902
        mock_monotonic.return_value = 1002.0
        updated = pool.mark_rate_limited(updated, "gpt-4", 429)
        assert updated.rate_limited["gpt-4"].consecutive_count == 3
        assert abs(updated.rate_limited["gpt-4"].reset_at - 1902.0) < 1.0

        # 4th 429: now=1003, TTL = 300 + 300*3 = 1200s (below 1h cap), reset_at = 2203
        mock_monotonic.return_value = 1003.0
        updated = pool.mark_rate_limited(updated, "gpt-4", 429)
        assert updated.rate_limited["gpt-4"].consecutive_count == 4
        assert abs(updated.rate_limited["gpt-4"].reset_at - 2203.0) < 1.0

    @patch("agent.credential_pool.time.monotonic")
    def test_model_isolation(self, mock_monotonic, pool_with_entry):
        """A 429 on model-A does not block model-B on the same credential."""
        pool, entry = pool_with_entry
        mock_monotonic.return_value = 1000.0

        # model-A triggers 429
        pool.mark_rate_limited(entry, "gpt-4", 429)

        # model-A should be excluded
        available_a = pool._available_entries(model_id="gpt-4", clear_expired=True)
        assert len(available_a) == 0

        # model-B should still be available
        available_b = pool._available_entries(model_id="claude-3", clear_expired=True)
        assert len(available_b) == 1
        assert available_b[0].id == entry.id

    @patch("agent.credential_pool.time.monotonic")
    def test_expired_rate_limit_recovery(self, mock_monotonic, pool_with_entry):
        """After TTL expires, the entry becomes available and rate_limited is pruned."""
        pool, entry = pool_with_entry

        # Set rate limit: TTL=300s, current time=1000
        mock_monotonic.return_value = 1000.0
        pool.mark_rate_limited(entry, "gpt-4", 429)

        # Confirm rate-limited
        available = pool._available_entries(model_id="gpt-4", clear_expired=True)
        assert len(available) == 0

        # Advance time past TTL (1000 + 300 = 1300)
        mock_monotonic.return_value = 1301.0

        # Should auto-recover and prune expired entry
        available = pool._available_entries(model_id="gpt-4", clear_expired=True)
        assert len(available) == 1
        assert available[0].id == entry.id
        assert "gpt-4" not in available[0].rate_limited


class TestFromDictRateLimitedRobustness:
    """Test from_dict handling of edge cases in rate_limited data."""

    def test_negative_reset_at_dropped(self):
        """Negative reset_at should be dropped with warning."""
        data = {
            "provider": "test",
            "auth_type": "api_key",
            "rate_limited": {"m1": {"reset_at": -100.0, "consecutive_count": 1}},
        }
        entry = PooledCredential.from_dict("test", data)
        assert "m1" not in entry.rate_limited

    def test_negative_consecutive_count_dropped(self):
        """Negative consecutive_count should be dropped with warning."""
        data = {
            "provider": "test",
            "auth_type": "api_key",
            "rate_limited": {"m1": {"reset_at": 1000.0, "consecutive_count": -1}},
        }
        entry = PooledCredential.from_dict("test", data)
        assert "m1" not in entry.rate_limited

    def test_string_numeric_fields_accepted(self):
        """String-formatted numeric fields should be converted and accepted."""
        data = {
            "provider": "test",
            "auth_type": "api_key",
            "rate_limited": {"m1": {"reset_at": "1000.5", "consecutive_count": "3"}},
        }
        entry = PooledCredential.from_dict("test", data)
        assert "m1" in entry.rate_limited
        assert entry.rate_limited["m1"].reset_at == 1000.5
        assert entry.rate_limited["m1"].consecutive_count == 3

    def test_non_numeric_string_dropped(self):
        """Non-numeric string values should be dropped."""
        data = {
            "provider": "test",
            "auth_type": "api_key",
            "rate_limited": {"m1": {"reset_at": "not_a_number", "consecutive_count": 1}},
        }
        entry = PooledCredential.from_dict("test", data)
        assert "m1" not in entry.rate_limited
