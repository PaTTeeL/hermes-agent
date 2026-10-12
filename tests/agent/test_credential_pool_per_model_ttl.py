"""Tests for 429 per-model rate-limit TTL core functionality."""

import pytest
from unittest.mock import patch

from agent.credential_pool import CredentialPool, PooledCredential

# Per-model TTL reset_at is now persisted as wall clock (time.time(), ~1.75e9).
# Tests mock time.time to a fixed wall-clock baseline so stale-guard logic
# (from_dict drops entries > TTL cap behind now) doesn't trip on small test values.
# Anchor to the real current time.time() so from_dict's stale guard sees test
# reset_at values as "ahead of now" (legitimate TTL), not stale residue.
import time as _time
WALL_NOW = _time.time()


@pytest.fixture(autouse=True)
def _explicit_ttl_config(monkeypatch):
    """Pin the TTL ladder explicitly instead of relying on machine-config defaults.

    The escalation tests assert 300/600/900/1200 against the *default* ladder;
    without this pin, a machine config that customizes rate_limit_ttl_* keys
    would silently change the numbers under test.
    """
    monkeypatch.setattr(
        "agent.credential_pool._load_config_safe",
        lambda: {"credential_pool_strategies": {
            "rate_limit_ttl_first_seconds": 300,
            "rate_limit_ttl_step_seconds": 300,
            "rate_limit_ttl_max_seconds": 3600,
        }},
    )


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
            access_token="test-key",
        )
        with patch.object(CredentialPool, "_persist", return_value=None):
            pool = CredentialPool("test", [entry])
        return pool, entry

    @patch("agent.credential_pool.time.time")
    def test_ttl_escalation(self, mock_time, pool_with_entry):
        """Consecutive 429s escalate TTL: 5min -> 10min -> 15min -> ... (1h cap)."""
        pool, entry = pool_with_entry

        # 1st 429: now=WALL_NOW, TTL = 300s (5min), reset_at = WALL_NOW + 300
        mock_time.return_value = WALL_NOW
        updated = pool.mark_rate_limited(entry, "gpt-4", 429)
        assert updated.model_cooldowns["gpt-4"].consecutive_count == 1
        assert abs(updated.model_cooldowns["gpt-4"].reset_at - (WALL_NOW + 300.0)) < 1.0

        # 2nd 429: now=WALL_NOW+1, TTL = 300 + 300*1 = 600s, reset_at = WALL_NOW + 1 + 600
        mock_time.return_value = WALL_NOW + 1.0
        updated = pool.mark_rate_limited(updated, "gpt-4", 429)
        assert updated.model_cooldowns["gpt-4"].consecutive_count == 2
        assert abs(updated.model_cooldowns["gpt-4"].reset_at - (WALL_NOW + 1.0 + 600.0)) < 1.0

        # 3rd 429: now=WALL_NOW+2, TTL = 300 + 300*2 = 900s, reset_at = WALL_NOW + 2 + 900
        mock_time.return_value = WALL_NOW + 2.0
        updated = pool.mark_rate_limited(updated, "gpt-4", 429)
        assert updated.model_cooldowns["gpt-4"].consecutive_count == 3
        assert abs(updated.model_cooldowns["gpt-4"].reset_at - (WALL_NOW + 2.0 + 900.0)) < 1.0

        # 4th 429: now=WALL_NOW+3, TTL = 300 + 300*3 = 1200s (below 1h cap), reset_at = WALL_NOW + 3 + 1200
        mock_time.return_value = WALL_NOW + 3.0
        updated = pool.mark_rate_limited(updated, "gpt-4", 429)
        assert updated.model_cooldowns["gpt-4"].consecutive_count == 4
        assert abs(updated.model_cooldowns["gpt-4"].reset_at - (WALL_NOW + 3.0 + 1200.0)) < 1.0

    @patch("agent.credential_pool.time.time")
    def test_model_isolation(self, mock_time, pool_with_entry):
        """A 429 on model-A does not block model-B on the same credential."""
        pool, entry = pool_with_entry
        mock_time.return_value = WALL_NOW

        # model-A triggers 429
        pool.mark_rate_limited(entry, "gpt-4", 429)

        # model-A should be excluded
        available_a, _pending_a = pool._available_entries(model="gpt-4", clear_expired=True)
        assert len(available_a) == 0

        # model-B should still be available
        available_b, _pending_b = pool._available_entries(model="claude-3", clear_expired=True)
        assert len(available_b) == 1
        assert available_b[0].id == entry.id

    @patch("agent.credential_pool.time.time")
    def test_expired_rate_limit_recovery(self, mock_time, pool_with_entry):
        """After TTL expires, the entry becomes available and the window is pruned."""
        pool, entry = pool_with_entry

        # Set rate limit: TTL=300s, current time=WALL_NOW
        mock_time.return_value = WALL_NOW
        pool.mark_rate_limited(entry, "gpt-4", 429)

        # Confirm rate-limited
        available, _pending = pool._available_entries(model="gpt-4", clear_expired=True)
        assert len(available) == 0

        # Advance time past TTL (WALL_NOW + 300)
        mock_time.return_value = WALL_NOW + 301.0

        # Should auto-recover and prune expired entry
        available, _pending = pool._available_entries(model="gpt-4", clear_expired=True)
        assert len(available) == 1
        assert available[0].id == entry.id
        assert "gpt-4" not in (available[0].model_cooldowns or {})

    @patch("agent.credential_pool.time.time")
    def test_success_clear_resets_consecutive(self, mock_time, pool_with_entry):
        """clear_rate_limit() drops the record; a later 429 counts from 1 again."""
        pool, entry = pool_with_entry
        mock_time.return_value = WALL_NOW
        first = pool.mark_rate_limited(entry, "gpt-4", 429)
        second = pool.mark_rate_limited(first, "gpt-4", 429)
        assert second.model_cooldowns["gpt-4"].consecutive_count == 2

        pool.clear_rate_limit("test1", "gpt-4")

        cleared = next(e for e in pool.entries() if e.id == "test1")
        assert "gpt-4" not in (cleared.model_cooldowns or {})
        mock_time.return_value = WALL_NOW + 5.0
        after = pool.mark_rate_limited(cleared, "gpt-4", 429)
        assert after.model_cooldowns["gpt-4"].consecutive_count == 1


class TestFromDictRateLimitedRobustness:
    """Test from_dict handling of edge cases in persisted window data."""

    def test_negative_reset_at_dropped(self):
        """Negative reset_at should be dropped with warning."""
        data = {
            "provider": "test",
            "auth_type": "api_key",
            "model_cooldowns": {"m1": {"reset_at": -100.0, "consecutive_count": 1}},
        }
        entry = PooledCredential.from_dict("test", data)
        assert "m1" not in (entry.model_cooldowns or {})

    def test_negative_consecutive_count_dropped(self):
        """Negative consecutive_count should be dropped with warning."""
        data = {
            "provider": "test",
            "auth_type": "api_key",
            "model_cooldowns": {"m1": {"reset_at": WALL_NOW + 300.0, "consecutive_count": -1}},
        }
        entry = PooledCredential.from_dict("test", data)
        assert "m1" not in (entry.model_cooldowns or {})

    def test_string_numeric_fields_accepted(self):
        """String-formatted numeric fields should be converted and accepted."""
        data = {
            "provider": "test",
            "auth_type": "api_key",
            "model_cooldowns": {"m1": {"reset_at": str(WALL_NOW + 300.5), "consecutive_count": "3"}},
        }
        entry = PooledCredential.from_dict("test", data)
        assert "m1" in entry.model_cooldowns
        assert entry.model_cooldowns["m1"].reset_at == WALL_NOW + 300.5
        assert entry.model_cooldowns["m1"].consecutive_count == 3

    def test_non_numeric_string_dropped(self):
        """Non-numeric string values should be dropped."""
        data = {
            "provider": "test",
            "auth_type": "api_key",
            "model_cooldowns": {"m1": {"reset_at": "not_a_number", "consecutive_count": 1}},
        }
        entry = PooledCredential.from_dict("test", data)
        assert "m1" not in (entry.model_cooldowns or {})


class TestRecoveryPathFirst429:
    """The recovery path records the per-model window even on a first 429."""

    @patch("agent.credential_pool.time.time")
    def test_first_429_on_recovery_path_writes_window(self, mock_time):
        from agent.agent_runtime_helpers import _recover_rate_limit

        mock_time.return_value = WALL_NOW
        entry = PooledCredential(
            id="test1", provider="test", label="Test Key", auth_type="api_key",
            priority=0, source="manual", access_token="test-key",
        )
        with patch.object(CredentialPool, "_persist", return_value=None):
            pool = CredentialPool("test", [entry])
        rotate_calls = []
        recovered, has_retried = _recover_rate_limit(
            pool, has_retried_429=False, error_context=None, api_key_hint=None,
            credential_id="test1",
            rotate_and_swap=lambda *a, **k: rotate_calls.append(a) or True,
            model_id="gpt-4",
        )
        # First 429: record the window, retry the same credential, don't rotate yet.
        assert recovered is False
        assert has_retried is True
        assert rotate_calls == []
        row = (pool.entries()[0].model_cooldowns or {}).get("gpt-4")
        assert row is not None
        assert abs(row.reset_at - (WALL_NOW + 300.0)) < 1.0
