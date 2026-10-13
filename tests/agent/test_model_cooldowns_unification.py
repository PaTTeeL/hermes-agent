"""Unified per-model window rows (``model_cooldowns``): legacy migration,
dual-slot semantics, success-clear scoping, and provider-reset priority."""
import time
from unittest.mock import patch

from agent.credential_pool import CredentialPool, ModelWindow, PooledCredential


def _payload(**kw):
    base = {
        "provider": "test", "id": "t1", "label": "x", "auth_type": "api_key",
        "priority": 0, "source": "manual", "access_token": "k",
    }
    base.update(kw)
    return base


class TestLegacyMigration:
    def test_asymmetric_legacy_keys_migrate(self):
        now = time.time()
        e1 = PooledCredential.from_dict("test", _payload(
            rate_limited={"m1": {"reset_at": now + 120, "consecutive_count": 2}},
        ))
        assert e1.model_cooldowns["m1"].reset_at == now + 120
        assert e1.model_cooldowns["m1"].consecutive_count == 2

        e2 = PooledCredential.from_dict("test", _payload(model_cooldowns={"m2": now + 600}))
        assert e2.model_cooldowns["m2"].blocked_until == now + 600
        assert e2.model_cooldowns["m2"].blocked_reason == "cooldown"

        e3 = PooledCredential.from_dict("test", _payload())
        assert not e3.model_cooldowns

    def test_both_legacy_keys_merge_into_one_row(self):
        now = time.time()
        e = PooledCredential.from_dict("test", _payload(
            rate_limited={"m": {"reset_at": now + 60, "consecutive_count": 1}},
            model_cooldowns={"m": now + 3600},
        ))
        row = e.model_cooldowns["m"]
        assert row.reset_at == now + 60 and row.consecutive_count == 1
        assert row.blocked_until == now + 3600 and row.blocked_reason == "cooldown"

    def test_legacy_keys_do_not_write_back(self):
        now = time.time()
        e = PooledCredential.from_dict("test", _payload(
            rate_limited={"m1": {"reset_at": now + 120, "consecutive_count": 1}},
            model_cooldowns={"m2": now + 600},
        ))
        out = e.to_dict()
        assert "rate_limited" not in out
        assert "model_cooldowns" in out
        assert all(isinstance(v, dict) for v in out["model_cooldowns"].values())
        again = PooledCredential.from_dict("test", out)
        assert set(again.model_cooldowns or {}) == {"m1", "m2"}
        assert again.model_cooldowns["m1"].consecutive_count == 1
        assert again.model_cooldowns["m2"].blocked_until == now + 600

    def test_bad_reason_tolerated_as_cooldown(self):
        now = time.time()
        e = PooledCredential.from_dict("test", _payload(
            model_cooldowns={"m": {"blocked_until": now + 60, "blocked_reason": "weird"}},
        ))
        assert e.model_cooldowns["m"].blocked_reason == "cooldown"

    def test_mixed_value_shapes_load(self):
        now = time.time()
        e = PooledCredential.from_dict("test", _payload(
            model_cooldowns={
                "f": now + 600,
                "d": {"reset_at": now + 60, "consecutive_count": 2},
            },
        ))
        assert e.model_cooldowns["f"].blocked_until == now + 600
        assert e.model_cooldowns["d"].reset_at == now + 60

    def test_merge_handles_mixed_shapes(self):
        now = time.time()
        from agent.credential_pool_model_cooldowns import merge_model_cooldowns
        merged = merge_model_cooldowns(
            {"m": now + 600},
            {"m": {"reset_at": now + 60, "consecutive_count": 3}},
        )
        row = merged["m"]
        assert row.blocked_until == now + 600
        assert row.reset_at == now + 60 and row.consecutive_count == 3

    def test_empty_values_are_dropped(self):
        now = time.time()
        e = PooledCredential.from_dict("test", _payload(
            model_cooldowns={"m": {"blocked_reason": 1}},
        ))
        assert not e.model_cooldowns
        from agent.credential_pool_model_cooldowns import merge_model_cooldowns
        assert merge_model_cooldowns({"m": {"blocked_reason": 1}}) == {}


class TestDualSlotSemantics:
    def _pool(self, entry):
        with patch.object(CredentialPool, "_persist", return_value=None):
            return CredentialPool("test", [entry])

    def test_select_skips_both_slots(self):
        now = time.time()
        entry = PooledCredential.from_dict("test", _payload(
            model_cooldowns={
                "rw": {"reset_at": now + 300, "consecutive_count": 1},
                "bw": {"blocked_until": now + 300, "blocked_reason": "cooldown"},
            },
        ))
        pool = self._pool(entry)
        assert pool.select(model="rw") is None
        assert pool.select(model="bw") is None
        assert pool.select(model="other") is not None

    def test_next_available_at_min_over_both_slots(self):
        """next_available_at counts the 429 window alongside the blocked slot."""
        now = time.time()
        entry = PooledCredential.from_dict("test", _payload(
            model_cooldowns={"m": {
                "reset_at": now + 300, "consecutive_count": 1,
                "blocked_until": now + 900, "blocked_reason": "cooldown",
            }},
        ))
        pool = self._pool(entry)
        assert pool.next_available_at(model="m") == now + 300

        entry2 = PooledCredential.from_dict("test", _payload(
            model_cooldowns={"m2": {
                "reset_at": now + 900, "consecutive_count": 1,
                "blocked_until": now + 300, "blocked_reason": "cooldown",
            }},
        ))
        pool2 = self._pool(entry2)
        assert pool2.next_available_at(model="m2") == now + 300

    def test_success_clear_keeps_blocked_slot(self):
        now = time.time()
        entry = PooledCredential.from_dict("test", _payload(
            model_cooldowns={"m": {
                "reset_at": now + 300, "consecutive_count": 1,
                "blocked_until": now + 3600, "blocked_reason": "cooldown",
            }},
        ))
        pool = self._pool(entry)
        pool.clear_rate_limit("t1", "m")
        row = pool.entries()[0].model_cooldowns["m"]
        assert row.reset_at is None and row.consecutive_count == 0
        assert row.blocked_until == now + 3600

    def test_auth_reset_clears_blocked_only(self):
        from agent.credential_pool_admin import _cleared_status_copy

        now = time.time()
        entry = PooledCredential.from_dict("test", _payload(
            model_cooldowns={"m": {
                "reset_at": now + 300, "consecutive_count": 1,
                "blocked_until": now + 3600, "blocked_reason": "entitlement",
            }},
        ))
        cleared = _cleared_status_copy(entry)
        row = cleared.model_cooldowns["m"]
        assert row.reset_at == now + 300 and row.consecutive_count == 1
        assert row.blocked_until is None and row.blocked_reason is None


class TestProviderResetPriority:
    def _pool(self, entry):
        with patch.object(CredentialPool, "_persist", return_value=None):
            return CredentialPool("test", [entry])

    def test_provider_reset_wins_and_syncs_last_error(self):
        entry = PooledCredential.from_dict("test", _payload())
        pool = self._pool(entry)
        now = time.time()
        updated = pool.mark_rate_limited(entry, "m1", 429, {"reset_at": now + 7200})
        row = updated.model_cooldowns["m1"]
        assert abs(row.reset_at - (now + 7200)) < 1.0
        assert updated.last_error_reset_at == row.reset_at
        assert row.consecutive_count == 1

    def test_unparseable_reset_falls_back_to_ladder(self):
        entry = PooledCredential.from_dict("test", _payload())
        pool = self._pool(entry)
        now = time.time()
        updated = pool.mark_rate_limited(entry, "m1", 429, {"reset_at": "not-a-time"})
        row = updated.model_cooldowns["m1"]
        assert row.reset_at is not None
        assert 290.0 <= row.reset_at - now <= 310.0  # repo default first window = 300s
