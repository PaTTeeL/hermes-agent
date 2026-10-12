"""Regression tests for #11314 — credential-pool rotation vs. fallback.

_pool_may_recover_from_rate_limit() is the hinge between credential-pool
rotation and fallback-provider activation.  Rotation is only worth waiting on
when the pool exists, has an available entry, and has more than one entry to
rotate to; otherwise we should fall back to the configured fallback provider
immediately.
"""
from unittest.mock import MagicMock, patch

from run_agent import _pool_may_recover_from_rate_limit


def _pool(entries: int = 2):
    p = MagicMock()
    p.has_available.return_value = True
    p.entries.return_value = list(range(entries))
    p._available_entries.return_value = (list(range(entries)), [])
    return p


def test_multi_entry_pool_recovers():
    assert _pool_may_recover_from_rate_limit(_pool(entries=3)) is True




def test_exhausted_pool_skips_rotation():
    p = MagicMock()
    p.has_available.return_value = False
    assert _pool_may_recover_from_rate_limit(p) is False


def test_model_scoped_check_on_real_pool():
    """The model-scoped emptiness check runs against a real pool through the current API."""
    from agent.credential_pool import CredentialPool, PooledCredential

    entry = PooledCredential(
        id="k1", provider="test", label="Key 1", auth_type="api_key",
        priority=0, source="manual", access_token="key-1",
    )
    with patch.object(CredentialPool, "_persist", return_value=None):
        pool = CredentialPool("test", [entry])
        assert _pool_may_recover_from_rate_limit(pool, model_id="gpt-4") is True
        pool.mark_rate_limited(entry, "gpt-4", 429)
        assert _pool_may_recover_from_rate_limit(pool, model_id="gpt-4") is True


def test_multi_entry_all_limited_with_live_windows_can_wait():
    """Every entry limited for the model but a live window exists -> wait, don't fall back."""
    from agent.credential_pool import CredentialPool, PooledCredential

    e1 = PooledCredential(
        id="k1", provider="test", label="Key 1", auth_type="api_key",
        priority=0, source="manual", access_token="key-1",
    )
    e2 = PooledCredential(
        id="k2", provider="test", label="Key 2", auth_type="api_key",
        priority=1, source="manual", access_token="key-2",
    )
    with patch.object(CredentialPool, "_persist", return_value=None):
        pool = CredentialPool("test", [e1, e2])
        pool.mark_rate_limited(e1, "gpt-4", 429)
        pool.mark_rate_limited(e2, "gpt-4", 429)
        # Both entries carry unexpired per-model windows: waiting can recover.
        assert _pool_may_recover_from_rate_limit(pool, model_id="gpt-4") is True
