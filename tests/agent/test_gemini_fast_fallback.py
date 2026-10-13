"""Regression tests for #11314 — credential-pool rotation vs. fallback.

_pool_may_recover_from_rate_limit() is the hinge between credential-pool
rotation and fallback-provider activation.  Rotation is only worth waiting on
when the pool exists, has an available entry, and has more than one entry to
rotate to; otherwise we should fall back to the configured fallback provider
immediately.
"""
import time
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


def _pool_with_window(ttl, entries=1):
    """A pool reporting nothing available for the model, carrying one live window."""
    p = MagicMock()
    p.has_available.return_value = True
    p.entries.return_value = list(range(entries))
    p._available_entries.return_value = ([], [])
    p.rate_limit_min_ttl.return_value = ttl
    return p


def test_long_window_prefers_switching_over_waiting():
    """A window beyond the default 600s threshold is not worth waiting out."""
    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        assert _pool_may_recover_from_rate_limit(_pool_with_window(time.time() + 1200), model_id="m") is False
        assert _pool_may_recover_from_rate_limit(_pool_with_window(time.time() + 300), model_id="m") is True


def test_wait_threshold_is_configurable():
    """rate_limit_wait_threshold_seconds shrinks the waitable window."""
    cfg = {"credential_pool_strategies": {"rate_limit_wait_threshold_seconds": 60}}
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        assert _pool_may_recover_from_rate_limit(_pool_with_window(time.time() + 300), model_id="m") is False
        assert _pool_may_recover_from_rate_limit(_pool_with_window(time.time() + 30), model_id="m") is True


def test_idle_single_key_pool_still_passes_through():
    """No window record (idle entry) keeps the pass-through, threshold or not."""
    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        assert _pool_may_recover_from_rate_limit(_pool_with_window(None), model_id="m") is True
