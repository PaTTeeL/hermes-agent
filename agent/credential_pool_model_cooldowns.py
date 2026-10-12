"""Model-scoped rate-limit cooldowns for pooled credentials.

Anthropic enforces its API rate limits (requests / tokens per minute) per
model, so a generic 429 for one Claude model says nothing about the same
credential's standing for its sibling models. Such a 429 is recorded as a
cooldown on the requested model only, beside the credential-wide status that
auth, billing and payment failures keep benching the whole credential with.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.credential_pool import PooledCredential

# A Codex ChatGPT-account model entitlement 400 is a plan property, not a window: bench the
# (credential, model) pair until an explicit ``hermes auth reset`` clears the blocked slot (#71970).
MODEL_ENTITLEMENT_BENCH_SECONDS = 365 * 24 * 60 * 60
# A usage-window plan limit: this model is refused until the allowance resets; every other model
# (a Free or non-Large one) stays usable on the same credential.
FAILURE_REASON_USAGE_WINDOW = "usage_window"


def pool_failure_reason(reason: Any, error_context: Optional[dict[str, Any]], billing_unverified: bool) -> str:
    """The pool's failure reason for a rotation (*reason* is a ``FailoverReason``). A usage-window 402
    benches only the refused model; an ambiguous billing body is sized as transient, not a 1-hour bench."""
    from agent.error_classifier import FailoverReason
    if reason == FailoverReason.billing and (error_context or {}).get("usage_window"):
        return FAILURE_REASON_USAGE_WINDOW
    if reason == FailoverReason.billing and billing_unverified:
        from agent.credential_pool import FAILURE_REASON_BILLING_UNVERIFIED
        return FAILURE_REASON_BILLING_UNVERIFIED
    return reason.value


def model_cooldown_until(entry: PooledCredential, model: Optional[str]) -> Optional[float]:
    """Active cooldown blocking *entry* for *model*, or ``None``.

    Callers that do not know the model stay conservative: any active model
    cooldown blocks them, so an unscoped route cannot reuse the credential.
    """
    windows = entry.model_cooldowns or {}
    values = windows.values() if not model else (windows.get(model),)
    now = time.time()
    active = [
        w.blocked_until
        for w in values
        if w is not None and w.blocked_until is not None and w.blocked_until > now
    ]
    return max(active) if active else None


def merge_model_cooldowns(*maps: Any) -> dict[str, Any]:
    """Merge per-model window rows across snapshots.

    Each slot resolves independently: the reset slot moves as an indivisible
    ``(reset_at, consecutive_count)`` pair so the later window keeps its own
    ladder count, and the blocked slot keeps the later ``blocked_until`` with
    its own reason.  Legacy value shapes (bare float benches, plain dicts)
    normalize on the way in; malformed or empty values are dropped.
    """
    from agent.credential_pool import ModelWindow, _window_row

    merged: dict[str, ModelWindow] = {}
    for snapshot in maps:
        if not isinstance(snapshot, dict):
            continue
        for model, raw in snapshot.items():
            row = _window_row(raw)
            if row is None:
                continue
            current = merged.get(model)
            if current is None:
                merged[model] = row
                continue
            if (row.reset_at or 0.0) > (current.reset_at or 0.0):
                reset_at, consecutive = row.reset_at, row.consecutive_count
            else:
                reset_at, consecutive = current.reset_at, current.consecutive_count
            if (row.blocked_until or 0.0) > (current.blocked_until or 0.0):
                blocked_until, blocked_reason = row.blocked_until, row.blocked_reason
            else:
                blocked_until, blocked_reason = current.blocked_until, current.blocked_reason
            merged[model] = ModelWindow(reset_at, consecutive, blocked_until, blocked_reason)
    return merged


class CredentialPoolModelCooldownMixin:
    def token_is_blocked(self, token: str, *, model: Optional[str] = None) -> bool:
        """Whether a pool cooldown blocks *token* for *model*.

        Closes the paths that hand out a native Anthropic token without
        selecting it from the pool (env / borrowed credentials). Tokens the
        pool does not know fail open: no row can attribute a cooldown to them.
        """
        with self._lock:
            return any(
                entry.runtime_api_key == token and model_cooldown_until(entry, model) is not None
                for entry in self._entries
            )

    def _is_model_scoped_failure(
        self, status_code: Optional[int], model: Optional[str], failure_reason: Optional[str],
    ) -> bool:
        """Anthropic per-model 429s, and a Codex ChatGPT-account model entitlement 400: the
        account cannot use *model*, but the credential stays valid for every other model (#71970)."""
        from agent.credential_pool import FAILURE_REASON_BILLING, FAILURE_REASON_BILLING_UNVERIFIED

        if not model:
            return False
        if failure_reason in ("model_entitlement", FAILURE_REASON_USAGE_WINDOW):
            return True
        return (
            self.provider == "anthropic" and status_code == 429
            and failure_reason not in (FAILURE_REASON_BILLING, FAILURE_REASON_BILLING_UNVERIFIED)
        )

    def _cool_down_model(
        self, entry: PooledCredential, model: str, error_context: Optional[dict[str, Any]],
        failure_reason: Optional[str] = None,
    ) -> None:
        """Record a cooldown for *model* on *entry* and every sibling sharing its key.

        Same TTL policy as a credential-wide 429 (provider ``reset_at`` wins, a
        sole credential keeps its short bench), except a ``model_entitlement``
        rejection, which stays benched until the explicit reset path clears it.
        Siblings matter because a ``model_config`` twin seeded from the same key
        would otherwise be re-selected for the very model that just failed.
        Caller holds the lock.
        """
        from agent.credential_pool import ModelWindow, _exhausted_ttl, _normalize_error_context

        if failure_reason == "model_entitlement":
            until = time.time() + MODEL_ENTITLEMENT_BENCH_SECONDS
            reason = "entitlement"
        else:
            until = _normalize_error_context(error_context).get("reset_at") or (
                time.time() + _exhausted_ttl(429, sole_credential=self._is_sole_credential())
            )
            reason = "cooldown"
        bench = ModelWindow(blocked_until=until, blocked_reason=reason)
        failed_key = entry.runtime_api_key
        for scoped in list(self._entries):
            if scoped.id != entry.id and not (failed_key and scoped.runtime_api_key == failed_key):
                continue
            windows = merge_model_cooldowns(scoped.model_cooldowns, {model: bench})
            self._adopt(scoped, persist=False, model_cooldowns=windows)
        self._persist()

    def limit_state(self, models: Iterable[str]) -> Optional[dict[str, Any]]:
        """What a picker should say about this pool's rate limits, or ``None`` when nothing is limited.

        ``{"scope": "account", "resets_at": epoch}`` when every live entry is benched credential-wide
        (the whole login is out; another model won't help), else ``{"scope": "models", "models":
        {model: epoch}}`` for the given *models* no usable entry can serve until a model cooldown
        ends. Entitlement benches (a year: the plan lacks the model) aren't windows that reset, so
        they're left out. Read-only: never clears or persists a cooldown.
        """
        from agent.credential_pool import STATUS_DEAD, _exhausted_until

        now = time.time()
        with self._lock:
            live = [entry for entry in self._entries if entry.last_status != STATUS_DEAD]
            sole = len(live) <= 1
            benched = {entry.id: until for entry in live
                       if (until := _exhausted_until(entry, sole_credential=sole)) and until > now}
            if live and len(benched) == len(live):
                return {"scope": "account", "resets_at": min(benched.values())}
            usable = [entry for entry in live if entry.id not in benched]
            cooled: dict[str, float] = {}
            for model in models:
                waits = [model_cooldown_until(entry, model) for entry in usable]
                if waits and all(waits) and min(waits) - now < MODEL_ENTITLEMENT_BENCH_SECONDS / 2:
                    cooled[model] = min(waits)
        return {"scope": "models", "models": cooled} if cooled else None
