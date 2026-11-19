"""Behavior contracts for kanban wake delivery to local CLI/TUI sessions.

A subscription stamped with a local-family platform label ("local", or the legacy
"cli"/"tui") must deliver its terminal event into the target local session's
authority FIFO through the ONE cached LocalSessionAdapter, and a refused wake
must be observable: the log carries the refusal reason and a per-subscription
consecutive refusal count, and the claim is rewound (retried loudly, never
silently consumed).
"""
import asyncio
import json
import logging

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.kanban_watchers_notifier import (
    _KanbanNotification,
    _adapter_for_subscription,
    _notifier_collect,
)
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_contract import SessionActor
from gateway.session_local import create_local_session
from gateway.session_local_recovery import local_adapter
from gateway.session_persistence import authority_persistence
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


async def _local_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    runner._kanban_dispatcher_lock_handle = object()
    runner._kanban_sub_fail_counts = {}
    runner._kanban_wake_refusal_counts = {}
    authority = await initialize_session_authority(
        runner, profile_id="default", instance_id="local-wake-fixture")
    return runner, authority


def _actor():
    return SessionActor("test-owner", "default",
                        frozenset({"session:create", "session:submit", "session:read"}), "fixture")


def _create_task_and_sub(conn, *, platform, chat_id, mode="notify+wake"):
    tid = kb.create_task(conn, title="local wake task", assignee="worker")
    kbn.add_notify_sub(conn, task_id=tid, platform=platform, chat_id=chat_id,
                       chat_type="dm", delivery_mode=mode)
    kb.complete_task(conn, tid, summary="local handoff")
    return tid


def _unseen(tid, platform, chat_id):
    conn = kbc.connect()
    try:
        return kbn.unseen_events_for_sub(
            conn, task_id=tid, platform=platform, chat_id=chat_id, kinds=["completed"])[1]
    finally:
        conn.close()


async def _tick(runner):
    deliveries = await asyncio.to_thread(
        _notifier_collect, runner, kb, notifier_profile=None, gc_due=False, gc_retention_days=30)
    for delivery in deliveries:
        await _KanbanNotification(runner, delivery, platform_cls=Platform,
                                  sub_fail_counts=runner._kanban_sub_fail_counts).deliver()
    return deliveries


def _admission_texts(authority, sid):
    rows = authority_persistence(authority).admissions.list(session_id=sid, pending_only=False)
    return [json.dumps(row.payload, default=str) for row in rows]


@pytest.mark.asyncio
async def test_local_subscription_wakes_the_live_local_session(tmp_path, monkeypatch):
    runner, authority = await _local_runner(tmp_path, monkeypatch)
    try:
        ref = create_local_session(authority, _actor(),
                                   dict(request_id="wake-1", source="gui", model="fixture", toolsets=[]))
        sid = ref.session_id
        conn = kbc.connect()
        try:
            tid = _create_task_and_sub(conn, platform="local", chat_id=sid)
        finally:
            conn.close()

        # The subscription resolves to the ONE cached local adapter the authority's
        # own path uses — never a throwaway instance.
        sub = {"task_id": tid, "platform": "local", "chat_id": sid, "thread_id": "",
               "notifier_profile": None, "delivery_mode": "notify+wake"}
        adapter = _adapter_for_subscription(runner, Platform.LOCAL, sub, None)
        assert adapter is local_adapter(authority)
        assert runner.adapters.get(Platform.LOCAL) is adapter

        await _tick(runner)

        texts = _admission_texts(authority, sid)
        assert any(tid in text for text in texts), texts
        assert not _unseen(tid, "local", sid)
        assert runner._kanban_wake_refusal_counts == {}
    finally:
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_legacy_tui_label_delivers_through_the_local_path(tmp_path, monkeypatch):
    runner, authority = await _local_runner(tmp_path, monkeypatch)
    try:
        ref = create_local_session(authority, _actor(),
                                   dict(request_id="wake-2", source="gui", model="fixture", toolsets=[]))
        sid = ref.session_id
        conn = kbc.connect()
        try:
            tid = _create_task_and_sub(conn, platform="tui", chat_id=sid)
        finally:
            conn.close()

        await _tick(runner)

        texts = _admission_texts(authority, sid)
        assert any(tid in text for text in texts), texts
        assert not _unseen(tid, "tui", sid)
    finally:
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_refused_local_wake_logs_reason_and_count_and_rewinds(tmp_path, monkeypatch, caplog):
    runner, _authority = await _local_runner(tmp_path, monkeypatch)
    try:
        conn = kbc.connect()
        try:
            # No such local session: the authority refuses / cannot admit the wake.
            tid = _create_task_and_sub(conn, platform="local", chat_id="missing-session-id")
        finally:
            conn.close()

        with caplog.at_level(logging.WARNING):
            await _tick(runner)
        key = (tid, "local", "missing-session-id", "")
        assert runner._kanban_wake_refusal_counts.get(key) == 1
        warnings = [r for r in caplog.records if "wake not accepted" in r.getMessage()]
        assert warnings, [r.getMessage() for r in caplog.records]
        assert "reason=" in warnings[-1].getMessage()
        assert "refusal #1" in warnings[-1].getMessage()
        # The claim was rewound, not consumed: the event is still unseen.
        assert _unseen(tid, "local", "missing-session-id")

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            await _tick(runner)
        assert runner._kanban_wake_refusal_counts.get(key) == 2
        warnings = [r for r in caplog.records if "wake not accepted" in r.getMessage()]
        assert "refusal #2" in warnings[-1].getMessage()
        assert _unseen(tid, "local", "missing-session-id")
    finally:
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_routing_key_chat_id_is_normalized_for_the_whole_local_family(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    kb.init_db()
    conn = kbc.connect()
    try:
        for platform in ("local", "cli", "tui"):
            tid = kb.create_task(conn, title=f"sub {platform}", assignee="worker")
            kbn.add_notify_sub(conn, task_id=tid, platform=platform,
                               chat_id="agent:default:local:dm:20260614_141414_a0d47b",
                               chat_type="dm", delivery_mode="notify+wake")
            row = conn.execute(
                "SELECT chat_id FROM kanban_notify_subs WHERE task_id = ? AND platform = ?",
                (tid, platform)).fetchone()
            assert row["chat_id"] == "20260614_141414_a0d47b", (platform, row["chat_id"])
    finally:
        conn.close()
