"""Duplicate user content + shapes agreeing must anchor the LATER row.

The identity/content anchor scan picks the durable row whose content matches the
warm target text. When a session legitimately contains the same user text in
two separate turns (re-ask / "continue" / "yes"), /undo of the LATER turn must
anchor the LATER row (positional semantics), not the first one.
"""

from hermes_state import SessionDB  # noqa: F401 (import path check)


def _fresh_db(tmp_path, name):
    from hermes_state import SessionDB as RootSessionDB

    db = RootSessionDB(db_path=tmp_path / f"{name}.db")
    sid = f"review-{name}"
    db.create_session(sid, source="cli")
    return db, sid


def test_undo_later_duplicate_turn_anchors_later_row(tmp_path):
    """durable: [first ask, answer one, first ask, answer two]; /undo 1 turn
    (target_ordinal=3, warm-derived like undo_last) must rewind row 3 only."""
    db, sid = _fresh_db(tmp_path, "dup-undo")
    db.append_message(sid, "user", "first ask")
    db.append_message(sid, "assistant", "answer one")
    db.append_message(sid, "user", "first ask")
    db.append_message(sid, "assistant", "answer two")
    model_history, _display = db.get_resume_conversations(sid)
    model_history = [m for m in model_history if m.get("role") != "session_meta"]

    warm_user_texts = [
        m.get("content") for m in model_history if m.get("role") == "user"
    ]
    # shapes agree (2 user turns both sides, no merging); undo_last derives
    # target_ordinal = len(user_indices) - turns_undone = 1 for the last turn
    outcome = db.rewind_user_turn(sid, 1, warm_history=model_history)

    remaining = db.get_messages_as_conversation(sid, include_row_ids=True)
    remaining_texts = [m.get("content") for m in remaining]
    assert outcome is not None
    assert remaining_texts == ["first ask", "answer one"], (
        f"later duplicate turn must anchor the later row; got remaining={remaining_texts!r} "
        f"(warm user texts were {warm_user_texts!r})"
    )
    db.close()


def test_retry_negative_ordinal_duplicate_turn_anchors_later_row(tmp_path):
    """Same scenario, /retry passes -1: must anchor the LAST duplicate row."""
    db, sid = _fresh_db(tmp_path, "dup-retry")
    db.append_message(sid, "user", "first ask")
    db.append_message(sid, "assistant", "answer one")
    db.append_message(sid, "user", "first ask")
    db.append_message(sid, "assistant", "answer two")
    model_history, _display = db.get_resume_conversations(sid)
    model_history = [m for m in model_history if m.get("role") != "session_meta"]

    outcome = db.rewind_user_turn(
        sid, -1, warm_history=model_history, require_retryable=True,
    )

    remaining = db.get_messages_as_conversation(sid, include_row_ids=True)
    remaining_texts = [m.get("content") for m in remaining]
    assert outcome is not None
    assert remaining_texts == ["first ask", "answer one"], (
        f"/retry -1 must anchor the LAST duplicate row; got remaining={remaining_texts!r}"
    )
    db.close()


def test_retry_merged_session_live_text_documents_behavior(tmp_path):
    """Documents what /retry re-sends on a merged session: the anchored durable
    row's live view (first segment) vs the caller's warm merged text."""
    db, sid = _fresh_db(tmp_path, "merged-retry")
    db.append_message(sid, "user", "first ask")
    db.append_message(sid, "user", "second ask")
    db.append_message(sid, "assistant", "answer")
    model_history, _display = db.get_resume_conversations(sid)
    model_history = [m for m in model_history if m.get("role") != "session_meta"]

    outcome = db.rewind_user_turn(
        sid, -1, warm_history=model_history, require_retryable=True,
    )

    assert outcome is not None
    remaining = db.get_messages_as_conversation(sid, include_row_ids=True)
    assert [m["role"] for m in remaining] == []
    print(f"live_text after merged-session retry: {outcome.live_text!r}")
    db.close()
