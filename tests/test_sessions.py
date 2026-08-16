from __future__ import annotations

import json

import pytest

from self_correct import cli, sessions


def test_session_round_trip(tmp_path) -> None:
    path = tmp_path / "nested" / "session.json"
    sessions.save_session(
        path,
        prompt="Explain the result.",
        config={"model": "gpt-test", "strictness": 0.8},
        result={"content": "Verified result."},
    )

    loaded = sessions.load_session(path)
    assert loaded["prompt"] == "Explain the result."
    assert loaded["config"]["model"] == "gpt-test"
    assert loaded["result"]["content"] == "Verified result."


def test_session_rejects_unknown_schema(tmp_path) -> None:
    path = tmp_path / "session.json"
    path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported session schema"):
        sessions.load_session(path)


def test_resume_restores_settings_and_allows_model_override(tmp_path, monkeypatch) -> None:
    path = tmp_path / "session.json"
    sessions.save_session(
        path,
        prompt="Original prompt",
        config={
            "model": "saved-model",
            "strictness": 0.7,
            "provider": "ollama",
            "tools": ["wikipedia"],
        },
        result={"content": "Earlier result"},
    )
    captured = {}

    def fake_verify(args):
        captured.update(vars(args))
        return 0

    monkeypatch.setattr(cli, "cmd_verify", fake_verify)
    assert cli.main(["resume", str(path), "--model", "new-model"]) == 0
    assert captured["prompt"] == "Original prompt"
    assert captured["model"] == "new-model"
    assert captured["strictness"] == 0.7
    assert captured["provider"] == "ollama"
    assert captured["tools"] == ["wikipedia"]
