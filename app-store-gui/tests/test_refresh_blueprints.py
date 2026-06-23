from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import webapp.main as main


def _cp(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_refresh_blueprints_not_git_managed(monkeypatch):
    # No .git in the worktree → not git-sync-managed → 400, no git calls.
    monkeypatch.setattr(main.os.path, "exists", lambda p: False)
    resp = asyncio.run(main.refresh_blueprints())
    assert resp.status_code == 400
    assert json.loads(resp.body)["ok"] is False


def test_refresh_blueprints_success_changed(monkeypatch):
    monkeypatch.setenv("GIT_SYNC_BRANCH", "main")
    monkeypatch.setattr(main.os.path, "exists", lambda p: True)

    calls = []
    revs = iter(["aaaaaaa", "bbbbbbb"])  # before, after

    def fake_run(args, **kwargs):
        gitargs = list(args[3:])  # strip ["git", "-C", worktree]
        calls.append(gitargs)
        if gitargs[:1] == ["rev-parse"]:
            return _cp(0, next(revs) + "\n")
        return _cp(0, "ok")

    monkeypatch.setattr(main.subprocess, "run", fake_run)
    resp = asyncio.run(main.refresh_blueprints())
    assert resp.status_code == 200
    body = json.loads(resp.body)
    assert body["ok"] is True
    assert body["changed"] is True
    assert body["before"] == "aaaaaaa" and body["after"] == "bbbbbbb"
    assert ["fetch", "--depth=1", "origin", "main"] in calls
    assert ["reset", "--hard", "FETCH_HEAD"] in calls


def test_refresh_blueprints_fetch_failure_returns_502(monkeypatch):
    monkeypatch.setattr(main.os.path, "exists", lambda p: True)

    def fake_run(args, **kwargs):
        gitargs = list(args[3:])
        if gitargs[:1] == ["rev-parse"]:
            return _cp(0, "aaaaaaa\n")
        if gitargs[:1] == ["fetch"]:
            return _cp(1, "", "could not resolve host")
        return _cp(0, "ok")

    monkeypatch.setattr(main.subprocess, "run", fake_run)
    resp = asyncio.run(main.refresh_blueprints())
    assert resp.status_code == 502
    assert json.loads(resp.body)["ok"] is False
