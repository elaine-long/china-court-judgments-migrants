"""T06v: Gemini backend selection and secret hygiene."""
import subprocess

import pytest

from src import llm_extract as L

ENV_KEYS = ("GEMINI_BACKEND", "GOOGLE_CLOUD_PROJECT", "VERTEX_API_KEY", "GOOGLE_CLOUD_LOCATION")


@pytest.fixture
def clean_env(monkeypatch):
    # gemini_backend() calls load_dotenv; stop it from reading the real .env
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


@pytest.mark.parametrize("env,expected", [
    ({}, "aistudio"),                                                   # nothing configured
    ({"GOOGLE_CLOUD_PROJECT": "p"}, "vertex"),                          # project -> vertex (ADC)
    ({"VERTEX_API_KEY": "k"}, "vertex"),                                # vertex key -> vertex
    ({"GEMINI_BACKEND": "aistudio", "GOOGLE_CLOUD_PROJECT": "p"}, "aistudio"),  # explicit override
    ({"GEMINI_BACKEND": "VERTEX"}, "vertex"),                           # case-insensitive
    ({"GEMINI_BACKEND": "bogus", "VERTEX_API_KEY": "k"}, "vertex"),     # unknown value ignored
])
def test_backend_selection(clean_env, env, expected):
    for k, v in env.items():
        clean_env.setenv(k, v)
    assert L.gemini_backend() == expected


def _secrets():
    from dotenv import dotenv_values
    env = dotenv_values(L.paths.ROOT / ".env")
    return [v for k, v in env.items() if v and ("KEY" in k or k == "GOOGLE_CLOUD_PROJECT") and len(v) >= 6]


def test_no_secret_in_git_tracked_or_staged_files():
    for s in _secrets():
        for extra in ([], ["--cached"]):
            r = subprocess.run(["git", "grep", "-l", "-F", *extra, s], cwd=L.paths.ROOT,
                               capture_output=True, text=True)
            assert r.stdout.strip() == "", "a secret from .env appears in a git-tracked file"


def test_no_secret_in_source():
    root = L.paths.ROOT
    files = [p for d in ("src", "scripts", "tests", "prompts") for p in (root / d).rglob("*")
             if p.is_file() and (p.suffix in (".py", ".md") or p.name.endswith(".bak_pre_vertex"))]
    for s in _secrets():
        for p in files:
            assert s not in p.read_text(encoding="utf-8", errors="ignore"), p
