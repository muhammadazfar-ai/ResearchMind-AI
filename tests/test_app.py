import functools
import os

import pytest

pytest.importorskip("gradio")

import app
from src import pipeline
from tests.conftest import fake_searcher


@pytest.fixture
def patched_app(monkeypatch, fake_llm, embedder):
    """Runs the real UI handler with fake search/LLM/embeddings injected."""
    monkeypatch.setattr(
        app,
        "run_research",
        functools.partial(pipeline.run_research, searcher=fake_searcher, embedder=embedder, llm_api=fake_llm),
    )
    return app


def collect(gen):
    return list(gen)


def test_ui_builds_with_expected_controls():
    assert app.app is not None


def test_run_agent_streams_then_offers_md_and_pdf(patched_app):
    outputs = collect(patched_app.run_agent("AI in agriculture", "Standard", "English", "Standard", True, None))
    welcome, trace, report, files = outputs[-1]
    assert "Agent activity" in trace["value"] and "Verification" in report["value"]
    paths = files["value"]
    assert files["visible"] and {os.path.splitext(p)[1] for p in paths} == {".md", ".pdf"}
    assert all(os.path.getsize(p) > 100 for p in paths)


def test_urdu_gets_markdown_only_and_rtl(patched_app):
    outputs = collect(patched_app.run_agent("AI in agriculture", "Standard", "Urdu", "Standard", True, None))
    _, _, report, files = outputs[-1]
    assert report.get("rtl") is True
    assert [os.path.splitext(p)[1] for p in files["value"]] == [".md"]


def test_empty_query_keeps_welcome_screen(patched_app):
    welcome, trace, report, files = collect(patched_app.run_agent("", "Standard", "English", "Standard", True, None))[0]
    assert welcome["visible"] is True and report["visible"] is False


def test_error_event_shows_message_and_no_files(patched_app):
    outputs = collect(patched_app.run_agent("topic here", "Standard", "English", "Standard", False, None))
    _, _, report, files = outputs[-1]
    assert "nothing to research" in report["value"] and files["visible"] is False


def test_pdf_handles_emoji_links_and_headings(tmp_path):
    path = app.create_pdf("# Title\n\n## Sources\n\n1. [Solar & crops](https://x.example/a?b=1&c=2)\n2. 📄 **notes.pdf** (uploaded)\n\n> 🔎 **Verification:** ok")
    assert path and os.path.getsize(path) > 500
