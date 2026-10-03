import os

import pytest

from src import agents, config, pipeline
from src.documents import load_documents
from src.evidence import EvidenceStore
from tests.conftest import WEB, fake_searcher


def run(fake_llm, embedder, **kwargs):
    events = list(pipeline.run_research(
        "AI in agriculture", searcher=fake_searcher, embedder=embedder, llm_api=fake_llm, **kwargs
    ))
    return events, events[-1]


# ------------------------------------------------------------ agents
def test_parse_json_handles_fences_and_prose():
    assert agents.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert agents.parse_json('Sure! {"a": 1} hope that helps') == {"a": 1}
    assert agents.parse_json("not json") is None


def test_planner_falls_back_to_topic_on_garbage(fake_llm):
    assert agents.plan("my topic", 3, lambda *a, **k: "garbage") == ["my topic"]


def test_planner_limits_and_dedupes():
    reply = '{"sub_questions": ["a", "a", "b", "c", "d"]}'
    assert agents.plan("t", 3, lambda *a, **k: reply) == ["a", "b", "c"]


def test_clean_citations():
    assert agents.clean_citations("A [1] B [9] C [2, 7] D [2026]", 3) == "A [1] B  C [2] D [2026]"
    assert agents.clean_citations("x [1, 2]", 5) == "x [1][2]"


def test_extract_claims_keeps_citation_with_its_sentence():
    report = (
        "## Heading\n"
        "- Solar panels share land with crops. [1] Drones detect disease early [2].\n"
        "This sentence has plenty of words but no citation at all in it.\n"
        "> quoted verification note [1]\n"
    )
    claims, uncited = agents.extract_claims(report)
    assert [c["cites"] for c in claims] == [[1], [2]]
    assert claims[0]["text"].startswith("Solar panels share land with crops")
    assert uncited == 1


def test_research_web_dedupes_urls_and_collects_errors():
    from src.search import SearchError

    def searcher(q, max_results, search_depth):
        if q == "bad":
            raise SearchError("boom")
        return [{"title": "T", "url": "https://x", "content": "text"}]

    sources, errors = agents.research_web(["a", "b", "bad"], {"results_per_question": 2, "search_depth": "basic"}, searcher)
    assert len(sources) == 1 and errors == ["boom"]


# ---------------------------------------------------------- evidence
def test_evidence_ranks_relevant_sources_first_and_drops_irrelevant(embedder):
    sources = [dict(s, kind="web") for s in WEB["solar"] + WEB["drone"]]
    store = EvidenceStore(sources, embedder)
    selected, numbered = store.gather(["solar panels on farmland", "drone crop monitoring"], 5)
    titles = [s["title"] for s in numbered]
    assert titles[0] == "Agrivoltaics explained"
    assert "Cooking pasta" not in titles
    assert len(selected) <= config.MAX_EVIDENCE_CHUNKS
    assert all(sum(1 for c in selected if c["source_idx"] == i) <= config.MAX_CHUNKS_PER_SOURCE for i in range(3))


def test_support_for_returns_passages_from_cited_source_only(embedder):
    sources = [dict(s, kind="web") for s in WEB["solar"] + WEB["drone"]]
    store = EvidenceStore(sources, embedder)
    store.gather(["solar panels on farmland", "drone crop monitoring"], 5)
    passages = store.support_for("drones detect plant disease", [2], k=2)
    assert passages and all("Drones" in p for p in passages)
    assert store.support_for("anything", [99]) == []


# ----------------------------------------------------------- pipeline
def test_full_agent_loop_catches_and_fixes_unsupported_claim(fake_llm, embedder):
    events, last = run(fake_llm, embedder)
    assert last["done"] and not last["error"]
    result = last["result"]

    # Checker flagged the planted claim, Reviser removed it, re-check is clean
    m = result["metrics"]
    assert m["first_pass"]["unsupported"] == 1
    assert m["final"]["unsupported"] == 0 and m["final"]["support_rate"] == 1.0
    assert m["revised"] is True
    assert "cheese" not in result["report"]

    # Invented citation [9] removed, sources + verification note appended
    assert "[9]" not in result["report"]
    assert "## Sources" in result["report"] and "https://a.example/agri" in result["report"]
    assert "Verification" in result["report"]

    # Trace shows every agent
    trace = last["trace"]
    for agent in ("Planner", "Researcher", "Retriever", "Writer", "Checker", "Reviser"):
        assert agent in trace
    assert "Translator" not in trace


def test_draft_streams_progressively(fake_llm, embedder):
    events, _ = run(fake_llm, embedder)
    drafts = [e["report"] for e in events if e["report"] and e["report"].startswith("*Draft")]
    assert len(drafts) >= 3


def test_urdu_runs_translator_after_verification(fake_llm, embedder):
    _, last = run(fake_llm, embedder, language="Urdu")
    result = last["result"]
    assert result["report"].startswith("[UR]")
    assert "cheese" not in result["report_english"]
    assert "Translator" in last["trace"]


def test_documents_only_mode(fake_llm, embedder, tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("Greenhouse farming with solar panels on farmland increases crop yield per hectare. " * 10)
    _, last = run(fake_llm, embedder, use_web=False, document_paths=[str(doc)])
    assert last["done"]
    sources = last["result"]["sources"]
    assert sources and all(s["kind"] == "document" for s in sources)
    assert "📄" in last["result"]["report"]


def test_no_web_and_no_documents_is_a_clear_error(fake_llm, embedder):
    _, last = run(fake_llm, embedder, use_web=False)
    assert last["error"] and "nothing to research" in last["report"]


def test_empty_topic_is_rejected(fake_llm, embedder):
    events = list(pipeline.run_research("  ", searcher=fake_searcher, embedder=embedder, llm_api=fake_llm))
    assert events[-1]["error"]


def test_search_failure_without_documents_surfaces_message(fake_llm, embedder):
    from src.search import SearchError

    def broken(q, max_results=4, search_depth="basic"):
        raise SearchError("`TAVILY_API_KEY` is missing")

    events = list(pipeline.run_research("topic here", searcher=broken, embedder=embedder, llm_api=fake_llm))
    assert events[-1]["error"] and "TAVILY_API_KEY" in events[-1]["report"]


def test_writer_failure_is_reported_not_raised(embedder, fake_llm):
    import types
    from src.llm import LLMError

    def bad_stream(*a, **k):
        raise LLMError("All Groq models failed.")
        yield

    broken = types.SimpleNamespace(complete=fake_llm.complete, stream=bad_stream)
    events = list(pipeline.run_research("topic here", searcher=fake_searcher, embedder=embedder, llm_api=broken))
    assert events[-1]["error"] and "All Groq models failed" in events[-1]["report"]


def test_checker_outage_still_delivers_report(embedder, fake_llm):
    import types
    from src.llm import LLMError

    def complete(messages, role="fast", temperature=0.0, json_mode=False):
        if "Planner" in messages[0]["content"]:
            return fake_llm.complete(messages, role, temperature, json_mode)
        raise LLMError("rate limited")

    llm_api = types.SimpleNamespace(complete=complete, stream=fake_llm.stream)
    _, last = None, list(pipeline.run_research("topic here", searcher=fake_searcher, embedder=embedder, llm_api=llm_api))[-1]
    assert last["done"] and "unavailable" in last["result"]["report"]


def test_reviser_guard_rejects_stub_rewrites(embedder):
    report = "word " * 200
    claims = [{"id": 1, "text": "x [1]", "cites": [1]}]
    results = {1: {"verdict": "unsupported", "reason": "r"}}

    class Store:
        def support_for(self, *a, **k):
            return ["evidence"]

    assert agents.revise(report, claims, results, Store(), lambda *a, **k: "ok") == report


# ---------------------------------------------------------- documents
def test_load_documents_pdf_txt_and_warnings(tmp_path):
    from reportlab.lib.pagesizes import LETTER
    from reportlab.platypus import Paragraph, SimpleDocTemplate
    from reportlab.lib.styles import getSampleStyleSheet

    pdf = tmp_path / "paper.pdf"
    SimpleDocTemplate(str(pdf), pagesize=LETTER).build(
        [Paragraph("Agrivoltaics lets farmers grow crops under solar panels and earn extra income. " * 5, getSampleStyleSheet()["BodyText"])]
    )
    txt = tmp_path / "empty.txt"
    txt.write_text("hi")
    odd = tmp_path / "data.xyz"
    odd.write_text("whatever " * 30)

    docs, warnings = load_documents([str(pdf), str(txt), str(odd)])
    assert [d["title"] for d in docs] == ["paper.pdf"]
    assert "Agrivoltaics" in docs[0]["content"] and docs[0]["kind"] == "document"
    assert len(warnings) == 2


# ---------------------------------------------------------- evaluation
def test_evaluate_aggregates_and_writes_reports(fake_llm, embedder, tmp_path):
    import functools

    import evaluate

    runner = functools.partial(pipeline.run_research, searcher=fake_searcher, embedder=embedder, llm_api=fake_llm)
    rows = [evaluate.evaluate_topic("AI in agriculture", "Quick", runner)]
    rows.append(evaluate.evaluate_topic("x", "Quick", runner))          # too short -> failure row
    assert rows[0]["ok"] and rows[0]["support_first"] < rows[0]["support_final"] == 1.0
    assert not rows[1]["ok"]

    summary = evaluate.aggregate(rows)
    assert summary["completed"] == 1 and summary["revised_count"] == 1
    evaluate.write_outputs(rows, summary, str(tmp_path), "Quick")
    text = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "first draft" in text and "100%" in text and (tmp_path / "results.json").exists()
