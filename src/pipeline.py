"""Orchestrates the agents: Planner -> Researcher -> Retriever -> Writer -> Checker -> Reviser -> Translator."""
import re
import time
import types

from src import agents, config, llm
from src.documents import load_documents
from src.evidence import EvidenceError, EvidenceStore
from src.search import search_topic

DEFAULT_LLM = types.SimpleNamespace(complete=llm.complete, stream=llm.stream_completion)

_ICONS = {"run": "⏳", "ok": "✅", "warn": "⚠️", "fail": "❌"}


class Trace:
    """The live 'agent activity' log shown in the UI."""

    def __init__(self):
        self.steps = []

    def start(self, name: str, detail: str = "") -> int:
        self.steps.append({"name": name, "status": "run", "detail": detail, "items": []})
        return len(self.steps) - 1

    def finish(self, idx: int, detail: str = None, status: str = "ok", items: list = None):
        step = self.steps[idx]
        step["status"] = status
        if detail is not None:
            step["detail"] = detail
        if items:
            step["items"] = items

    def render(self) -> str:
        lines = ["### 🤖 Agent activity", ""]
        for s in self.steps:
            lines.append(f"{_ICONS[s['status']]} **{s['name']}** — {s['detail']}".rstrip(" —"))
            for item in s["items"]:
                lines.append(f"   - {item}")
        return "\n".join(lines)


def _event(trace, report=None, done=False, error=False, result=None):
    return {"trace": trace.render(), "report": report, "done": done, "error": error, "result": result}


def sources_markdown(sources: list) -> str:
    lines = []
    for s in sources:
        if s.get("kind") == "document":
            lines.append(f"{s['n']}. 📄 **{s['title']}** (uploaded document)")
        else:
            title = re.sub(r"[\[\]]", "", s["title"]) or s["url"]
            lines.append(f"{s['n']}. [{title}]({s['url']})")
    return "\n\n---\n\n## Sources\n\n" + "\n".join(lines)


def verification_markdown(final: dict, first: dict, revised: bool) -> str:
    if not final or not final["checked"]:
        return "\n\n> 🔎 **Verification:** the automatic fact-check was unavailable for this report. Please verify key facts yourself."
    text = f"{final['supported']} of {final['checked']} cited claims are fully supported by their sources"
    extras = []
    if final["partial"]:
        extras.append(f"{final['partial']} partially supported")
    if final["unsupported"]:
        extras.append(f"{final['unsupported']} unsupported")
    if extras:
        text += " (" + ", ".join(extras) + ")"
    if revised and first and first["checked"]:
        text += f". The first draft scored {first['supported']}/{first['checked']}; the Reviser agent corrected the rest"
    return f"\n\n> 🔎 **Verification:** {text}. This is an automated check — verify critical facts yourself."


def _check(report, store, complete):
    claims, uncited = agents.extract_claims(report)
    results = agents.check_claims(claims, store, complete) if claims else {}
    return claims, results, agents.summarize_check(claims, results, uncited)


def run_research(
    topic: str,
    mode: str = "Standard",
    language: str = "English",
    depth: str = config.DEFAULT_DEPTH,
    use_web: bool = True,
    document_paths=None,
    *,
    searcher=None,
    embedder=None,
    llm_api=None,
):
    """Generator. Yields {"trace", "report", "done", "error", "result"} events as the agents work.

    The final event has done=True and a `result` dict with the report, sources and fact-check metrics.
    `searcher`, `embedder` and `llm_api` can be replaced for testing.
    """
    started = time.time()
    searcher = searcher or search_topic
    llm_api = llm_api or DEFAULT_LLM
    cfg = config.DEPTH_PRESETS.get(depth, config.DEPTH_PRESETS[config.DEFAULT_DEPTH])
    trace = Trace()

    topic = (topic or "").strip()
    if len(topic) < 2:
        yield _event(trace, report="### ⚠️ Please enter a research topic.", error=True)
        return

    # ---- 0. uploaded documents ---------------------------------------------------------
    documents = []
    if document_paths:
        step = trace.start("Documents", "reading uploaded files...")
        yield _event(trace)
        documents, warnings = load_documents(document_paths)
        trace.finish(
            step,
            f"{len(documents)} file(s) loaded" if documents else "no readable documents",
            "ok" if documents else "warn",
            items=warnings,
        )

    if not use_web and not documents:
        yield _event(trace, report="### ⚠️ Web search is off and no readable documents were uploaded, so there is nothing to research.", error=True)
        return

    # ---- 1. Planner --------------------------------------------------------------------
    step = trace.start("Planner", "splitting the topic into research questions...")
    yield _event(trace)
    try:
        questions = agents.plan(topic, cfg["sub_questions"], llm_api.complete)
        trace.finish(step, f"{len(questions)} research question(s)", items=[f"{i}. {q}" for i, q in enumerate(questions, 1)])
    except llm.LLMError as e:
        questions = [topic]
        trace.finish(step, f"planner unavailable ({e}); using the topic as a single question", "warn")
    yield _event(trace)

    # ---- 2. Researcher -----------------------------------------------------------------
    sources, errors = list(documents), []
    if use_web:
        step = trace.start("Researcher", f"searching the web ({len(questions)} parallel searches)...")
        yield _event(trace)
        web_sources, errors = agents.research_web(questions, cfg, searcher)
        sources = web_sources + sources
        if web_sources:
            trace.finish(step, f"{len(web_sources)} web source(s) found", items=[f"note: {e}" for e in errors])
        else:
            trace.finish(step, "no web results", "warn", items=errors)
        yield _event(trace)

    if not sources:
        message = "### ⚠️ " + (errors[0] if use_web and errors else "No sources were found for this topic.")
        yield _event(trace, report=message, error=True)
        return

    # ---- 3. Index + retrieve -----------------------------------------------------------
    step = trace.start("Retriever", "chunking, embedding and indexing sources...")
    yield _event(trace)
    try:
        store = EvidenceStore(sources, embedder)
        selected, numbered = store.gather(questions, cfg["chunks_per_question"])
    except EvidenceError as e:
        trace.finish(step, str(e), "fail")
        yield _event(trace, report=f"### ⚠️ {e}", error=True)
        return
    except Exception as e:
        print(f"[DEBUG ERROR] Retrieval failed: {e}")
        trace.finish(step, f"retrieval failed: {e}", "fail")
        yield _event(trace, report=f"### ❌ Retrieval failed: `{e}`", error=True)
        return
    trace.finish(step, f"{len(store.records)} passages indexed, {len(selected)} selected from {len(numbered)} source(s)")
    yield _event(trace)

    # ---- 4. Writer ---------------------------------------------------------------------
    step = trace.start("Writer", "drafting the cited report...")
    draft = ""
    try:
        for delta in agents.write_stream(topic, mode, questions, selected, numbered, llm_api.stream):
            draft += delta
            shown = agents.clean_citations(draft, len(numbered))
            yield _event(trace, report="*Draft — fact-checking will follow...*\n\n" + shown)
    except llm.LLMError as e:
        trace.finish(step, str(e), "fail")
        yield _event(trace, report=f"### ❌ {e}", error=True)
        return
    report = agents.clean_citations(draft, len(numbered)).strip()
    if not report:
        trace.finish(step, "empty answer", "fail")
        yield _event(trace, report="### ⚠️ The model returned an empty answer. Please try again.", error=True)
        return
    trace.finish(step, f"draft written ({len(report.split())} words)")
    yield _event(trace, report="*Draft — fact-checking in progress...*\n\n" + report)

    # ---- 5. Checker (+ Reviser loop) ---------------------------------------------------
    first_metrics = final_metrics = None
    revised_flag = False
    step = trace.start("Checker", "verifying every cited claim against its sources...")
    yield _event(trace, report="*Draft — fact-checking in progress...*\n\n" + report)
    try:
        claims, results, first_metrics = _check(report, store, llm_api.complete)
        final_metrics = first_metrics
        if first_metrics["checked"]:
            m = first_metrics
            trace.finish(step, f"{m['supported']}/{m['checked']} claims supported, {m['partial']} partial, {m['unsupported']} unsupported")
        else:
            trace.finish(step, "no claims could be verified", "warn")
        yield _event(trace, report="*Draft — fact-checking in progress...*\n\n" + report)

        for _ in range(config.MAX_REVISION_ROUNDS):
            flagged = final_metrics["partial"] + final_metrics["unsupported"]
            if not flagged:
                break
            step = trace.start("Reviser", f"rewriting {flagged} flagged claim(s)...")
            yield _event(trace, report="*Draft — revising flagged claims...*\n\n" + report)
            candidate = agents.revise(report, claims, results, store, llm_api.complete)
            if candidate == report:
                trace.finish(step, "no usable revision", "warn")
                break
            claims2, results2, metrics2 = _check(candidate, store, llm_api.complete)
            old_rate = final_metrics["support_rate"] or 0
            new_rate = metrics2["support_rate"] or 0
            if metrics2["checked"] and new_rate >= old_rate:
                report, claims, results, final_metrics, revised_flag = candidate, claims2, results2, metrics2, True
                trace.finish(step, f"revised; now {metrics2['supported']}/{metrics2['checked']} claims supported")
            else:
                trace.finish(step, "revision did not improve the report; kept the original draft", "warn")
            yield _event(trace, report=report)
    except llm.LLMError as e:
        trace.finish(step, f"unavailable ({e}); report delivered unverified", "warn")
        yield _event(trace, report=report)

    # ---- 6. Translator -----------------------------------------------------------------
    english_report = report
    if language != "English":
        step = trace.start("Translator", f"translating the verified report to {language}...")
        yield _event(trace, report=report)
        try:
            report = agents.translate(report, language, llm_api.complete)
            trace.finish(step, f"translated to {language}")
        except llm.LLMError as e:
            trace.finish(step, f"translation failed ({e}); showing English", "warn")

    # ---- 7. Assemble -------------------------------------------------------------------
    final_md = report + sources_markdown(numbered) + verification_markdown(final_metrics, first_metrics, revised_flag)
    elapsed = round(time.time() - started, 1)
    result = {
        "topic": topic,
        "report": final_md,
        "report_english": english_report,
        "language": language,
        "questions": questions,
        "sources": numbered,
        "metrics": {
            "first_pass": first_metrics,
            "final": final_metrics,
            "revised": revised_flag,
            "sources": len(numbered),
            "passages_indexed": len(store.records),
            "seconds": elapsed,
        },
    }
    yield _event(trace, report=final_md, done=True, result=result)
