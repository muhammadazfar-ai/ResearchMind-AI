"""The agents of ResearchMind. Each one is a small, testable function.

Planner    -> splits the topic into sub-questions
Researcher -> searches the web for every sub-question (in parallel)
Writer     -> drafts a cited report from retrieved evidence
Checker    -> verifies every cited claim against the passages it cites
Reviser    -> rewrites flagged claims so they match the evidence
Translator -> renders the verified report in another language
"""
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from src import config
from src.llm import LLMError
from src.prompts import (
    SYSTEM_CHECKER,
    SYSTEM_PLANNER,
    SYSTEM_REVISER,
    SYSTEM_TRANSLATOR,
    SYSTEM_WRITER,
    build_check_prompt,
    build_planner_prompt,
    build_revise_prompt,
    build_translate_prompt,
    build_writer_prompt,
)
from src.search import SearchError

_CITATION = re.compile(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]")
VERDICTS = ("supported", "partial", "unsupported")


# ------------------------------------------------------------------ helpers
def parse_json(text: str):
    """Parses JSON even if the model wrapped it in prose or code fences."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def clean_citations(text: str, max_n: int) -> str:
    """Normalises [1, 2] to [1][2] and drops citation numbers the model invented (outside 1..max_n)."""
    def _replace(match):
        numbers = [int(n) for n in re.split(r"\s*,\s*", match.group(1))]
        return "".join(f"[{n}]" for n in numbers if 1 <= n <= max_n)

    return _CITATION.sub(_replace, text)


# ------------------------------------------------------------------ Planner
def plan(topic: str, n: int, complete) -> list:
    """Returns up to n search-friendly sub-questions; falls back to the topic itself."""
    messages = [
        {"role": "system", "content": SYSTEM_PLANNER},
        {"role": "user", "content": build_planner_prompt(topic, n)},
    ]
    data = parse_json(complete(messages, role="fast", temperature=0.2, json_mode=True))
    questions = []
    for q in (data or {}).get("sub_questions", []):
        if isinstance(q, str) and q.strip() and q.strip() not in questions:
            questions.append(q.strip())
    return questions[:n] or [topic]


# --------------------------------------------------------------- Researcher
def research_web(questions: list, cfg: dict, searcher) -> tuple:
    """Searches every sub-question in parallel. Returns (unique_sources, error_messages)."""
    def one(question):
        try:
            return searcher(
                question,
                max_results=cfg["results_per_question"],
                search_depth=cfg["search_depth"],
            ), None
        except SearchError as e:
            return [], str(e)

    sources, errors, seen = [], [], set()
    with ThreadPoolExecutor(max_workers=min(len(questions), 5) or 1) as pool:
        for results, error in pool.map(one, questions):
            if error and error not in errors:
                errors.append(error)
            for item in results:
                key = item.get("url") or item["title"]
                if key in seen:
                    continue
                seen.add(key)
                sources.append({**item, "kind": "web"})
    return sources, errors


# ------------------------------------------------------------------- Writer
def format_context(selected: list, sources: list) -> str:
    """Renders retrieved passages grouped under their numbered source."""
    blocks = []
    for source in sources:
        texts = [c["text"] for c in selected if c["cite"] == source["n"]]
        body = "\n".join(f"- {t}" for t in texts)
        where = source["url"] if source.get("url") else "uploaded document"
        blocks.append(f"[{source['n']}] {source['title']} ({where})\n{body}")
    return "\n\n".join(blocks)


def write_stream(topic, mode, questions, selected, sources, stream):
    """Yields text deltas of the first draft."""
    messages = [
        {"role": "system", "content": SYSTEM_WRITER},
        {
            "role": "user",
            "content": build_writer_prompt(topic, mode, questions, format_context(selected, sources), len(sources)),
        },
    ]
    yield from stream(messages, role="writer", temperature=0.3)


# ------------------------------------------------------------------ Checker
def extract_claims(report: str) -> tuple:
    """Splits a report into cited sentences (claims to verify) and counts uncited factual sentences."""
    claims, uncited = [], 0
    for raw in report.split("\n"):
        line = raw.strip()
        if not line or line.startswith(("#", ">", "---")):
            continue
        line = re.sub(r"^([-*\u2022]|\d+[.)])\s+", "", line)
        line = re.sub(r"\*\*|__", "", line)
        # "claim. [1]" -> "claim[1]." so the citation stays attached to its sentence
        line = re.sub(r"([.!?])\s*((?:\[\d{1,2}\])+)", r"\2\1", line)
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            sentence = sentence.strip()
            cites = sorted({int(n) for n in re.findall(r"\[(\d{1,2})\]", sentence)})
            if cites:
                claims.append({"id": len(claims) + 1, "text": sentence, "cites": cites})
            elif len(sentence.split()) >= 8:
                uncited += 1
    return claims, uncited


def check_claims(claims: list, store, complete) -> dict:
    """Verifies claims in small batches against the passages they cite. Returns {claim_id: {verdict, reason}}."""
    results = {}
    to_check = claims[: config.MAX_CLAIMS_CHECKED]

    for start in range(0, len(to_check), config.CHECK_BATCH_SIZE):
        batch = to_check[start : start + config.CHECK_BATCH_SIZE]
        items = [
            {"id": c["id"], "text": c["text"], "evidence": store.support_for(c["text"], c["cites"], k=2)}
            for c in batch
        ]
        messages = [
            {"role": "system", "content": SYSTEM_CHECKER},
            {"role": "user", "content": build_check_prompt(items)},
        ]
        try:
            data = parse_json(complete(messages, role="fast", temperature=0.0, json_mode=True))
        except LLMError as e:
            print(f"[DEBUG WARNING] Checker batch skipped: {e}")
            continue
        for entry in (data or {}).get("results", []):
            try:
                claim_id = int(entry["id"])
            except (KeyError, TypeError, ValueError):
                continue
            verdict = str(entry.get("verdict", "")).lower().strip()
            if verdict in VERDICTS and any(c["id"] == claim_id for c in batch):
                results[claim_id] = {"verdict": verdict, "reason": str(entry.get("reason", ""))[:200]}
    return results


def summarize_check(claims: list, results: dict, uncited: int) -> dict:
    counts = Counter(results[c["id"]]["verdict"] for c in claims if c["id"] in results)
    checked = sum(counts.values())
    total_sentences = len(claims) + uncited
    return {
        "claims": len(claims),
        "checked": checked,
        "supported": counts["supported"],
        "partial": counts["partial"],
        "unsupported": counts["unsupported"],
        "support_rate": (counts["supported"] / checked) if checked else None,
        "citation_coverage": (len(claims) / total_sentences) if total_sentences else None,
    }


# ------------------------------------------------------------------ Reviser
def revise(report: str, claims: list, results: dict, store, complete) -> str:
    """Rewrites the report so flagged claims match their evidence. Returns the original if the rewrite looks broken."""
    issues = []
    for claim in claims:
        verdict = results.get(claim["id"])
        if verdict and verdict["verdict"] in ("unsupported", "partial"):
            issues.append({
                "text": claim["text"],
                "verdict": verdict["verdict"],
                "reason": verdict["reason"] or "not backed by the cited source",
                "evidence": store.support_for(claim["text"], claim["cites"], k=2),
            })
    if not issues:
        return report

    messages = [
        {"role": "system", "content": SYSTEM_REVISER},
        {"role": "user", "content": build_revise_prompt(report, issues)},
    ]
    revised = complete(messages, role="writer", temperature=0.1).strip()
    # Guard against a stub or truncated answer replacing the whole report
    if len(revised) < 0.5 * len(report):
        return report
    return revised


# --------------------------------------------------------------- Translator
def translate(report: str, language: str, complete) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_TRANSLATOR},
        {"role": "user", "content": build_translate_prompt(report, language)},
    ]
    translated = complete(messages, role="writer", temperature=0.1).strip()
    if len(translated) < 0.3 * len(report):
        raise LLMError("The translation came back too short.")
    return translated
