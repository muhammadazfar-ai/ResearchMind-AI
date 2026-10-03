"""Batch evaluation of ResearchMind on a list of topics.

Usage:
    python evaluate.py                       # all topics in evaluation/topics.txt
    python evaluate.py --limit 3 --depth Quick
    python evaluate.py --topics my_topics.txt --pause 15

For every topic it runs the full agent pipeline and records the fact-checker's verdicts:
  * support rate     - share of cited claims fully supported by the passages they cite
  * first draft vs final - how much the Checker/Reviser loop improved the report
  * citation coverage - share of factual sentences that carry a citation
Results are written to evaluation/results.json and evaluation/RESULTS.md.
"""
import argparse
import json
import os
import statistics
import time

from dotenv import load_dotenv

load_dotenv(override=True)


def evaluate_topic(topic: str, depth: str, runner) -> dict:
    row = {"topic": topic, "ok": False}
    try:
        final = None
        for event in runner(topic, "Standard", "English", depth, True, None):
            if event["error"]:
                row["error"] = event["report"]
                return row
            if event["done"]:
                final = event["result"]
        if not final:
            row["error"] = "pipeline produced no result"
            return row
    except Exception as e:
        row["error"] = str(e)
        return row

    m = final["metrics"]
    first, last = m["first_pass"] or {}, m["final"] or {}
    row.update({
        "ok": True,
        "sources": m["sources"],
        "passages": m["passages_indexed"],
        "claims": last.get("claims"),
        "checked": last.get("checked"),
        "support_first": first.get("support_rate"),
        "support_final": last.get("support_rate"),
        "citation_coverage": last.get("citation_coverage"),
        "revised": m["revised"],
        "seconds": m["seconds"],
    })
    return row


def _mean(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(values) if values else None


def aggregate(rows: list) -> dict:
    ok = [r for r in rows if r["ok"]]
    return {
        "topics": len(rows),
        "completed": len(ok),
        "avg_sources": _mean(ok, "sources"),
        "avg_claims": _mean(ok, "claims"),
        "avg_support_first": _mean(ok, "support_first"),
        "avg_support_final": _mean(ok, "support_final"),
        "avg_citation_coverage": _mean(ok, "citation_coverage"),
        "avg_seconds": _mean(ok, "seconds"),
        "revised_count": sum(1 for r in ok if r.get("revised")),
    }


def _pct(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


def _num(x, digits=1):
    return "n/a" if x is None else f"{x:.{digits}f}"


def write_outputs(rows: list, summary: dict, out_dir: str, depth: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"depth": depth, "summary": summary, "rows": rows}, f, indent=2)

    lines = [
        "# Evaluation results",
        "",
        f"Depth preset: **{depth}** · Topics completed: **{summary['completed']}/{summary['topics']}**",
        "",
        "| Topic | Sources | Claims | Support (1st draft) | Support (final) | Cited sentences | Revised | Time (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["ok"]:
            lines.append(
                f"| {r['topic']} | {r['sources']} | {r['claims']} | {_pct(r['support_first'])} | "
                f"{_pct(r['support_final'])} | {_pct(r['citation_coverage'])} | {'yes' if r['revised'] else 'no'} | {_num(r['seconds'])} |"
            )
        else:
            lines.append(f"| {r['topic']} | failed: {r.get('error', 'unknown')[:60]} | | | | | | |")
    lines += [
        "",
        "## Summary",
        "",
        f"- Average claim support: **{_pct(summary['avg_support_first'])}** in the first draft → **{_pct(summary['avg_support_final'])}** after the Checker/Reviser loop",
        f"- Average share of factual sentences with a citation: **{_pct(summary['avg_citation_coverage'])}**",
        f"- Average sources per report: **{_num(summary['avg_sources'])}**, claims checked per report: **{_num(summary['avg_claims'])}**",
        f"- Reports improved by the Reviser: **{summary['revised_count']}/{summary['completed']}**",
        f"- Average time per report: **{_num(summary['avg_seconds'])} s**",
        "",
        "_Support is judged by an automated LLM fact-checker against the passages each claim cites; "
        "it measures grounding in the retrieved sources, not absolute truth._",
    ]
    with open(os.path.join(out_dir, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ResearchMind on a list of topics.")
    parser.add_argument("--topics", default="evaluation/topics.txt")
    parser.add_argument("--depth", default="Standard", choices=["Quick", "Standard", "Deep"])
    parser.add_argument("--limit", type=int, default=None, help="only run the first N topics")
    parser.add_argument("--pause", type=float, default=8.0, help="seconds to wait between topics (API rate limits)")
    parser.add_argument("--out", default="evaluation")
    args = parser.parse_args()

    from src.pipeline import run_research

    with open(args.topics, encoding="utf-8") as f:
        topics = [line.strip() for line in f if line.strip()]
    if args.limit:
        topics = topics[: args.limit]

    rows = []
    for i, topic in enumerate(topics, 1):
        print(f"[{i}/{len(topics)}] {topic}")
        row = evaluate_topic(topic, args.depth, run_research)
        print("   ->", f"support {_pct(row['support_first'])} -> {_pct(row['support_final'])}, {row['seconds']}s" if row["ok"] else f"FAILED: {row.get('error')}")
        rows.append(row)
        if i < len(topics):
            time.sleep(args.pause)

    summary = aggregate(rows)
    write_outputs(rows, summary, args.out, args.depth)
    print(f"\nWrote {args.out}/results.json and {args.out}/RESULTS.md")


if __name__ == "__main__":
    main()
