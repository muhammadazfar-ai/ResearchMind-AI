# ---------------------------------------------------------------- Planner
SYSTEM_PLANNER = (
    "You are the Planner of a research agent. You break a topic into focused research sub-questions "
    "and reply with JSON only."
)


def build_planner_prompt(topic: str, n: int) -> str:
    return (
        f"Topic: {topic}\n\n"
        f"Break this topic into exactly {n} specific, non-overlapping research sub-questions that together cover it "
        "(for example: background, current state, evidence or data, challenges or risks, outlook). "
        "Each sub-question must be a self-contained web-search query written in English, at most 15 words.\n\n"
        'Return ONLY JSON in this form: {"sub_questions": ["...", "..."]}'
    )


# ----------------------------------------------------------------- Writer
SYSTEM_WRITER = (
    "You are ResearchMind, a careful research writer. "
    "You write reports using ONLY the numbered sources provided by the user. "
    "Every factual claim must end with a citation such as [1] or [2][3]. "
    "If the sources do not cover something, say so explicitly instead of filling the gap from memory. "
    "Never invent sources, URLs, statistics or quotes. "
    "Do not write a references list; it is added automatically."
)

MODE_INSTRUCTIONS = {
    "Standard": (
        "Write a clear, structured report with markdown sections such as Introduction, Key Findings, "
        "Current Developments, Challenges, Outlook and Conclusion. Use short paragraphs and bullet points where helpful."
    ),
    "Summary": (
        "Write a concise summary: a 2-3 sentence overview followed by 5-7 bullet points "
        "covering the most important findings."
    ),
    "Academic": (
        "Write in a formal academic tone with sections such as Abstract, Introduction, Background, Analysis, "
        "Discussion, Limitations and Conclusion. Be precise, avoid hype, and note where sources disagree or are limited."
    ),
}


def build_writer_prompt(topic: str, mode: str, questions: list, context: str, num_sources: int) -> str:
    outline = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
    return (
        f"Topic: {topic}\n\n"
        f"Research questions to cover:\n{outline}\n\n"
        f"Sources:\n{context}\n\n"
        f"Task: {MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS['Standard'])} "
        "Make sure the report addresses each research question.\n\n"
        f"Rules: cite only with the numbers 1 to {num_sources}, placed right after the claim they support. "
        "Use only information found in the sources above."
    )


# ---------------------------------------------------------------- Checker
SYSTEM_CHECKER = (
    "You are a strict fact-checker. You judge whether a claim is supported by the evidence quoted from "
    "its cited sources. Use only the evidence given; never use outside knowledge. Reply with JSON only."
)


def build_check_prompt(items: list) -> str:
    blocks = []
    for item in items:
        evidence = "\n".join(f"- {e}" for e in item["evidence"]) or "- (no evidence available)"
        blocks.append(f"CLAIM {item['id']}: {item['text']}\nEVIDENCE {item['id']}:\n{evidence}")
    return (
        "Judge each claim against its evidence.\n"
        'Verdicts: "supported" (the evidence clearly states or directly implies the claim), '
        '"partial" (the evidence supports only part of the claim, or is vaguer than the claim), '
        '"unsupported" (the evidence does not contain the claim or contradicts it).\n\n'
        'Return ONLY JSON: {"results": [{"id": 1, "verdict": "supported", "reason": "max 15 words"}]}\n\n'
        + "\n\n".join(blocks)
    )


# ---------------------------------------------------------------- Reviser
SYSTEM_REVISER = (
    "You are the Reviser of a research agent. You fix a report so that every claim is strictly backed "
    "by the evidence provided."
)


def build_revise_prompt(report: str, issues: list) -> str:
    blocks = []
    for i, issue in enumerate(issues, 1):
        evidence = "\n".join(f"  - {e}" for e in issue["evidence"]) or "  - (none)"
        blocks.append(
            f"{i}. Claim: {issue['text']}\n   Fact-check verdict: {issue['verdict']} ({issue['reason']})\n"
            f"   Evidence from the cited source:\n{evidence}"
        )
    return (
        f"REPORT:\n{report}\n\n"
        "A fact-checker flagged these claims:\n\n" + "\n\n".join(blocks) + "\n\n"
        "For each flagged claim either (a) delete it, or (b) rewrite it so it states ONLY what the evidence supports. "
        "Do not add new facts or new sources. Keep the headings, structure and [n] citation markers of everything else. "
        "Return the complete corrected report and nothing else."
    )


# ------------------------------------------------------------- Translator
SYSTEM_TRANSLATOR = "You are a professional translator who preserves formatting exactly."


def build_translate_prompt(report: str, language: str) -> str:
    return (
        f"Translate the following research report into {language}. "
        "Keep the markdown structure (headings, bullets). Keep citation markers such as [1] exactly as they are. "
        "Do not translate URLs. Return only the translation.\n\n" + report
    )
