import hashlib
import json
import re
import types

import numpy as np
import pytest


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder, so similarity reflects shared words (no model download)."""

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        out = np.zeros((len(texts), 256), dtype="float32")
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                word = re.sub(r"(ing|s)$", "", word)  # crude stemming: drone/drones, monitor/monitoring
                out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1
        if normalize_embeddings:
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out


WEB = {
    "solar": [
        {"title": "Agrivoltaics explained", "url": "https://a.example/agri",
         "content": "Agrivoltaics combines solar panels with crops on the same farmland. It improves land use and farm income. " * 10},
    ],
    "drone": [
        {"title": "Farm drones", "url": "https://b.example/drones",
         "content": "Drones monitor crops on farmland and help farmers detect plant disease early using imaging. " * 10},
        {"title": "Cooking pasta", "url": "https://c.example/pasta",
         "content": "Boil water, add salt, cook pasta for ten minutes and serve with sauce. " * 8},
    ],
}


def fake_searcher(question, max_results=4, search_depth="basic"):
    results = []
    for key, items in WEB.items():
        if key in question.lower():
            results += items
    return results or WEB["solar"]


class FakeLLM:
    """Scripted stand-in for Groq. Role is detected from the system prompt."""

    def __init__(self, bad_claim="The moon is made of cheese"):
        self.bad_claim = bad_claim
        self.calls = []

    def complete(self, messages, role="fast", temperature=0.0, json_mode=False):
        system, user = messages[0]["content"], messages[-1]["content"]
        self.calls.append(system.split(".")[0])
        if "Planner" in system:
            return json.dumps({"sub_questions": ["solar panels on farmland", "drone crop monitoring"]})
        if "fact-checker" in system:
            results = []
            for match in re.finditer(r"CLAIM (\d+): (.*)", user):
                verdict = "unsupported" if self.bad_claim.lower() in match.group(2).lower() else "supported"
                results.append({"id": int(match.group(1)), "verdict": verdict, "reason": "scripted"})
            return json.dumps({"results": results})
        if "Reviser" in system:
            report = user.split("A fact-checker flagged")[0].replace("REPORT:\n", "")
            return "\n".join(l for l in report.split("\n") if self.bad_claim.lower() not in l.lower())
        if "translator" in system:
            return "[UR] " + user.split("\n\n", 1)[1]
        raise AssertionError("unexpected prompt: " + system)

    def stream(self, messages, role="writer", temperature=0.3):
        self.calls.append("writer")
        text = (
            "## Key Findings\n"
            "Solar panels and crops can share the same farmland [1]. "
            "Drones help farmers detect plant disease early [2][9].\n"
            f"{self.bad_claim} [1].\n"
            "## Conclusion\n"
            "Both technologies improve how farmland is used [1, 2].\n"
        )
        for i in range(0, len(text), 25):
            yield text[i : i + 25]


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_llm():
    llm = FakeLLM()
    return types.SimpleNamespace(complete=llm.complete, stream=llm.stream, raw=llm)
