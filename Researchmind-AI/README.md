# 🔬 ResearchMind AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-UI-FF6F00?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLM-000000?style=for-the-badge)
![RAG](https://img.shields.io/badge/RAG-Retrieval--Augmented--Generation-6A5ACD?style=for-the-badge)
![Agentic](https://img.shields.io/badge/Agentic-Plan%20%E2%86%92%20Write%20%E2%86%92%20Verify-E91E63?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-009688?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-26%20passing-2E7D32?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

### A multi-agent research assistant that plans, researches, writes, and **fact-checks its own reports**

</div>

---

## 📖 Overview

Most AI research tools return fluent text you have to trust blindly. ResearchMind AI is built around a different idea: **every claim in the report is cited, and a second agent checks each claim against the exact passages it cites** before you see the result.

Give it a topic and a team of cooperating agents does the work:

| Agent | Job |
|---|---|
| 🧭 **Planner** | Breaks the topic into focused research sub-questions |
| 🔍 **Researcher** | Searches the web for every sub-question in parallel (Tavily) |
| 🧩 **Retriever** | Chunks and embeds all sources, then ranks the best passages with FAISS cosine search |
| ✍️ **Writer** | Drafts a report that may only use the retrieved sources, citing them as `[1]`, `[2]` |
| 🔎 **Checker** | Verifies each cited claim against the passages it cites: *supported / partial / unsupported* |
| 🛠️ **Reviser** | Rewrites or removes flagged claims, then the Checker re-verifies (the feedback loop) |
| 🌍 **Translator** | Optionally renders the *verified* English report in Urdu |

The UI shows this activity live: the plan, sources found, passages retrieved, and the fact-check verdicts.

---

## ✨ Features

- **Agentic pipeline** with a self-correcting verify → revise loop
- **Real RAG**: per-request in-memory FAISS index (normalized embeddings, cosine similarity), per-source caps so one page cannot dominate, relevance floor to drop weak passages
- **Inline citations** `[n]` with a sources list built by code (URLs are never invented by the model); invented citation numbers are stripped automatically
- **Claim-level verification** with an honest score shown on every report
- **Bring your own documents**: upload PDF / TXT / MD, mix them with web results, or run on your documents only
- **Research depth** presets: Quick, Standard, Deep
- **Report styles**: Standard, Summary, Academic
- **English and Urdu** output (Urdu is translated *after* verification, so the checking happens on the original English text)
- **Export** to Markdown (all languages) and PDF (English) with clickable source links
- **Model fallback and rate-limit retry** across several Groq models
- **Offline test suite** (26 tests) and a **batch evaluation script**

---

## 🏗️ Architecture

```text
                    Topic + options (style, depth, language, documents)
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │  Planner (fast LLM)    │──► 2–5 sub-questions
                         └───────────┬────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
                  │ Researcher: parallel Tavily search  │  + uploaded PDF/TXT/MD
                  └───────────────────┬─────────────────┘
                                      ▼
                  ┌─────────────────────────────────────┐
                  │ Retriever: chunk → embed (MiniLM)   │
                  │ → FAISS cosine index → top passages │
                  │   per sub-question                  │
                  └───────────────────┬─────────────────┘
                                      ▼
                         ┌────────────────────────┐
                         │ Writer (70B, streaming)│──► cited draft
                         └───────────┬────────────┘
                                     ▼
              ┌─────────────────────────────────────────────┐
              │ Checker: each claim vs. the passages it cites│
              └───────────┬─────────────────────────────────┘
                          │ flagged claims?
                  yes ┌───┴───┐ no
                      ▼       │
              ┌──────────────┐│
              │   Reviser    ││   (re-check; keep the revision only if it scores better)
              └──────┬───────┘│
                     └────────┤
                              ▼
                  Translator (optional) → Report + Sources + Verification
```

---

## 🧰 Tech Stack

Python · Gradio · Groq (Llama 3.3 70B for writing, Llama 3.1 8B for planning and checking, with fallbacks) · Tavily · Sentence-Transformers (`all-MiniLM-L6-v2`) · FAISS · LangChain text splitters · pypdf · ReportLab

---

## 🚀 Getting Started

```bash
git clone https://github.com/muhammadazfar-ai/ResearchMind-AI.git
cd ResearchMind-AI
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # then add your keys
python app.py
```

You need a free [Groq](https://console.groq.com) key and a [Tavily](https://tavily.com) key. Never commit `.env`.

### Run on Hugging Face Spaces
1. Create a new **Gradio** Space and push this repository.
2. In *Settings → Variables and secrets*, add `GROQ_API_KEY` and `TAVILY_API_KEY` as **secrets**.
3. The Space installs `requirements.txt` and runs `app.py` automatically.

---

## 📊 Evaluation

Run the pipeline over a topic list and measure how well-grounded the reports are:

```bash
python evaluate.py --limit 5            # first 5 topics in evaluation/topics.txt
python evaluate.py --depth Quick --pause 15
```

It writes `evaluation/results.json` and `evaluation/RESULTS.md` with, per topic and on average:
- **Claim support rate**: share of cited claims fully supported by their sources, for the first draft and after the Checker/Reviser loop
- **Citation coverage**: share of factual sentences that carry a citation
- Sources per report, claims checked, and time per report

> Paste your own `RESULTS.md` summary here after running it, so the numbers in this README are yours.

---

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The suite runs fully offline with scripted fake search, embedding and LLM components. It covers retrieval ranking, citation cleaning, claim extraction, the full verify → revise loop (including a planted false claim that must be caught and removed), document loading, Urdu translation flow, error paths, and the UI handler.

---

## ⚙️ Configuration

All tuning lives in [`src/config.py`](src/config.py): chunk size, depth presets, how many passages go to the Writer, relevance thresholds, fact-check batch size. Model order per role is in [`src/llm.py`](src/llm.py) and can be overridden with `GROQ_WRITER_MODEL` / `GROQ_FAST_MODEL`.

---

## ⚠️ Limitations

- **The fact-check measures grounding, not absolute truth.** An LLM judges whether a claim matches the passages it cites. If a source is wrong, the report can still be "supported". Verify critical facts yourself.
- **Free-tier rate limits.** Deep research makes many API calls; the app retries and falls back to other models, but it can still be slow or fail under tight limits.
- **Urdu PDF export is not supported** (standard PDF fonts cannot shape Urdu). Urdu reports download as Markdown.
- **Scanned PDFs** have no text layer and need OCR before upload.
- The Urdu translation is machine-generated and was not reviewed by a native speaker.

---

## 📂 Project Structure

```text
ResearchMind-AI/
├── app.py                 # Gradio UI: controls, live agent log, exports
├── evaluate.py            # Batch evaluation (support rate, coverage, timing)
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .env.example
├── evaluation/
│   └── topics.txt         # Topics used by evaluate.py
├── src/
│   ├── agents.py          # Planner, Researcher, Writer, Checker, Reviser, Translator
│   ├── pipeline.py        # Orchestration + live activity trace
│   ├── evidence.py        # Chunking, FAISS index, passage selection per question
│   ├── embeddings.py      # Sentence-Transformers + cosine index
│   ├── search.py          # Tavily web search
│   ├── documents.py       # PDF / TXT / MD loading
│   ├── llm.py             # Groq client, model fallback, streaming
│   ├── prompts.py         # All agent prompts
│   ├── chunking.py        # Recursive text splitter
│   └── config.py          # Tunable settings
├── tests/                 # Offline test suite
└── screenshot/
```

---

## 📄 License

MIT. See [LICENSE](LICENSE).
