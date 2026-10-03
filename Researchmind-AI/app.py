import html
import os
import re
import tempfile
import threading
import uuid

from dotenv import find_dotenv, load_dotenv

# Search up the directory tree to find .env and load environment variables
load_dotenv(find_dotenv(), override=True)

# Startup verification logs for keys
if not os.getenv("GROQ_API_KEY"):
    print("[WARNING] GROQ_API_KEY was not found in .env or environment variables.")
if not os.getenv("TAVILY_API_KEY"):
    print("[WARNING] TAVILY_API_KEY was not found in .env or environment variables.")

import gradio as gr
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from src.config import DEFAULT_DEPTH, DEPTH_PRESETS, LANGUAGES, MAX_UPLOAD_FILES, MODES
from src.embeddings import get_embedding_model
from src.pipeline import run_research

REPORTS_DIR = os.path.join(tempfile.gettempdir(), "researchmind_reports")


# ======================================================
# EXPORT (Markdown + PDF)
# ======================================================
def _pdf_safe(text: str) -> str:
    """Standard PDF fonts only cover Windows-1252, so drop emoji and other unsupported symbols."""
    return text.encode("cp1252", "ignore").decode("cp1252")


def create_markdown(text: str):
    """Saves the report as a .md file (works for every language)."""
    if not text or len(text.strip()) < 10:
        return None
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, f"ResearchMind_Report_{uuid.uuid4().hex[:8]}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def create_pdf(text: str):
    """Builds a PDF from the markdown report. English/Latin text only (see README for Urdu)."""
    if not text or len(text.strip()) < 10:
        return None
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        path = os.path.join(REPORTS_DIR, f"ResearchMind_Report_{uuid.uuid4().hex[:8]}.pdf")
        styles = getSampleStyleSheet()
        story = []
        for raw in text.split("\n"):
            line = _pdf_safe(raw).strip()
            if not line:
                story.append(Spacer(1, 8))
                continue
            if set(line) <= {"-", " "}:  # horizontal rule
                story.append(Spacer(1, 6))
                continue

            style, bullet = styles["BodyText"], None
            heading = re.match(r"^(#+)\s*(.*)", line)
            if heading:
                style = styles["Heading1"] if len(heading.group(1)) == 1 else styles["Heading2"]
                line = heading.group(2)
            elif line.startswith(">"):
                style, line = styles["Italic"], line.lstrip("> ").strip()
            elif re.match(r"^[-*]\s+", line):
                bullet, line = "\u2022", re.sub(r"^[-*]\s+", "", line)

            line = html.escape(line)
            line = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)
            line = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2" color="blue">\1</a>', line)
            story.append(Paragraph(line, style, bulletText=bullet))

        SimpleDocTemplate(path, pagesize=LETTER).build(story)
        return path
    except Exception as e:
        print(f"PDF Generation Error: {e}")
        return None


def export_files(report: str, language: str) -> list:
    files = [create_markdown(report)]
    if language == "English":
        files.append(create_pdf(report))
    return [f for f in files if f]


# ======================================================
# CORE HANDLER
# ======================================================
def run_agent(topic, mode, language, depth, use_web, doc_files):
    """Streams the agent pipeline into the UI: live activity log, report, then downloadable files."""
    if not topic or len(topic.strip()) < 2:
        yield (
            gr.update(visible=True),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value=None, visible=False),
        )
        return

    rtl = language == "Urdu"
    try:
        for event in run_research(
            topic.strip(), mode, language, depth, use_web, [getattr(f, "name", f) for f in (doc_files or [])]
        ):
            report_update = gr.update() if event["report"] is None else gr.update(value=event["report"], visible=True, rtl=rtl)
            files_update = gr.update(value=None, visible=False)
            if event["done"]:
                files = export_files(event["result"]["report"], language)
                files_update = gr.update(value=files, visible=bool(files))
            yield gr.update(visible=False), gr.update(value=event["trace"], visible=True), report_update, files_update
            if event["error"]:
                return
    except Exception as e:
        yield (
            gr.update(visible=False),
            gr.update(),
            gr.update(value=f"### ❌ An Error Occurred:\n`{str(e)}`", visible=True),
            gr.update(value=None, visible=False),
        )


def reset_ui():
    """Resets the UI state for starting a new research session."""
    return (
        "",                                    # clear search input
        gr.update(visible=True),               # show welcome text
        gr.update(value="", visible=False),    # clear activity log
        gr.update(value="", visible=False),    # clear report
        gr.update(value=None, visible=False),  # clear downloads
    )


# ======================================================
# STYLES
# ======================================================
custom_css = """
body, .gradio-container { 
    background-color: #0b0e14 !important; 
    color: #adbac7 !important; 
    font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}
footer { display: none !important; }

.sidebar-gpt { 
    background-color: #0d1117 !important; 
    border-right: 1px solid #2d333b !important; 
    padding: 16px !important; 
}
.sidebar-btn { 
    background: transparent !important;
    border: 1px solid #2d333b !important;
    color: #adbac7 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 10px 14px !important; 
    border-radius: 10px !important; 
    cursor: pointer !important; 
    font-size: 14px !important;
    margin-bottom: 8px !important;
    width: 100% !important;
}
.sidebar-btn:hover { 
    background: #1c2128 !important; 
    color: #ffffff !important; 
}

.chat-container { 
    max-width: 850px !important; 
    margin: 0 auto !important; 
    padding-bottom: 40px !important; 
}
.welcome-text { 
    font-size: 32px; 
    font-weight: 700; 
    text-align: center; 
    margin-top: 40px; 
    color: #ffffff; 
}

.report-gpt { 
    color: #adbac7 !important; 
    font-size: 16px !important; 
    line-height: 1.7 !important; 
}
.report-gpt h1, .report-gpt h2, .report-gpt h3 { 
    color: #ffffff !important; 
}

.input-bar-gpt { 
    background: #1c2128 !important; 
    border: 1px solid #2d333b !important; 
    border-radius: 16px !important; 
    padding: 8px 16px !important; 
    align-items: center !important;
}

.trace-box {
    background: #0d1117 !important;
    border: 1px solid #2d333b !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
    margin-bottom: 12px !important;
}

.send-btn { 
    background: #10a37f !important; 
    color: white !important; 
    border-radius: 50% !important; 
    min-width: 40px !important;
    height: 40px !important;
    border: none !important;
}
"""

# ======================================================
# UI STRUCTURE
# ======================================================
with gr.Blocks(title="ResearchMind AI") as app:
    with gr.Row():
        # LEFT SIDEBAR
        with gr.Column(scale=1, elem_classes="sidebar-gpt", min_width=240):
            gr.HTML("""<div style="font-size:20px; font-weight:800; margin-bottom:20px; color:#ffffff;">ResearchMind</div>""")
            new_research_btn = gr.Button("➕  New Research", elem_classes="sidebar-btn")
            gr.Markdown(
                "**How it works**\n\n"
                "1. 🧭 **Planner** splits your topic\n"
                "2. 🔍 **Researcher** searches the web\n"
                "3. 🧩 **Retriever** indexes & ranks passages\n"
                "4. ✍️ **Writer** drafts with citations\n"
                "5. 🔎 **Checker** verifies every claim\n"
                "6. 🛠️ **Reviser** fixes weak claims\n"
            )

        # MAIN AREA
        with gr.Column(scale=4):
            with gr.Row():
                mode = gr.Dropdown(choices=MODES, value="Standard", label="Style", scale=2)
                depth = gr.Dropdown(choices=list(DEPTH_PRESETS), value=DEFAULT_DEPTH, label="Depth", scale=2)
                language = gr.Dropdown(choices=LANGUAGES, value="English", label="Language", scale=2)

            with gr.Column(elem_classes="chat-container"):
                welcome = gr.HTML('<div class="welcome-text">What would you like to research?</div>')
                trace_out = gr.Markdown(visible=False, elem_classes="trace-box")
                report_out = gr.Markdown(visible=False, elem_classes="report-gpt")
                files_out = gr.File(label="Download report", file_count="multiple", visible=False)

            with gr.Row(elem_classes="input-bar-gpt"):
                query = gr.Textbox(placeholder="Ask anything...", lines=1, container=False, scale=10)
                submit_btn = gr.Button("↑", variant="primary", elem_classes="send-btn", scale=1)

            with gr.Row():
                use_web = gr.Checkbox(label="🌐 Include web search", value=True)
            with gr.Accordion(f"📎 Add your own documents (PDF, TXT, MD — up to {MAX_UPLOAD_FILES})", open=False):
                doc_files = gr.File(file_count="multiple", file_types=[".pdf", ".txt", ".md"], label="Your documents")

    # Event handlers
    run_inputs = [query, mode, language, depth, use_web, doc_files]
    run_outputs = [welcome, trace_out, report_out, files_out]

    submit_event = submit_btn.click(fn=run_agent, inputs=run_inputs, outputs=run_outputs)
    submit_event.then(fn=lambda: "", outputs=[query])

    query_event = query.submit(fn=run_agent, inputs=run_inputs, outputs=run_outputs)
    query_event.then(fn=lambda: "", outputs=[query])

    new_research_btn.click(fn=reset_ui, outputs=[query, welcome, trace_out, report_out, files_out])


def _warm_up():
    """Loads the embedding model in the background so the first search is not slow."""
    try:
        get_embedding_model()
    except Exception as e:
        print(f"[DEBUG WARNING] Embedding model warm-up failed: {e}")

if __name__ == "__main__":
    threading.Thread(target=_warm_up, daemon=True).start()
    app.queue(default_concurrency_limit=3)
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        inbrowser=not os.getenv("SPACE_ID"),
        css=custom_css
    )