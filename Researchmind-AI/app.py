import os
import sys

# ======================================================
# 1. PATH RESOLUTION (Must run before any 'src' imports)
# ======================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

PARENT_DIR = os.path.dirname(APP_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# ======================================================
# 2. STANDARD LIBRARY IMPORTS
# ======================================================
import html
import re
import tempfile
import threading
import uuid

# ======================================================
# 3. THIRD-PARTY IMPORTS
# ======================================================
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

# ======================================================
# 4. LOCAL MODULE IMPORTS (src)
# ======================================================
from src.config import DEFAULT_DEPTH, DEPTH_PRESETS, LANGUAGES, MAX_UPLOAD_FILES, MODES
from src.embeddings import get_embedding_model
from src.pipeline import run_research

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Startup verification logs for keys
if not os.getenv("GROQ_API_KEY"):
    print("[WARNING] GROQ_API_KEY was not found in .env or environment variables.")
if not os.getenv("TAVILY_API_KEY"):
    print("[WARNING] TAVILY_API_KEY was not found in .env or environment variables.")

REPORTS_DIR = os.path.join(tempfile.gettempdir(), "researchmind_reports")

# ======================================================
# 5. PAGE & STYLES CONFIGURATION
# ======================================================
st.set_page_config(
    page_title="ResearchMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

custom_css = """
<style>
    /* Dark Theme Setup */
    .stApp {
        background-color: #0b0e14;
        color: #adbac7;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #2d333b;
    }

    /* Input Bar & Containers */
    .stTextInput input {
        background-color: #1c2128 !important;
        color: #ffffff !important;
        border: 1px solid #2d333b !important;
        border-radius: 10px !important;
    }

    /* Trace Box Styling */
    .trace-box {
        background: #0d1117;
        border: 1px solid #2d333b;
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 14px;
        margin-bottom: 12px;
    }

    /* RTL Support for Urdu */
    .rtl-text {
        direction: rtl;
        text-align: right;
    }

    /* Hide standard Streamlit header & footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ======================================================
# 6. EXPORT HELPERS (Markdown + PDF)
# ======================================================
def _pdf_safe(text: str) -> str:
    """Standard PDF fonts only cover Windows-1252, so drop emoji and unsupported symbols."""
    return text.encode("cp1252", "ignore").decode("cp1252")


def create_markdown(text: str):
    """Saves the report as a .md file."""
    if not text or len(text.strip()) < 10:
        return None
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, f"ResearchMind_Report_{uuid.uuid4().hex[:8]}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def create_pdf(text: str):
    """Builds a PDF from the markdown report."""
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
            if set(line) <= {"-", " "}:  # Horizontal rule
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
# 7. BACKGROUND WARM-UP
# ======================================================
@st.cache_resource
def warm_up_embeddings():
    """Loads the embedding model into cache in the background."""
    def _warm_up():
        try:
            get_embedding_model()
        except Exception as e:
            print(f"[DEBUG WARNING] Embedding model warm-up failed: {e}")
    
    threading.Thread(target=_warm_up, daemon=True).start()

warm_up_embeddings()


# ======================================================
# 8. SIDEBAR UI
# ======================================================
with st.sidebar:
    st.title("ResearchMind")
    
    if st.button("➕  New Research", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "1. 🧭 **Planner** splits your topic\n"
        "2. 🔍 **Researcher** searches the web\n"
        "3. 🧩 **Retriever** indexes & ranks passages\n"
        "4. ✍️ **Writer** drafts with citations\n"
        "5. 🔎 **Checker** verifies every claim\n"
        "6. 🛠️️ **Reviser** fixes weak claims"
    )


# ======================================================
# 9. MAIN CONTENT AREA
# ======================================================
col1, col2, col3 = st.columns(3)
with col1:
    mode = st.selectbox("Style", options=MODES, index=MODES.index("Standard") if "Standard" in MODES else 0)
with col2:
    depth = st.selectbox("Depth", options=list(DEPTH_PRESETS), index=list(DEPTH_PRESETS).index(DEFAULT_DEPTH) if DEFAULT_DEPTH in DEPTH_PRESETS else 0)
with col3:
    language = st.selectbox("Language", options=LANGUAGES, index=LANGUAGES.index("English") if "English" in LANGUAGES else 0)

query = st.text_input("What would you like to research?", placeholder="Ask anything...", key="user_query")
use_web = st.checkbox("🌐 Include web search", value=True)

with st.expander(f"📎 Add your own documents (PDF, TXT, MD — up to {MAX_UPLOAD_FILES})"):
    uploaded_files = st.file_uploader(
        "Your documents", 
        type=["pdf", "txt", "md"], 
        accept_multiple_files=True
    )

# Save uploaded files into temp storage for pipeline processing
temp_doc_paths = []
if uploaded_files:
    if len(uploaded_files) > MAX_UPLOAD_FILES:
        st.warning(f"Maximum allowed files is {MAX_UPLOAD_FILES}. Processing first {MAX_UPLOAD_FILES} files only.")
        uploaded_files = uploaded_files[:MAX_UPLOAD_FILES]
    
    for uf in uploaded_files:
        temp_path = os.path.join(tempfile.gettempdir(), uf.name)
        with open(temp_path, "wb") as f:
            f.write(uf.getbuffer())
        temp_doc_paths.append(temp_path)

# ======================================================
# 10. EXECUTION & STREAMING LOGIC
# ======================================================
if st.button("Start Research 🚀", type="primary", use_container_width=True):
    if not query or len(query.strip()) < 2:
        st.warning("Please enter a valid research topic.")
    else:
        trace_container = st.empty()
        report_container = st.empty()
        downloads_container = st.container()

        rtl = (language == "Urdu")
        
        try:
            for event in run_research(
                query.strip(), mode, language, depth, use_web, temp_doc_paths
            ):
                # Update trace output live
                if event.get("trace"):
                    trace_container.markdown(f'<div class="trace-box">{event["trace"]}</div>', unsafe_allow_html=True)
                
                # Update report content live
                if event.get("report"):
                    if rtl:
                        report_container.markdown(f'<div class="rtl-text">{event["report"]}</div>', unsafe_allow_html=True)
                    else:
                        report_container.markdown(event["report"])
                
                # Render file export buttons when execution completes
                if event.get("done"):
                    final_report = event["result"]["report"]
                    files = export_files(final_report, language)
                    
                    with downloads_container:
                        st.markdown("### 📥 Download Reports")
                        d_cols = st.columns(len(files))
                        for idx, file_path in enumerate(files):
                            file_name = os.path.basename(file_path)
                            mime_type = "application/pdf" if file_name.endswith(".pdf") else "text/markdown"
                            
                            with open(file_path, "rb") as f:
                                d_cols[idx].download_button(
                                    label=f"Download {file_name.split('.')[-1].upper()}",
                                    data=f.read(),
                                    file_name=file_name,
                                    mime=mime_type,
                                    key=f"download_{idx}"
                                )

                if event.get("error"):
                    st.error("An error occurred during execution.")
                    break

        except Exception as e:
            st.error(f"### ❌ An Error Occurred:\n`{str(e)}`")