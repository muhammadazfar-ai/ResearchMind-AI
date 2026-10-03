"""Central settings for ResearchMind. Tune the pipeline here."""

# --- chunking -------------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# --- research depth presets (shown in the UI) -----------------------------
DEPTH_PRESETS = {
    "Quick":    {"sub_questions": 2, "results_per_question": 3, "search_depth": "basic",    "chunks_per_question": 4},
    "Standard": {"sub_questions": 3, "results_per_question": 4, "search_depth": "advanced", "chunks_per_question": 5},
    "Deep":     {"sub_questions": 5, "results_per_question": 4, "search_depth": "advanced", "chunks_per_question": 5},
}
DEFAULT_DEPTH = "Standard"

# --- evidence selection ---------------------------------------------------
MAX_EVIDENCE_CHUNKS = 20          # total passages given to the Writer
MAX_CHUNKS_PER_SOURCE = 4         # across the whole report, so no single page dominates
MAX_PER_SOURCE_PER_QUESTION = 2   # per sub-question
MIN_SCORE = 0.2                   # cosine floor for a passage to count as relevant
MIN_KEEP_PER_QUESTION = 2         # always keep this many passages per sub-question

# --- fact-checking --------------------------------------------------------
CHECK_BATCH_SIZE = 5
MAX_CLAIMS_CHECKED = 40
MAX_REVISION_ROUNDS = 1

# --- UI options -----------------------------------------------------------
MODES = ["Standard", "Summary", "Academic"]
LANGUAGES = ["English", "Urdu"]

# --- uploaded documents ---------------------------------------------------
MAX_UPLOAD_FILES = 5
MAX_CHARS_PER_DOCUMENT = 150_000
