from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    """Splits raw input text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter."""
    if not text or not isinstance(text, str) or not text.strip():
        print("[DEBUG WARNING] Empty or non-string input passed to split_text.")
        return []

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_text(text)
        return chunks

    except Exception as e:
        print(f"[DEBUG ERROR] Failed to split text: {e}")
        return []