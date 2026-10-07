from graphrag.embeddings.embedder import HuggingFaceEmbedder
from graphrag.ingestion.chunker import Chunker, detect_language
from graphrag.models import LoadedPage


def _page(text: str) -> LoadedPage:
    return LoadedPage(doc_id="doc:test", source="t.txt", doc_type="txt", text=text)


def test_chunks_respect_token_limit_and_overlap(embedder: HuggingFaceEmbedder) -> None:
    text = " ".join(f"Sentence number {i} talks about vendors and clauses." for i in range(400))
    chunks = Chunker(100, 20, embedder.count_tokens).split([_page(text)])
    assert len(chunks) > 1
    assert all(c.token_count <= 100 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # overlap: the start of each chunk repeats the end of the previous one
    assert chunks[1].text.split()[0] in chunks[0].text


def test_chunk_ids_are_deterministic(embedder: HuggingFaceEmbedder) -> None:
    chunker = Chunker(100, 20, embedder.count_tokens)
    text = "Ganga Roadways builds highways. " * 50
    assert [c.chunk_id for c in chunker.split([_page(text)])] == [c.chunk_id for c in chunker.split([_page(text)])]


def test_hindi_splits_on_danda(embedder: HuggingFaceEmbedder) -> None:
    text = "गंगा रोडवेज़ ने राजमार्ग बनाया। " * 60
    chunks = Chunker(60, 0, embedder.count_tokens).split([_page(text)])
    assert len(chunks) > 1
    assert all(c.text.endswith("।") for c in chunks[:-1])
    assert chunks[0].language == "hi"


def test_detect_language() -> None:
    assert detect_language("The vendor must keep customer data in India.") == "en"
    assert detect_language("शक्ति स्टील वर्क्स को परिवीक्षा पर रखा गया।") == "hi"


def test_bullets_start_a_new_piece(embedder: HuggingFaceEmbedder) -> None:
    items = " ".join(f"⏩ Item {i}: vendor update about smart meters and data residency rules." for i in range(30))
    chunks = Chunker(60, 0, embedder.count_tokens).split([_page(items)])
    assert len(chunks) > 1
    assert all(c.text.startswith("⏩") for c in chunks)
