from graphrag.ingestion.normalizer import normalize_text


def test_expands_pdf_ligatures() -> None:
    assert normalize_text("Group Compliance Oﬃce") == "Group Compliance Office"


def test_removes_zero_width_but_keeps_indic_joiners() -> None:
    assert normalize_text("a​b") == "ab"
    assert "‍" in normalize_text("क्‍ष")  # ZWJ changes how Devanagari renders; keep it


def test_collapses_whitespace() -> None:
    assert normalize_text("  one   two\n\n\n\nthree  ") == "one two\n\nthree"
