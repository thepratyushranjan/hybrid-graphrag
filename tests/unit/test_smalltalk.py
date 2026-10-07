import pytest

from graphrag.generation.smalltalk import detect_smalltalk, smalltalk_reply


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("Hi", "greeting"), ("hello!", "greeting"), ("Hey there 👋", "greeting"), ("Good morning", "greeting"),
        ("नमस्ते 🙏", "greeting"), ("thanks", "thanks"), ("Thank you so much!", "thanks"), ("धन्यवाद", "thanks"),
        ("bye", "goodbye"), ("What can you do?", "help"), ("आप कौन हैं?", "help"),
    ],
)
def test_detects_smalltalk(message: str, kind: str) -> None:
    assert detect_smalltalk(message) == kind


@pytest.mark.parametrize(
    "message",
    ["Hi, who leads Ganga Roadways?", "hello world revenue 2025", "Who are the vendors?", "help me find Clause 7.2",
     "Thanks, and which vendors are impacted by Clause 7.2?", ""],
)
def test_real_questions_are_not_smalltalk(message: str) -> None:
    assert detect_smalltalk(message) is None


def test_reply_mentions_documents_in_the_right_language() -> None:
    assert "(3 documents)" in smalltalk_reply("greeting", "en", 3)
    assert smalltalk_reply("greeting", "hi", 3).startswith("नमस्ते")
