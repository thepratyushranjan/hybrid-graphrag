"""Greetings, thanks, goodbyes and "what can you do?" are answered directly: no retrieval, no LLM, no evidence.
(Vector search always returns *some* nearest chunks, so "Hi" would otherwise be answered from documents.)"""

import re
import unicodedata
from typing import Literal

SmallTalk = Literal["greeting", "thanks", "goodbye", "help"]

_PHRASES: dict[SmallTalk, set[str]] = {
    "greeting": {
        "hi", "hii", "hiii", "hello", "helo", "hey", "heya", "yo", "hola", "hi there", "hello there", "hey there",
        "good morning", "good afternoon", "good evening", "namaste", "namaskar",
        "नमस्ते", "नमस्कार", "हेलो", "हैलो", "हाय", "प्रणाम",
    },
    "thanks": {
        "thanks", "thank you", "thankyou", "thx", "ty", "thanks a lot", "thank you so much", "great thanks",
        "ok thanks", "okay thanks", "धन्यवाद", "शुक्रिया", "थैंक्स", "थैंक यू",
    },
    "goodbye": {"bye", "goodbye", "good bye", "see you", "see ya", "cya", "अलविदा", "बाय", "फिर मिलेंगे"},
    "help": {
        "help", "what can you do", "who are you", "what are you", "how do you work", "how does this work",
        "what can i ask", "आप कौन हैं", "तुम कौन हो", "आप क्या कर सकते हैं", "मदद",
    },
}
_SPACES = re.compile(r"\s+")


def _strip_symbols(text: str) -> str:
    """Drop punctuation and symbols/emoji (Unicode P* / S*), keeping letters *and* combining marks:
    a regex \\w class would cut Devanagari vowel signs out of words like "नमस्ते"."""
    return "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)

REPLIES: dict[str, dict[SmallTalk, str]] = {
    "en": {
        "greeting": "Hi! I answer questions about the documents in the knowledge base{docs}, citing the text "
                    "excerpts [C#] and knowledge-graph facts [G#] I used. Ask about people, organisations, "
                    "how things are connected, counts, or what happened when.",
        "thanks": "You're welcome! Ask me anything else about your documents.",
        "goodbye": "Goodbye! Come back any time.",
        "help": "I search your documents{docs} two ways at once - semantic search over the text (Qdrant) and a "
                "knowledge graph of the entities and relationships in them (Neo4j) - and answer with citations. "
                "Try questions like \"Who leads X?\", \"How is X connected to Y?\", \"Which X has the most Y?\" "
                "or \"What happened in October 2025?\". Upload PDF, Markdown or text files in the sidebar.",
    },
    "hi": {
        "greeting": "नमस्ते! मैं नॉलेज बेस के दस्तावेज़ों{docs} से जुड़े सवालों के जवाब देता हूँ और इस्तेमाल किए गए "
                    "स्रोत [C#] और ग्राफ़ तथ्य [G#] बताता हूँ। लोगों, संगठनों, उनके संबंधों या तारीख़ों के बारे में पूछिए।",
        "thanks": "आपका स्वागत है! दस्तावेज़ों के बारे में कुछ और पूछिए।",
        "goodbye": "अलविदा! फिर मिलेंगे।",
        "help": "मैं आपके दस्तावेज़ों{docs} में दो तरह से खोजता हूँ - टेक्स्ट की सिमैंटिक खोज और एंटिटी-संबंधों का नॉलेज "
                "ग्राफ़ - और स्रोतों के साथ जवाब देता हूँ। जैसे पूछें: \"X का प्रमुख कौन है?\" या \"X और Y कैसे जुड़े हैं?\"",
    },
}


def detect_smalltalk(message: str) -> SmallTalk | None:
    """The whole message must be small talk ("hi", "thanks!", "नमस्ते 🙏"); "hi, who leads X?" is a question."""
    text = _SPACES.sub(" ", _strip_symbols(unicodedata.normalize("NFC", message.casefold()))).strip()
    if not text or len(text.split()) > 6:
        return None
    for kind, phrases in _PHRASES.items():
        if text in phrases:
            return kind
    return None


def smalltalk_reply(kind: SmallTalk, language: str, documents: int | None) -> str:
    replies = REPLIES.get(language, REPLIES["en"])
    if language == "hi":
        docs = f" ({documents})" if documents else ""
    else:
        docs = f" ({documents} document{'s' if documents != 1 else ''})" if documents else ""
    return replies[kind].format(docs=docs)
