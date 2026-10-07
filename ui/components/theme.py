"""Colors for the UI's charts, light and dark (validated categorical palette, fixed order)."""

import streamlit as st

# Entity types in a fixed order -> fixed palette slots (never re-assigned by rank or count).
# Five types exceed the 3-slot all-pairs limit, so every type also gets its own node shape.
ENTITY_TYPES = ["Person", "Organization", "Location", "Event", "Concept"]
SHAPES = {"Person": "dot", "Organization": "square", "Location": "triangle", "Event": "diamond", "Concept": "star"}
_SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
}
_INK = {
    "light": {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8984", "surface": "#fcfcfb", "edge": "#b9b8b3"},
    "dark": {"primary": "#ffffff", "secondary": "#c3c2b7", "muted": "#8f8e86", "surface": "#1a1a19", "edge": "#5c5b56"},
}


def mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except AttributeError:  # older Streamlit
        return "light"


def type_color(entity_type: str | None) -> str:
    series = _SERIES[mode()]
    return series[ENTITY_TYPES.index(entity_type)] if entity_type in ENTITY_TYPES else _INK[mode()]["muted"]


def ink(role: str) -> str:
    return _INK[mode()][role]


def series_1() -> str:
    return _SERIES[mode()][0]
