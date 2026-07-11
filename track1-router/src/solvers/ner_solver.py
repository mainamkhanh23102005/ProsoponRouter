"""NER solver for official PERSON/ORG/LOCATION/DATE labels."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from src.task_utils import task_text


DATE_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|"
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}\b"
)
SPACY_LABELS = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "DATE": "DATE",
}
KNOWN_COMMON_ORGS = (
    "The United Nations",
    "Stanford University",
    "Google DeepMind",
    "Fireworks AI",
    "Anthropic",
    "Microsoft",
    "DeepMind",
    "OpenAI",
    "United Nations",
    "Amazon",
    "Apple",
    "Google",
    "Tesla",
    "Netflix",
    "IBM",
    "Nike",
    "Google",
    "Meta",
    "MIT",
)
KNOWN_COMMON_LOCATIONS = (
    "New York",
    "California",
    "Paris",
    "London",
    "Berlin",
    "Seattle",
    "Geneva",
    "Tokyo",
    "Dublin",
    "Germany",
    "Brussels",
    "Canada",
    "Melbourne",
    "Singapore",
    "Boston",
    "Hanoi",
)
PERSON_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
NON_PERSON_PHRASES = {
    "New York",
    "United Nations",
    "The United Nations",
    "Stanford University",
    "Google DeepMind",
    "Fireworks AI",
}


def solve(task: dict[str, Any]) -> tuple[str | None, float]:
    text = task_text(task)
    entities: list[tuple[int, int, dict[str, str]]] = []

    for match in DATE_PATTERN.finditer(text):
        entities.append((match.start(), match.end(), {"text": match.group(0), "label": "DATE"}))

    nlp = load_spacy_model()
    if nlp is not None:
        for ent in nlp(text).ents:
            label = SPACY_LABELS.get(ent.label_)
            if label is None or overlaps_existing(ent.start_char, ent.end_char, entities):
                continue
            entities.append((ent.start_char, ent.end_char, {"text": ent.text, "label": label}))

    apply_known_common_orgs(text, entities)
    apply_known_common_locations(text, entities)
    apply_person_patterns(text, entities)

    if not entities:
        return "NONE", 0.92
    entities.sort(key=lambda item: (item[0], item[1]))
    return format_entities([entity for _, _, entity in entities]), 0.92


@lru_cache(maxsize=1)
def load_spacy_model() -> Any | None:
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def overlaps_existing(start: int, end: int, entities: list[tuple[int, int, dict[str, str]]]) -> bool:
    return any(start < existing_end and end > existing_start for existing_start, existing_end, _ in entities)


def apply_known_common_orgs(text: str, entities: list[tuple[int, int, dict[str, str]]]) -> None:
    for org_name in KNOWN_COMMON_ORGS:
        pattern = re.compile(rf"(?<!\w){re.escape(org_name)}(?!\w)")
        for match in pattern.finditer(text):
            add_or_correct_known_org(match.start(), match.end(), org_name, entities)


def apply_known_common_locations(text: str, entities: list[tuple[int, int, dict[str, str]]]) -> None:
    for location in KNOWN_COMMON_LOCATIONS:
        pattern = re.compile(rf"(?<!\w){re.escape(location)}(?!\w)")
        for match in pattern.finditer(text):
            add_entity_if_clear(match.start(), match.end(), location, "LOCATION", entities)


def apply_person_patterns(text: str, entities: list[tuple[int, int, dict[str, str]]]) -> None:
    for match in PERSON_PATTERN.finditer(text):
        name = match.group(0)
        if name in NON_PERSON_PHRASES:
            continue
        if overlaps_existing(match.start(), match.end(), entities):
            continue
        add_entity_if_clear(match.start(), match.end(), name, "PERSON", entities)


def add_or_correct_known_org(
    start: int,
    end: int,
    org_name: str,
    entities: list[tuple[int, int, dict[str, str]]],
) -> None:
    for existing_start, existing_end, entity in entities:
        if existing_start == start and existing_end == end and entity["label"] == "ORG":
            return
        if start < existing_end and end > existing_start and entity["label"] == "ORG":
            if (existing_end - existing_start) >= (end - start):
                return

    entities[:] = [
        item
        for item in entities
        if not (start < item[1] and end > item[0])
    ]
    entities.append((start, end, {"text": org_name, "label": "ORG"}))


def add_entity_if_clear(
    start: int,
    end: int,
    text: str,
    label: str,
    entities: list[tuple[int, int, dict[str, str]]],
) -> None:
    if overlaps_existing(start, end, entities):
        return
    entities.append((start, end, {"text": text, "label": label}))


def format_entities(entities: list[dict[str, str]]) -> str:
    return "; ".join(f"{entity['label']}: {entity['text']}" for entity in entities)
