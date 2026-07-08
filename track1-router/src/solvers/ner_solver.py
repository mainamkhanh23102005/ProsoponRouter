"""NER solver for official PERSON/ORG/LOCATION/DATE labels."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from src.task_utils import task_text


DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
SPACY_LABELS = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
}


def solve(task: dict[str, Any]) -> tuple[list[dict[str, str]] | None, float]:
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

    if not entities:
        return [], 0.92
    entities.sort(key=lambda item: (item[0], item[1]))
    return [entity for _, _, entity in entities], 0.92


@lru_cache(maxsize=1)
def load_spacy_model() -> Any | None:
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def overlaps_existing(start: int, end: int, entities: list[tuple[int, int, dict[str, str]]]) -> bool:
    return any(start < existing_end and end > existing_start for existing_start, existing_end, _ in entities)
