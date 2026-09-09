# docspatial

"""Heuristic spatial reasoning on document OCR output.

The layer between "I have OCR words" and "I need structured extractions".
Bring word-level output from any OCR engine as a list of dicts with ``text``
and ``quad``, run it through :func:`geometry.prepare_words`, and query it.

    from docspatial import geometry, phrase_search, layout

    words = geometry.prepare_words(my_ocr_words)
    value = layout.get_next_word_after_phrase("Invoice Number", words)

Modules
-------
geometry
    Coordinate conversions, merging, IoU, rotation, affine transforms,
    normalization, reading order. Also ``prepare_words``, which derives the
    fields the other modules expect.
phrase_search
    Find a phrase in OCR words with spatial coherence — handles duplicates,
    multi-line wrapping and consistent text angle.
layout
    Directional filtering, closest-section queries, anchor-based extraction,
    "the value after this label".
sections
    Line reconstruction, visual and sentence-based section merging,
    directional neighbour graphs.
text
    Number/ID detection, punctuation utilities, fuzzy substring matching.
live_ocr
    Text inside a region; watermark filtering by rotation consistency.
merge_sections
    Spatial set operations over OCR, key-value and table sections.
viz
    Draw sections and points on document images for debugging.
datatypes
    Date/number/address/name parsing. Optional extras.
"""

from . import (
    assets,
    geometry,
    layout,
    live_ocr,
    lookup_assets,
    merge_sections,
    phrase_search,
    sections,
    text,
    utils,
    viz,
)

__all__ = [
    "assets",
    "geometry",
    "layout",
    "live_ocr",
    "lookup_assets",
    "merge_sections",
    "phrase_search",
    "sections",
    "text",
    "utils",
    "viz",
]

# datatypes is deliberately not imported here — it needs the optional
# 'datatypes' extra (dateparser, usaddress, nameparser). Import it directly.
