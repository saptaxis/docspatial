# `docspatial` — Direction Document

## The Problem

Document processing in 2025+ is dominated by two poles:

- **End-to-end models** (LayoutLM, Donut, DocTR) that require training data and GPU infrastructure
- **Cloud APIs** (Google Document AI, AWS Textract, Azure DI) that are black-box and per-page cost

But a huge class of practical document extraction tasks sits in the middle: you already have word-level OCR output (from any engine), and you need to **spatially reason over it** — find a phrase that's split across lines, extract the value to the right of a label, identify the closest field in a direction.

Everyone reimplements this ad-hoc. There's no lightweight, OCR-engine-agnostic library for spatial reasoning on document words.

LLMs are powerful for document understanding, but they hallucinate coordinates, are expensive at volume, and can't replace deterministic spatial logic when you need reliable geometric operations.

## The Idea

A small, focused Python library for **heuristic spatial reasoning on OCR output**.

Not an OCR engine. Not a layout model. A toolkit that takes word-level OCR output (from any source) and provides composable utilities for spatial queries, phrase finding, and layout reasoning.

Think of it as the spatial query layer between "I have OCR words" and "I need structured extractions."

## Core Modules

### 1. Spatial Primitives (`geometry`)

Coordinate conversions, merging, and geometric operations on document regions.

- **Coordinate formats**: quad (4-point polygon) ↔ rect (LTRB) ↔ points list — the format zoo that every doc project deals with
- **Merging**: combine multiple word/region quads into bounding regions
- **IoU**: intersection-over-union for both rects and arbitrary polygons
- **Rotation**: rotate image coordinates with full-image preservation (no cropping), rotate sections/words along with the image
- **Affine transforms**: apply arbitrary affine matrices to sets of sections (useful for cropping-as-translation, deskewing)
- **Normalization**: normalize coordinates to [0,1] for page-relative positions
- **Multi-page support**: offset normalized coordinates by page number for unified multi-page coordinate spaces

### 2. Phrase Search (`find_phrase`)

Find a text phrase in OCR word output, handling real-world messiness.

The algorithm:
1. Find all occurrences of each word in the phrase across the OCR word list
2. Generate all possible combinations (with a sane upper bound)
3. Filter for unique word instances (no word used twice)
4. Filter for spatial coherence — successive words must be to the right or below within tolerance
5. Filter for consistent text angle (handles rotated documents)
6. Minimize successive inter-word distance to pick the best match
7. Return within a distance tolerance (handles multiple valid instances)

Key capabilities:
- **Case insensitive** matching
- **Punctuation handling** — optionally strip punctuation from both query and OCR
- **Same-line mode** — restrict to single-line matches
- **Document-order mode** — allow left→right, top→bottom flow across lines
- **Multiple results** — return all valid instances, sorted top-to-bottom, left-to-right
- **Section filtering** — search within a bounding region only

### 3. Layout Queries (`layout`)

Directional and proximity queries for anchor-based extraction.

- **Directional filtering**: given an anchor section and a direction (`east`, `south`, `south-east`, etc.), filter candidate sections that lie in that direction
- **Closest section**: find the nearest section to an anchor, optionally constrained by direction
- **Key-value filtering**: filter sections by key text (for KV-pair based extractions)
- **Words after phrase**: given a found phrase, get the next N words by OCR reading order — useful for "label: value" patterns
- **Numbers after phrase**: same but filtered to numeric tokens
- **Document sorting**: sort sections in reading order (top→bottom, left→right) using median text height for line grouping

### 4. Lines & Sections (`sections`)

Line reconstruction, section merging, and spatial neighbor graphs.

- **Line detection from words**: reconstruct text lines from word-level OCR using vertical tolerance — groups words that share the same vertical band, then sorts left-to-right within each line
- **Visual section merging**: render sections as binary masks, dilate by line spacing, find connected components via OpenCV, re-associate original text — a clever geometry-only approach to merge fragmented OCR blocks without any ML
- **Sentence-based merging**: merge consecutive sections until sentence-ending punctuation is found — useful for reconstructing paragraphs from over-segmented OCR
- **Directional neighbor mapping**: for each section, compute its nearest neighbor in each cardinal direction (north/south/east/west) with configurable distance and orthogonal thresholds — builds a spatial graph over document regions
- **Spatial set operations**: filter one set of sections by non-overlap with another set (e.g., remove OCR paragraphs that fall inside detected tables or key-value regions)
- **Words inside region**: given a bounding region and a word list, find which words fall inside (polygon intersection with configurable threshold). Optionally filter by rotation angle consistency to remove watermark text

### 5. Text Utilities (`text`)

Lightweight text classification and processing for OCR output.

- **Number detection**: `is_readable_number` — handles Indian (1,23,456) and international (1,234,567) comma formatting, decimals, signs. Thorough validation of comma group structure
- **ID code detection**: detect alphanumeric identifier strings (e.g., invoice numbers, account codes) that mix letters and digits
- **Punctuation utilities**: filter/strip punctuation, detect punctuation-only tokens, properly re-space text around punctuation
- **Title case detection**: check if text follows title-case conventions (with exception words like "the", "of")
- **Fuzzy substring search**: find a query phrase within a longer text using ngram windowing + SequenceMatcher — useful for approximate matching against OCR text that may have errors
- **Word splitting**: split text into words with or without treating punctuation as separators

### 6. Visualization (`viz`)

Debug and inspect spatial queries by drawing results on document images. Pillow-only, no heavy dependencies.

- **Draw sections on image**: render bounding polygons (not just axis-aligned rects) with optional text labels on a document image — the core debugging tool for any spatial query
- **Draw points on image**: visualize centroids, intersection points, or other coordinates as dots — useful for debugging distance and direction calculations
- **Image grid**: arrange multiple page images into a grid layout with auto-sizing — handy for visualizing multi-page documents side-by-side
- **Cleanup needed**: current source hardcodes a font path (`Pillow/Tests/fonts/FreeMono.ttf`) — should fallback to PIL default font gracefully

### 7. Data Types (`datatypes`) — optional extras

Data type detection and normalization for extracted text values. Heavier dependencies (`dateparser`, `usaddress`, `nameparser`), so packaged as optional.

- **Type classifier**: given extracted text, classify as date / quantitative number / generic number / ID code / address / freeform / empty
- **Date parsing**: wraps `dateparser` and `dateutil` with a comprehensive format map (dd-mm-yyyy, mm/dd/yyyy, yyyy-mm-dd, month-only, year-only, etc.). Handles `is_date`, `contains_date`, `search_dates`, and `format_date`
- **Number formatting**: parse and reformat numbers across styles — integer, Indian comma-separated, international comma-separated, decimal rounding. Handles the Indian numbering system (lakhs/crores) correctly
- **Name parsing**: wraps `nameparser.HumanName` — extract first/middle/last/suffix, reformat to "Last, First" or "Initials. Last" etc.
- **US Address parsing**: wraps `usaddress` — extract street, city, state, zip, normalize state codes, split zip+4
- **Data type vectors**: create feature vectors from text for downstream classification (contains_date, contains_id, contains_numeric, contains_address, is_short, is_long)

## Input Format

The library operates on a simple, universal input: **a list of word dicts**.

```python
words = [
    {
        "text": "Invoice",
        "quad": [
            {"x": 100, "y": 50},   # top-left
            {"x": 200, "y": 50},   # top-right
            {"x": 200, "y": 80},   # bottom-right
            {"x": 100, "y": 80},   # bottom-left
        ],
        # optional fields:
        "confidence": 0.98,
        "rotation_angle": 0.0,
    },
    ...
]
```

Adapter functions can convert from common OCR output formats (Google Vision, Textract, Tesseract, PaddleOCR) into this standard format, but the core library is format-agnostic.

## What This Is Not

- Not an OCR engine — bring your own words
- Not a layout detection model — no ML, no training
- Not a document parser — no PDF/image handling
- Not a replacement for LLMs on understanding tasks — this is for deterministic spatial logic

## Dependencies

Minimal:
- `numpy` — core numeric operations
- `shapely` — polygon intersection/IoU (already the standard for geometric ops)

Optional:
- `opencv-python` — only for image rotation (deskewing)
- `Pillow` — only for image I/O if using visualization utilities

Explicitly **not** depending on: torch, torchvision, cloud SDKs.

## Who Would Use This

- Teams extracting structured data from invoices, receipts, forms, medical documents
- Anyone post-processing OCR output who needs "find this label and get the value next to it"
- Pipelines that use LLMs for understanding but need deterministic spatial operations for geometry
- Developers building document processing without wanting to train or fine-tune models

## Existing Landscape

| Library | What it does | Gap |
|---------|-------------|-----|
| `pdfplumber` | PDF text extraction with spatial methods | PDF-only, no rotated/scanned doc support |
| `layoutparser` | ML-based layout detection | Requires models, heavy |
| `docTR` | End-to-end OCR | Is the OCR engine itself, not post-processing |
| `paddleocr` | OCR engine + some layout | Engine-specific, not composable |
| **this** | Spatial queries on OCR words | Engine-agnostic, lightweight, heuristic |

## Module Summary

| Module | Source files | What it does | Dependencies |
|--------|-------------|-------------|--------------|
| **geometry** | `geometry.py` | Coordinate conversions, merging, IoU, rotation, affine transforms | numpy, shapely |
| **phrase_search** | `find_in_ocr.py` | Find phrases in OCR with spatial coherence | numpy |
| **layout** | `layout_utils.py`, `live_ocr.py` | Directional queries, words-in-region, anchor-based extraction | numpy, shapely |
| **sections** | `section_detection.py`, `merge_sections.py` | Line reconstruction, visual merging, neighbor graphs, section filtering | numpy, shapely, opencv (optional) |
| **text** | `text_utils.py` (subset) | Number/ID/punctuation detection, fuzzy substring matching | stdlib only |
| **viz** | `visualization.py` | Draw sections/points on images, image grids | Pillow |
| **datatypes** | `datatypes.py` | Date/number/address/name parsing and normalization | dateparser, usaddress, nameparser |

Core modules (1-6): ~1700-2200 lines, numpy + shapely + Pillow.
Optional extras (6): ~500 lines, heavier dependencies.
Format adapters: ~200 lines (Google Vision, Textract, Tesseract, PaddleOCR → standard format).
