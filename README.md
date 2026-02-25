# docspatial

Heuristic spatial reasoning on document OCR output.

`docspatial` is the layer between "I have OCR words" and "I need structured extractions." It provides composable utilities for spatial queries, phrase finding, and layout reasoning on word-level OCR output — from any OCR engine.

No models. No training. No cloud APIs. Just deterministic spatial logic.

## Why

Document processing is dominated by end-to-end models and cloud APIs. But a huge class of practical extraction tasks just needs spatial reasoning over already-extracted words: find a phrase split across lines, get the value to the right of a label, identify the closest field in a direction.

Everyone reimplements this ad-hoc. This library packages those patterns.

LLMs are powerful for document understanding, but they hallucinate coordinates, cost money at volume, and can't replace deterministic geometric operations when you need them.

## Modules

| Module | What it does |
|--------|-------------|
| **geometry** | Coordinate conversions (quad/rect/points), merging, IoU, rotation, affine transforms, normalization |
| **phrase_search** | Find text phrases in OCR word output with spatial coherence, handling duplicates, rotation, multi-line |
| **layout** | Directional filtering, closest-section queries, anchor-based extraction, words-after-phrase |
| **sections** | Line reconstruction from words, visual section merging, sentence merging, directional neighbor graphs |
| **text** | Number/ID detection, punctuation utilities, fuzzy substring matching |
| **viz** | Draw sections and points on document images for debugging |
| **datatypes** *(optional)* | Date/number/address/name parsing and normalization |

## Input Format

The library operates on a simple, universal input — a list of word dicts:

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
        # optional:
        "confidence": 0.98,
        "rotation_angle": 0.0,
    },
    ...
]
```

Bring your own words from Google Vision, Textract, Tesseract, PaddleOCR, Surya, or anything else.

## How This Fits In

| Library | What it does | Gap |
|---------|-------------|-----|
| `pdfplumber` | PDF text extraction with spatial methods | PDF-only, no rotated/scanned doc support |
| `layoutparser` | ML-based layout detection | Requires models, heavy |
| `docTR` | End-to-end OCR | Is the OCR engine itself, not post-processing |
| `paddleocr` | OCR engine + some layout | Engine-specific, not composable |
| `surya` | OCR + layout + reading order | Engine-specific, model-dependent |
| **docspatial** | Spatial queries on OCR words | Engine-agnostic, lightweight, heuristic |

`docspatial` is not an alternative to these — it's the layer that comes after them. Run your OCR engine of choice, then use `docspatial` to reason over the output.

## Status

**Work in progress.** Source files have been moved in from a private codebase. Still needs:

- [ ] Clean up imports and internal dependencies
- [ ] Remove provider-specific coupling (Google OCR pickle format, etc.)
- [ ] Replace heavy dependencies (torch/torchvision) with lightweight alternatives
- [ ] Consistent API surface across modules
- [ ] Tests
- [ ] PyPI packaging
- [ ] Documentation and examples

## Dependencies

Core: `numpy`, `shapely`

Optional: `opencv-python` (visual section merging, image rotation), `Pillow` (visualization)

Optional extras: `dateparser`, `usaddress`, `nameparser` (datatypes module)

## License

MIT
