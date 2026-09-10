# docspatial

Reconstructing document layout from the geometry of character and word bounding
boxes, rather than from a model.

Reading order, line and paragraph reconstruction, text size, rotation
normalisation, "the value to the right of this label", whether a phrase
continues onto the next line — all derived from where the boxes are, with no
training and no inference.

This is the geometric half of a document-understanding system built for
commercial invoice and form processing between roughly 2022 and 2024. In the
original the geometry ran alongside learned models and a hybrid shipped. The
learned half stayed behind. What survives is the part that turned out to be
durable, and the part you can read.

**It is an artifact, not a library.** It is here to be read, argued with, and
borrowed from. It is not packaged, not versioned, and not maintained. If you
want something to depend on, see [Neighbours](#neighbours).

---

## The idea

A page is words at coordinates. Almost everything people want from a document —
which text is a heading, what reads before what, which value belongs to which
label, where a table cell sits — is recoverable from the arrangement alone.

The interesting consequence is that these operations are *checkable*. A model
asked for a bounding box can return a plausible, wrong one. A median of word
heights cannot: it is either the median or it is a bug, and you can assert it in
a test. When the task is spatial and you need the answer to be grounded, the
substrate matters.

Article: [The Layer That Document Systems Keep Rebuilding](https://saptaxis.dev/articles/document-systems-rebuild-spatial-layer/).
This repository is the working sketch behind it.

---

## What it looks like

Bring words from any OCR engine as `text` plus a four-point `quad`.
`prepare_words` derives the rest — bounding rect, centroid, reading-order id,
text height, and the rotation angle of each word from the slope of its top edge.

```python
from docspatial import geometry, layout, phrase_search, sections

def word(text, l, t, r, b):
    return {"text": text, "quad": geometry.rect_std_to_quad_std([l, t, r, b])}

page = [
    word("Invoice", 10, 10, 90, 30),  word("Number", 100, 10, 190, 30),  word("INV2024A", 200, 10, 320, 30),
    word("Total",   10, 60, 70, 80),  word("Amount",  80, 60, 170, 80),  word("1,23,456.78", 180, 60, 320, 80),
    word("Invoice", 10, 110, 90, 130), word("Date",   100, 110, 160, 130),
]
words = geometry.prepare_words(page)
```

Label to value — the workhorse:

```python
>>> layout.get_next_word_after_phrase("Invoice Number", words)["raw_text"]
'INV2024A'

>>> layout.get_next_word_after_phrase("Total Amount", words, numbers=True)["raw_text"]
'1,23,456.78'
```

Lines are reconstructed from words by vertical band, not taken from the engine:

```python
>>> [l["text"] for l in sections.extract_lines_from_ocr(words)]
['Invoice Number INV2024A', 'Total Amount 1,23,456.78', 'Invoice Date']
```

Note that `Invoice` appears twice on this page. Phrase search picks the right
one, because it scores candidate word sequences on spatial coherence rather than
on string position:

```python
>>> phrase_search.find_phrase("Invoice Number", words)["word_ids"]
[0, 1]
```

---

## Modules

| Module | What it does |
|---|---|
| `geometry` | Coordinate conversions, merging, IoU, rotation, affine transforms, normalisation, reading order. Also `prepare_words`. |
| `phrase_search` | Find a phrase in OCR words with spatial coherence — duplicates, line wrapping, consistent text angle. |
| `layout` | Directional filtering, closest-section queries, anchor-based extraction, label-to-value. |
| `sections` | Line reconstruction, visual and sentence-based section merging, directional neighbour graphs. |
| `live_ocr` | Text inside a region; watermark filtering by rotation consistency. |
| `merge_sections` | Spatial set operations across OCR, key-value and table sections. |
| `text` | Number/ID detection, punctuation handling, fuzzy substring matching. |
| `document` | `DocumentPage` / `Document`, carrying deskew → bound → normalise → query. |
| `viz` | Draw sections and points on page images. |
| `metrics` | Score extracted field values against ground truth. |
| `adapters` | Optional Google Vision and Textract converters. Last verified 2024. |
| `datatypes` | Date/number/address/name parsing. Optional extras. |

---

## Running it

Not packaged. Clone it and work in the directory.

```
pip install numpy shapely pillow opencv-python
```

`scikit-learn` for the visual section merge; `dateparser`, `usaddress`,
`nameparser` for `datatypes`; `nltk`, `thefuzz`, `sentence-transformers` for the
fuzzy and embedding helpers in `text`. All are imported inside the functions
that need them, so the core works without any of them.

Tests need `pytest`:

```
pytest
```

137 tests covering the coordinate conversions, reading order, phrase search,
deskew, bounding, the neighbour graph, the Google Vision adapter, and the
extraction metrics. They also pin two limits: phrase search tolerates about five
degrees of skew before it stops matching in place, and a line wrap is followed
for two text heights down and two word widths left.

---

## Neighbours

Prior art and better-maintained work in the same area:

[`pdfplumber`](https://github.com/jsvine/pdfplumber) for spatial extraction from
born-digital PDFs · [`layoutparser`](https://github.com/Layout-Parser/layout-parser)
for model-based layout detection · [`docTR`](https://github.com/mindee/doctr)
and [`PaddleOCR`](https://github.com/PaddlePaddle/PaddleOCR) as OCR engines with
layout output · [`surya`](https://github.com/VikParuchuri/surya) for OCR, layout
and reading order · [`marker`](https://github.com/VikParuchuri/marker) for
document conversion.

If you need something supported, use one of those.

## License

MIT
