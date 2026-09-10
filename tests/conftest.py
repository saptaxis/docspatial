import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docspatial import geometry  # noqa: E402


def word(text, left, top, right, bottom):
    """A word in the input format the library documents: text plus a quad."""
    return {"text": text, "quad": geometry.rect_std_to_quad_std([left, top, right, bottom])}


@pytest.fixture
def invoice_page():
    """Three lines. "Invoice" appears twice, on the first line and the third."""
    return [
        word("Invoice", 10, 10, 90, 30),
        word("Number", 100, 10, 190, 30),
        word("INV2024A", 200, 10, 320, 30),
        word("Total", 10, 60, 70, 80),
        word("Amount", 80, 60, 170, 80),
        word("1,23,456.78", 180, 60, 320, 80),
        word("Invoice", 10, 110, 90, 130),
        word("Date", 100, 110, 160, 130),
    ]


@pytest.fixture
def invoice_words(invoice_page):
    return geometry.prepare_words(invoice_page)
