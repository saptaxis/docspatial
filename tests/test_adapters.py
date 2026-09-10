"""Engine responses converted into the standard word format."""

import pytest

from docspatial import phrase_search
from docspatial.adapters import google_vision


def symbol(char, left, top, right, bottom):
    return {
        "text": char,
        "boundingBox": {
            "vertices": [
                {"x": left, "y": top},
                {"x": right, "y": top},
                {"x": right, "y": bottom},
                {"x": left, "y": bottom},
            ]
        },
    }


def vision_word(text, left, top, right, bottom, confidence=0.98):
    char_width = (right - left) // len(text)
    symbols = [
        symbol(char, left + i * char_width, top, left + (i + 1) * char_width, bottom)
        for i, char in enumerate(text)
    ]
    return {
        "symbols": symbols,
        "confidence": confidence,
        "boundingBox": {
            "vertices": [
                {"x": left, "y": top},
                {"x": right, "y": top},
                {"x": right, "y": bottom},
                {"x": left, "y": bottom},
            ]
        },
    }


def vision_response(*lines):
    """A DOCUMENT_TEXT_DETECTION response, one block per line."""
    return {
        "textAnnotations": [{"description": "page text"}],
        "fullTextAnnotation": {
            "pages": [{"blocks": [{"paragraphs": [{"words": list(line)}]} for line in lines]}]
        },
    }


@pytest.fixture
def response():
    return vision_response(
        [vision_word("Invoice", 10, 10, 90, 30), vision_word("Number", 100, 10, 190, 30)],
        [vision_word("Total", 10, 60, 70, 80)],
    )


class TestGoogleVision:
    def test_converts_words_into_standard_format(self, response):
        words = google_vision.get_page_ocr_words(response)
        assert [w["text"] for w in words] == ["Invoice", "Number", "Total"]
        assert words[0]["quad"] == [
            {"x": 10, "y": 10},
            {"x": 90, "y": 10},
            {"x": 90, "y": 30},
            {"x": 10, "y": 30},
        ]

    def test_derives_font_metrics_from_symbol_boxes(self, response):
        """Symbol geometry is what Google gives that a word box alone does not."""
        first = google_vision.get_page_ocr_words(response)[0]
        assert first["font_height"] == 20.0
        assert first["font_width"] > 0

    def test_assigns_ids_in_reading_order(self, response):
        words = google_vision.get_page_ocr_words(response)
        assert [w["id"] for w in words] == [0, 1, 2]

    def test_carries_confidence_through(self, response):
        assert google_vision.get_page_ocr_words(response)[0]["confidence"] == 0.98

    def test_output_feeds_the_core_without_prepare_words(self, response):
        """The adapter already emits the derived fields."""
        words = google_vision.get_page_ocr_words(response)
        assert phrase_search.find_phrase("Invoice Number", words)["word_ids"] == [0, 1]

    def test_an_empty_response_gives_no_words(self):
        assert google_vision.get_page_ocr_words({"textAnnotations": []}) == []

    def test_punctuation_can_be_stripped_on_the_way_in(self):
        resp = vision_response([vision_word("Total:", 10, 10, 80, 30)])
        words = google_vision.get_page_ocr_words(resp, remove_punctuation=True)
        assert words[0]["text"] == "Total"


class TestResponseSchemaVariants:
    def test_camel_case_response_is_detected_as_legacy(self, response):
        assert google_vision.is_legacy_ocr_format(response) is True
        assert google_vision.get_ocr_json_key_names(response)["quad"] == "boundingBox"

    def test_snake_case_response_uses_the_other_key_names(self):
        keys = google_vision.get_ocr_json_key_names({"text_annotations": [{}]})
        assert keys["fta"] == "full_text_annotation"
        assert keys["quad"] == "bounding_box"

    def test_break_types_are_named(self):
        assert int(google_vision.OCRBreakType.EOL_SURE_SPACE) == 3
        assert int(google_vision.OCRBreakType.LINE_BREAK) == 5
