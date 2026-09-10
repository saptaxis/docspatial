"""Finding a phrase by spatial coherence rather than string position."""

import pytest

from conftest import word
from docspatial import geometry, phrase_search


class TestFindPhrase:
    def test_finds_a_phrase_on_one_line(self, invoice_words):
        found = phrase_search.find_phrase("Invoice Number", invoice_words)
        assert found["word_ids"] == [0, 1]
        assert found["text"] == "invoice number"

    def test_picks_the_right_one_when_a_word_repeats_elsewhere(self, invoice_words):
        """"Invoice" is on line 1 and line 3. "Invoice Date" is the line 3 pair."""
        assert phrase_search.find_phrase("Invoice Number", invoice_words)["word_ids"] == [0, 1]
        assert phrase_search.find_phrase("Invoice Date", invoice_words)["word_ids"] == [6, 7]

    def test_matching_is_case_insensitive_by_default(self, invoice_words):
        assert phrase_search.find_phrase("INVOICE NUMBER", invoice_words)["word_ids"] == [0, 1]

    def test_returns_empty_when_a_word_is_absent(self, invoice_words):
        assert phrase_search.find_phrase("Invoice Reference", invoice_words) == {}

    def test_return_all_gives_every_candidate(self, invoice_words):
        found = phrase_search.find_phrase("Invoice", invoice_words, return_all=True)
        assert len(found) == 2
        assert [f["word_ids"] for f in found] == [[0], [6]]

    def test_results_are_ordered_top_to_bottom(self, invoice_words):
        found = phrase_search.find_phrase("Invoice", invoice_words, return_all=True)
        centroid_y = [f["centroid"][1] for f in found]
        assert centroid_y == sorted(centroid_y)

    def test_the_merged_quad_covers_both_words(self, invoice_words):
        found = phrase_search.find_phrase("Invoice Number", invoice_words)
        assert geometry.quad_std_to_rect_std(found["quad"]) == [10, 10, 190, 30]

    def test_confidence_is_averaged_over_the_words(self, invoice_page):
        page = list(invoice_page)
        page[0] = dict(page[0], confidence=0.6)
        page[1] = dict(page[1], confidence=1.0)
        found = phrase_search.find_phrase("Invoice Number", geometry.prepare_words(page))
        assert found["confidence"] == pytest.approx(0.8)


def wrapped_page(next_line_x, line_gap=30):
    """"Terms" at the end of one line, "and Conditions" starting the next."""
    return [
        word("Terms", 200, 10, 280, 30),
        word("and", next_line_x, 10 + line_gap, next_line_x + 50, 30 + line_gap),
        word("Conditions", next_line_x + 60, 10 + line_gap, next_line_x + 180, 30 + line_gap),
    ]


class TestAcrossLineBreaks:
    def test_finds_a_phrase_that_wraps_to_the_next_line(self):
        found = phrase_search.find_phrase(
            "Terms and", geometry.prepare_words(wrapped_page(next_line_x=100))
        )
        assert found["word_ids"] == [0, 1]

    def test_same_line_mode_rejects_a_wrapped_phrase(self):
        words = geometry.prepare_words(wrapped_page(next_line_x=100))
        assert phrase_search.find_phrase("Terms and", words, same_line=True) == {}
        assert (
            phrase_search.find_phrase("and Conditions", words, same_line=True)["word_ids"]
            == [1, 2]
        )

    @pytest.mark.parametrize(
        "line_gap, found",
        [(24, True), (30, True), (40, True), (44, False), (50, False)],
        ids=["1.2x", "1.5x", "2.0x", "2.2x", "2.5x"],
    )
    def test_the_wrap_reaches_two_text_heights_down(self, line_gap, found):
        """Vertical tolerance is 2x text height, so ordinary line spacing fits."""
        words = geometry.prepare_words(wrapped_page(next_line_x=200, line_gap=line_gap))
        assert bool(phrase_search.find_phrase("Terms and", words)) is found

    @pytest.mark.parametrize(
        "next_line_x, found",
        [(200, True), (100, True), (60, True), (20, False), (10, False)],
    )
    def test_the_wrap_reaches_two_word_widths_left(self, next_line_x, found):
        """Horizontal tolerance is relative to word width, not page width, so a
        wrap back across a wide page is not matched."""
        words = geometry.prepare_words(wrapped_page(next_line_x=next_line_x))
        assert bool(phrase_search.find_phrase("Terms and", words)) is found


class TestRotationTolerance:
    """Search in place tolerates a few degrees of skew. Beyond that, deskew first.

    The bound is the asymmetric vertical tolerance in
    get_right_or_bottom_combinations: half a text height up, two down. Text that
    rises to the right hits the tight bound first.
    """

    def test_finds_the_phrase_on_slightly_skewed_text(self, invoice_page):
        skewed = geometry.rotate_sections_on_image(invoice_page, 5, (400, 200))
        found = phrase_search.find_phrase("Invoice Number", geometry.prepare_words(skewed))
        assert found["word_ids"] == [0, 1]

    def test_misses_once_the_skew_is_large(self, invoice_page):
        skewed = geometry.rotate_sections_on_image(invoice_page, 25, (400, 200))
        assert phrase_search.find_phrase("Invoice Number", geometry.prepare_words(skewed)) == {}

    def test_deskewing_first_makes_it_findable_again(self, invoice_page):
        skewed = geometry.rotate_sections_on_image(invoice_page, 25, (400, 200))
        measured = geometry.prepare_words(skewed)[0]["rotation_angle"]
        upright = geometry.prepare_words(
            geometry.rotate_sections_on_image(skewed, measured, (400, 200))
        )
        assert phrase_search.find_phrase("Invoice Number", upright)["word_ids"] == [0, 1]

    def test_widening_the_vertical_tolerance_also_works(self, invoice_page):
        """Shows the bound is the tolerance, not the angle filter."""
        skewed = geometry.prepare_words(
            geometry.rotate_sections_on_image(invoice_page, 25, (400, 200))
        )
        candidates = [
            (skewed[0], skewed[1]),
            (skewed[6], skewed[1]),
        ]
        assert phrase_search.get_right_or_bottom_combinations(candidates) == []
        widened = phrase_search.get_right_or_bottom_combinations(
            candidates, y_tolerance_fraction=(2.0, 2.0)
        )
        assert len(widened) == 1
