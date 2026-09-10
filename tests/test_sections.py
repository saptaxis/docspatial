"""Line reconstruction, block building, and the directional neighbour graph."""

import pytest

from conftest import word
from docspatial import geometry, live_ocr, sections


class TestLineReconstruction:
    def test_groups_words_into_lines(self, invoice_words):
        lines = sections.extract_lines_from_ocr(invoice_words)
        assert [l["text"] for l in lines] == [
            "Invoice Number INV2024A",
            "Total Amount 1,23,456.78",
            "Invoice Date",
        ]

    def test_a_line_quad_spans_its_words(self, invoice_words):
        first = sections.extract_lines_from_ocr(invoice_words)[0]
        assert geometry.quad_std_to_rect_std(first["quad"]) == [10, 10, 320, 30]

    def test_words_are_ordered_left_to_right_within_a_line(self):
        page = [
            word("third", 200, 10, 260, 30),
            word("first", 10, 10, 70, 30),
            word("second", 100, 10, 170, 30),
        ]
        lines = sections.extract_lines_from_ocr(geometry.prepare_words(page))
        assert lines[0]["text"] == "first second third"

    def test_small_vertical_drift_stays_one_line(self):
        """Membership uses a fraction of font height, not exact y equality."""
        page = [
            word("aligned", 10, 10, 90, 30),
            word("drifted", 100, 16, 180, 36),
        ]
        lines = sections.extract_lines_from_ocr(geometry.prepare_words(page))
        assert len(lines) == 1

    def test_a_full_line_apart_splits(self):
        page = [
            word("first", 10, 10, 90, 30),
            word("second", 10, 45, 90, 65),
        ]
        lines = sections.extract_lines_from_ocr(geometry.prepare_words(page))
        assert len(lines) == 2


class TestBlocksFromWords:
    def test_builds_one_block_per_line(self, invoice_words):
        blocks = sections.build_blocks_from_words(invoice_words)
        assert [b["text"] for b in blocks] == [
            "Invoice Number INV2024A",
            "Total Amount 1,23,456.78",
            "Invoice Date",
        ]

    def test_derives_font_metrics_from_word_boxes(self, invoice_words):
        block = sections.build_blocks_from_words(invoice_words)[0]
        assert block["font_metadata"]["font_size"] == 20.0
        assert block["font_metadata"]["font_width"] > 0

    def test_weight_is_not_detected(self, invoice_words):
        """bold and italic are placeholders; nothing reads them from geometry."""
        metadata = sections.build_blocks_from_words(invoice_words)[0]["font_metadata"]
        assert metadata["bold"] is False
        assert metadata["italic"] is False


class TestNeighbourGraph:
    def test_chains_stacked_blocks_north_to_south(self, invoice_words):
        blocks = sections.build_blocks_from_words(invoice_words)
        assert blocks[0]["north_block"][1] is None
        assert blocks[0]["south_block"][1] == 1
        assert blocks[1]["north_block"][1] == 0
        assert blocks[1]["south_block"][1] == 2
        assert blocks[2]["south_block"][1] is None

    def test_records_the_distance_to_each_neighbour(self, invoice_words):
        blocks = sections.build_blocks_from_words(invoice_words)
        distance, index = blocks[1]["north_block"]
        assert index == 0
        assert distance > 0


class TestSentenceMerge:
    def test_merges_until_terminal_punctuation(self):
        parts = [
            word("One sentence here", 0, 0, 100, 20),
            word("and it ends.", 0, 30, 100, 50),
            word("Next.", 0, 60, 100, 80),
        ]
        merged = sections.merge_sections__sentence_method(parts)
        assert [m["text"] for m in merged] == ["One sentence here and it ends.", "Next."]

    def test_an_unterminated_tail_is_kept_unmerged(self):
        parts = [
            word("Finished.", 0, 0, 100, 20),
            word("Dangling", 0, 30, 100, 50),
        ]
        merged = sections.merge_sections__sentence_method(parts)
        assert [m["text"] for m in merged] == ["Finished.", "Dangling"]


class TestWordsInsideRegion:
    def test_collects_the_words_that_fall_in_a_region(self, invoice_words):
        region = {"quad": geometry.rect_std_to_quad_std([0, 50, 340, 90])}
        found = live_ocr.get_text_section_inside_section(region, invoice_words)
        assert found["text"] == "Total Amount 1,23,456.78"
        assert found["word_ids"] == [3, 4, 5]

    def test_an_empty_region_gives_empty_text(self, invoice_words):
        region = {"quad": geometry.rect_std_to_quad_std([0, 200, 100, 300])}
        found = live_ocr.get_text_section_inside_section(region, invoice_words)
        assert found["text"] == ""

    def test_uniform_words_drops_rotation_outliers(self, invoice_page):
        """A watermark sits at a different angle from the body text."""
        page = list(invoice_page)
        watermark = word("COPY", 20, 55, 160, 85)
        watermark["quad"] = geometry.rotate_points_on_image(watermark["quad"], 40, (400, 200))
        page.append(watermark)
        words = geometry.prepare_words(page)

        region = {"quad": geometry.rect_std_to_quad_std([0, 40, 400, 120])}
        kept = live_ocr.get_text_section_inside_section(region, words, uniform_words=True)
        assert "COPY" not in kept["text"]
