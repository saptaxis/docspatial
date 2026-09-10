"""The page pipeline: measure skew, deskew, bound, normalise, query."""

import pytest

from conftest import word
from docspatial import document, geometry, layout


class TestPage:
    def test_takes_words_and_reconstructs_lines(self, invoice_page):
        page = document.DocumentPage(invoice_page, name="p1")
        assert [l["text"] for l in page.lines] == [
            "Invoice Number INV2024A",
            "Total Amount 1,23,456.78",
            "Invoice Date",
        ]

    def test_an_upright_page_measures_no_skew(self, invoice_page):
        assert document.DocumentPage(invoice_page).rotation_angle == 0.0

    def test_page_size_falls_back_to_the_word_extent(self, invoice_page):
        """No image is required for the coordinate work."""
        page = document.DocumentPage(invoice_page)
        assert (page.image_width, page.image_height) == (320, 130)

    def test_finds_a_phrase_through_the_page(self, invoice_page):
        page = document.DocumentPage(invoice_page)
        assert page.find_phrase("Invoice Number")[0]["word_ids"] == [0, 1]

    def test_find_all_phrases_keys_by_phrase(self, invoice_page):
        page = document.DocumentPage(invoice_page)
        found = page.find_all_phrases(["Invoice Number", "Total Amount"])
        assert sorted(found) == ["Invoice Number", "Total Amount"]
        assert found["Total Amount"][0]["word_ids"] == [3, 4]


class TestDeskew:
    @pytest.mark.parametrize("angle", [5, 12, 25, 40, -18, -33])
    def test_auto_deskew_removes_the_skew(self, invoice_page, angle):
        skewed = geometry.rotate_sections_on_image(invoice_page, angle, (400, 200))
        page = document.DocumentPage(skewed, auto_deskew=True)
        assert page.find_document_rotation() == pytest.approx(0.0, abs=0.01)

    def test_the_measured_angle_matches_the_skew_applied(self, invoice_page):
        skewed = geometry.rotate_sections_on_image(invoice_page, 25, (400, 200))
        page = document.DocumentPage(skewed, auto_deskew=True)
        assert page.rotation_angle == pytest.approx(-25, abs=0.5)

    def test_a_phrase_is_findable_after_deskewing(self, invoice_page):
        skewed = geometry.rotate_sections_on_image(invoice_page, 25, (400, 200))
        page = document.DocumentPage(skewed, auto_deskew=True)
        assert page.find_phrase("Invoice Number")[0]["word_ids"] == [0, 1]

    def test_without_deskew_a_skewed_page_keeps_its_angle(self, invoice_page):
        skewed = geometry.rotate_sections_on_image(invoice_page, 25, (400, 200))
        page = document.DocumentPage(skewed)
        assert abs(page.find_document_rotation()) > 20

    def test_deskewing_twice_is_a_no_op(self, invoice_page):
        skewed = geometry.rotate_sections_on_image(invoice_page, 25, (400, 200))
        page = document.DocumentPage(skewed, auto_deskew=True)
        page.rotate_page()
        assert page.find_document_rotation() == pytest.approx(0.0, abs=0.01)

    def test_rotation_is_measured_from_the_longest_words(self):
        """Long words give a more reliable slope, so a short outlier is ignored."""
        page_words = [word("Acknowledgement", 10, 10, 310, 30)] + [
            word("x", 10 + i * 20, 60, 25 + i * 20, 80) for i in range(5)
        ]
        skewed = geometry.rotate_sections_on_image(page_words, 15, (400, 200))
        page = document.DocumentPage(skewed)
        assert page.rotation_angle == pytest.approx(-15, abs=1.0)


class TestBounding:
    def test_bounding_translates_words_to_the_origin(self, invoice_page):
        page = document.DocumentPage(invoice_page, bounded=True)
        assert page.document_bbox == [10, 10, 320, 130]
        assert page.words[0]["rect"] == [0, 0, 80, 20]

    def test_bounding_resizes_the_page(self, invoice_page):
        page = document.DocumentPage(invoice_page, bounded=True)
        assert (page.image_width, page.image_height) == (310, 120)

    def test_bounding_restores_normalised_coordinates(self, invoice_page):
        page = document.DocumentPage(invoice_page, bounded=True)
        assert "norm_centroid" in page.words[0]

    def test_a_phrase_is_findable_after_bounding(self, invoice_page):
        page = document.DocumentPage(invoice_page, bounded=True)
        assert page.find_phrase("Invoice Number")[0]["word_ids"] == [0, 1]

    def test_deskew_and_bound_compose(self, invoice_page):
        skewed = geometry.rotate_sections_on_image(invoice_page, 25, (400, 200))
        page = document.DocumentPage(
            skewed, auto_deskew=True, bounded=True, normalize_sections=True
        )
        assert page.find_document_rotation() == pytest.approx(0.0, abs=0.01)
        assert page.find_phrase("Invoice Number")[0]["word_ids"] == [0, 1]


class TestMultiPage:
    def test_numbers_pages_from_one(self, invoice_page):
        doc = document.Document(
            [document.DocumentPage(invoice_page), document.DocumentPage(invoice_page)]
        )
        assert doc.num_pages == 2
        assert doc.page_nums == [1, 2]

    def test_searches_every_page(self, invoice_page):
        doc = document.Document(
            [document.DocumentPage(invoice_page), document.DocumentPage(invoice_page)]
        )
        found = doc.find_phrase("Total Amount")
        assert sorted(found) == [1, 2]
        assert found[1][0]["word_ids"] == [3, 4]

    def test_an_unknown_page_number_is_rejected(self, invoice_page):
        doc = document.Document([document.DocumentPage(invoice_page)])
        with pytest.raises(ValueError):
            doc.find_phrase("Total", page_nums=[7])

    def test_merging_offsets_each_page_in_a_shared_space(self, invoice_page):
        pages = [document.DocumentPage(invoice_page, normalize_sections=True) for _ in range(2)]
        doc = document.Document(pages, merge_method="zero_to_n")
        merged = doc.merge_sections_across_pages({1: pages[0].words, 2: pages[1].words})
        page_one_y = merged[0]["norm_centroid"][1]
        page_two_y = merged[len(pages[0].words)]["norm_centroid"][1]
        assert page_two_y == pytest.approx(page_one_y + 1.0)


class TestLabelToValue:
    def test_takes_the_word_after_a_label(self, invoice_words):
        found = layout.get_next_word_after_phrase("Invoice Number", invoice_words)
        assert found["raw_text"] == "INV2024A"

    def test_takes_the_number_after_a_label(self, invoice_words):
        found = layout.get_next_word_after_phrase("Total Amount", invoice_words, numbers=True)
        assert found["raw_text"] == "1,23,456.78"

    def test_an_absent_label_gives_nothing(self, invoice_words):
        assert layout.get_next_word_after_phrase("Purchase Order", invoice_words) == {}
