"""Coordinate conversions, the derived fields, and reading order."""

import numpy as np
import pytest

from conftest import word
from docspatial import geometry

QUAD = [{"x": 10, "y": 20}, {"x": 90, "y": 20}, {"x": 90, "y": 50}, {"x": 10, "y": 50}]
RECT = [10, 20, 90, 50]


class TestConversionsRoundTrip:
    def test_quad_to_rect_and_back(self):
        assert geometry.quad_std_to_rect_std(QUAD) == RECT
        assert geometry.rect_std_to_quad_std(RECT) == QUAD

    def test_quad_to_points_list_and_back(self):
        points = geometry.quad_std_to_quad_points_list(QUAD)
        assert points == [[10, 20], [90, 20], [90, 50], [10, 50]]
        assert geometry.quad_points_list_to_quad_std(points) == QUAD

    def test_rect_to_points_list_matches_quad_route(self):
        via_rect = geometry.rect_std_to_quad_points_list(RECT)
        via_quad = geometry.quad_std_to_quad_points_list(geometry.rect_std_to_quad_std(RECT))
        assert via_rect == via_quad

    def test_rect_cts_round_trip(self):
        cts = geometry.quad_std_to_rect_cts(QUAD)
        assert cts == {"left": 10, "top": 20, "right": 90, "bottom": 50}
        assert geometry.rect_cts_to_quad_std(cts) == QUAD

    def test_a_rotated_quad_is_not_recoverable_from_its_rect(self):
        """rect is axis-aligned, so the round trip is lossy once a quad is skewed."""
        skewed = geometry.rotate_points_on_image(QUAD, 20, (400, 200))
        recovered = geometry.rect_std_to_quad_std(geometry.quad_std_to_rect_std(skewed))
        assert recovered != skewed


class TestBoxIou:
    def test_identical_boxes(self):
        assert geometry.get_box_iou(RECT, RECT) == 1.0

    def test_disjoint_boxes(self):
        assert geometry.get_box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_half_overlap(self):
        # two 10x10 boxes sharing half their width: 50 / 150
        assert geometry.get_box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)

    def test_is_symmetric(self):
        a, b = [0, 0, 10, 10], [5, 5, 20, 20]
        assert geometry.get_box_iou(a, b) == geometry.get_box_iou(b, a)

    def test_touching_edges_do_not_overlap(self):
        assert geometry.get_box_iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0


class TestPrepareWords:
    def test_derives_every_field_from_text_and_quad_alone(self):
        prepared = geometry.prepare_words([word("Invoice", 10, 20, 90, 50)])[0]
        assert prepared["rect"] == RECT
        assert prepared["centroid"] == [50.0, 35.0]
        assert prepared["height"] == 30
        assert prepared["width"] == 80
        assert prepared["font_height"] == 30
        assert prepared["rotation_angle"] == 0.0
        assert prepared["confidence"] == 1.0
        assert prepared["id"] == 0

    def test_ids_follow_input_order(self):
        words = geometry.prepare_words(
            [word("a", 0, 0, 10, 10), word("b", 0, 0, 10, 10), word("c", 0, 0, 10, 10)]
        )
        assert [w["id"] for w in words] == [0, 1, 2]

    def test_keeps_values_an_engine_already_supplied(self):
        supplied = word("Invoice", 10, 20, 90, 50)
        supplied["confidence"] = 0.42
        supplied["rotation_angle"] = 7.5
        prepared = geometry.prepare_words([supplied])[0]
        assert prepared["confidence"] == 0.42
        assert prepared["rotation_angle"] == 7.5

    def test_recompute_overrides_supplied_geometry(self):
        supplied = word("Invoice", 10, 20, 90, 50)
        supplied["rotation_angle"] = 7.5
        prepared = geometry.prepare_words([supplied], recompute=True)[0]
        assert prepared["rotation_angle"] == 0.0

    def test_does_not_mutate_the_input(self):
        page = [word("Invoice", 10, 20, 90, 50)]
        geometry.prepare_words(page)
        assert "rect" not in page[0]

    def test_rotation_angle_comes_from_the_top_edge_slope(self):
        # top edge dropping to the right by 45 degrees
        quad = [{"x": 0, "y": 0}, {"x": 10, "y": 10}, {"x": 10, "y": 20}, {"x": 0, "y": 10}]
        prepared = geometry.prepare_words([{"text": "x", "quad": quad}])[0]
        assert prepared["rotation_angle"] == pytest.approx(45.0)


class TestReadingOrder:
    def test_words_on_one_line_stay_together_despite_uneven_y(self):
        """Row banding uses median text height, so a few pixels of drift is one line."""
        page = [
            word("second", 10, 60, 90, 80),
            word("right", 100, 12, 180, 32),  # 2px higher than "left"
            word("left", 10, 10, 90, 30),
        ]
        ordered = geometry.sort_sections_as_document(geometry.prepare_words(page))
        assert [w["text"] for w in ordered] == ["left", "right", "second"]

    def test_orders_top_to_bottom_then_left_to_right(self, invoice_words):
        ordered = [w["text"] for w in geometry.sort_sections_as_document(invoice_words)]
        assert ordered == [
            "Invoice", "Number", "INV2024A",
            "Total", "Amount", "1,23,456.78",
            "Invoice", "Date",
        ]

    def test_empty_input_is_returned_unchanged(self):
        assert geometry.sort_sections_as_document([]) == []


class TestDerivedFieldsSurviveTransforms:
    """A moved quad invalidates everything derived from it."""

    def test_rotation_rederives_the_angle(self, invoice_words):
        rotated = geometry.rotate_sections_on_image(invoice_words, 30, (400, 200))
        assert rotated[0]["rotation_angle"] != invoice_words[0]["rotation_angle"]
        assert rotated[0]["rotation_angle"] == pytest.approx(-29.74, abs=0.5)

    def test_rotation_rederives_the_rect(self, invoice_words):
        rotated = geometry.rotate_sections_on_image(invoice_words, 30, (400, 200))
        assert rotated[0]["rect"] == geometry.quad_std_to_rect_std(rotated[0]["quad"])

    def test_affine_rederives_the_rect(self, invoice_words):
        moved = geometry.affine_transform_sections(
            invoice_words, np.array([[1, 0, -10], [0, 1, -10]])
        )
        assert moved[0]["rect"] == [0, 0, 80, 20]

    def test_normalized_fields_are_dropped_not_left_stale(self, invoice_words):
        geometry.normalize_coordinates_in_sections(invoice_words, 400, 200)
        assert "norm_centroid" in invoice_words[0]
        rotated = geometry.rotate_sections_on_image(invoice_words, 30, (400, 200))
        assert "norm_centroid" not in rotated[0]

    def test_a_section_without_derived_fields_does_not_gain_any(self):
        bare = {"text": "x", "quad": QUAD}
        rotated = geometry.rotate_sections_on_image([bare], 30, (400, 200))[0]
        assert "rect" not in rotated
        assert "rotation_angle" not in rotated


class TestRotation:
    def test_rotating_by_the_measured_angle_returns_to_upright(self, invoice_page):
        """Deskew rotates by +measured. Its negative doubles the skew instead."""
        skewed = geometry.rotate_sections_on_image(invoice_page, 25, (400, 200))
        measured = geometry.prepare_words(skewed)[0]["rotation_angle"]
        upright = geometry.prepare_words(
            geometry.rotate_sections_on_image(skewed, measured, (400, 200))
        )
        assert upright[0]["rotation_angle"] == pytest.approx(0.0, abs=0.01)

    def test_negating_the_measured_angle_makes_the_skew_worse(self, invoice_page):
        skewed = geometry.rotate_sections_on_image(invoice_page, 25, (400, 200))
        measured = geometry.prepare_words(skewed)[0]["rotation_angle"]
        wrong_way = geometry.prepare_words(
            geometry.rotate_sections_on_image(skewed, -measured, (400, 200))
        )
        assert abs(wrong_way[0]["rotation_angle"]) > abs(measured)

    @pytest.mark.parametrize("angle", [5, 12, 25, 40, -18, -33])
    def test_deskew_holds_across_angles(self, invoice_page, angle):
        skewed = geometry.rotate_sections_on_image(invoice_page, angle, (400, 200))
        measured = geometry.prepare_words(skewed)[0]["rotation_angle"]
        upright = geometry.prepare_words(
            geometry.rotate_sections_on_image(skewed, measured, (400, 200))
        )
        assert upright[0]["rotation_angle"] == pytest.approx(0.0, abs=0.01)


class TestMerging:
    def test_merged_quad_bounds_all_inputs(self, invoice_words):
        merged = geometry.merge_sections(invoice_words[:3])
        assert geometry.quad_std_to_rect_std(merged["quad"]) == [10, 10, 320, 30]

    def test_merged_text_joins_in_order(self, invoice_words):
        assert geometry.merge_sections(invoice_words[:3])["text"] == "Invoice Number INV2024A"

    def test_merging_nothing_gives_an_empty_section(self):
        assert geometry.merge_sections([]) == {}
