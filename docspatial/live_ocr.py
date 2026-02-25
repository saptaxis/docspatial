# live_ocr.py
import numpy as np
import shapely

from models import geometry, text_utils, utils


def update_section_with_liveocr(
    section, words, uniform_words=False, keep_original_quad=False
):
    merged_section = get_text_section_inside_section(
        section,
        words,
        uniform_words=uniform_words,
        keep_original_quad=keep_original_quad,
    )
    extra_keys = [k for k in section.keys() if k not in merged_section]
    for k in extra_keys:
        merged_section[k] = section[k]
    return merged_section


def get_text_section_inside_section(
    section,
    words,
    uniform_words=False,
    keep_original_quad=False,
    word_intersection_threshold=0,
):
    """Get text inside a section.

    section: section (standard format)
    words: list of word sections (standard format)
    """

    words_inside = get_words_inside_section(
        section, words, word_intersection_threshold=word_intersection_threshold
    )

    if not len(words_inside):
        merged_section = section
        merged_section["text"] = ""
        return merged_section

    # this can help remove watermarks
    # only keep text that is rotated the same way
    # take the median rotation angle of all words
    # have a threshold, and remove words that are not within the threshold
    if uniform_words:
        ROTATION_THRESHOLD = 3
        rotation_angles = [w["rotation_angle"] for w in words_inside]
        median_rotation = np.median(rotation_angles)
        words_inside = [
            w
            for w in words_inside
            if abs(w["rotation_angle"] - median_rotation) < ROTATION_THRESHOLD
        ]

    merged_section = geometry.merge_sections(words_inside)

    if keep_original_quad:
        merged_section["quad"] = section["quad"]

    # assign word IDs to the merged section
    word_ids = [w.get("id") for w in words_inside]
    merged_section["word_ids"] = word_ids

    text = merged_section["text"]

    # text = text_utils.remove_leading_punctuations_and_spaces(text)
    text = text.strip()
    text = text_utils.properly_space_text_with_punctuation(text)

    merged_section["text"] = text
    return merged_section


def get_words_inside_section(
    section, words, filter_text=None, word_intersection_threshold=0
):
    if "vertices" not in section:
        quad_vertices = geometry.quad_std_to_quad_points_list(section["quad"])
    else:
        quad_vertices = section["vertices"]

    search_quad = shapely.Polygon(quad_vertices)
    words_inside_quad = []

    for word in words:
        if "vertices" not in word:
            word["vertices"] = geometry.quad_std_to_quad_points_list(word["quad"])
        word_quad = shapely.Polygon(word["vertices"])
        # iou = geometry.get_polygon_iou(search_quad, word_quad)
        intrsn = search_quad.intersection(word_quad)
        if intrsn.area / word_quad.area > word_intersection_threshold:
            words_inside_quad.append(word)

    if filter_text:
        # Filter for words that are present inside the given filter_text
        words_inside_quad = [
            w for w in words_inside_quad if w["text"].lower() in filter_text.lower()
        ]

    sorted_words_inside = geometry.sort_sections_as_document(words_inside_quad)
    return sorted_words_inside
