# layout_utils.py

"""
Utility functions written using layout semantics, OCR 
"""

import numpy as np

from . import find_in_ocr, geometry, ocr


def get_section_to_anchor_directions(anchor_rect, kv_rect):
    """Return possible directions for KV to be in relation to anchor."""
    anchor_left, anchor_top, anchor_right, anchor_bottom = anchor_rect
    left, top, right, bottom = kv_rect

    kv_directions = []

    # South - KV top should be below anchor bottom
    is_south = top > anchor_bottom
    if is_south:
        kv_directions.append("south")

    # North - KV bottom should be above anchor top
    is_north = bottom < anchor_bottom
    if is_north:
        kv_directions.append("north")

    # East - KV left should be to the right of anchor right
    is_east = left > anchor_right
    if is_east:
        kv_directions.append("east")

    # West - KV right should be to the left of anchor left
    is_west = right < anchor_left
    if is_west:
        kv_directions.append("west")
    return kv_directions


def filter_section_with_directions_from_anchor(
    kv_sections_list, anchor_section, direction=""
):
    """Filter KV sections based on direction from anchor section."""
    if not direction:
        return kv_sections_list

    VALID_DIRECTIONS = ["east", "west", "north", "south"]

    required_directions = direction.split("-")

    if not all([ii in VALID_DIRECTIONS for ii in required_directions]):
        print(f"Invalid directions specified.\n'{direction}' not in {VALID_DIRECTIONS}")
        return kv_sections_list

    anchor_rect = geometry.quad_std_to_rect_std(anchor_section["quad"])

    filtered_kv_sections = []
    for kv in kv_sections_list:
        kv_directions_check = []
        kv_rect = geometry.quad_std_to_rect_std(kv["quad"])

        kv_directions_check = get_section_to_anchor_directions(anchor_rect, kv_rect)

        if all([ii in kv_directions_check for ii in required_directions]):
            filtered_kv_sections.append(kv)
    return filtered_kv_sections


def find_section_closest_to(sections_list, anchor_section, direction=None):
    """Find the closest KV section to the anchor section, optionally in a specific direction."""
    if not len(sections_list):
        return None

    filtered_kv_sections = sections_list
    if direction:
        filtered_kv_sections = filter_section_with_directions_from_anchor(
            filtered_kv_sections, anchor_section, direction=direction
        )

    if not len(filtered_kv_sections):
        return None

    # get distance to centroid
    anchor_centroid = geometry.get_centroid(anchor_section["quad"])

    distance_to_centroid = []
    for idx, nkv in enumerate(filtered_kv_sections):
        if "centroid" not in nkv:
            nkv["centroid"] = geometry.get_centroid(nkv["quad"])
        dist = np.linalg.norm(
            np.array(nkv["centroid"]) - np.array(anchor_centroid), ord=2
        )
        distance_to_centroid.append(dist)

    closest_kv = filtered_kv_sections[np.argmin(distance_to_centroid)]
    return closest_kv


def filter_kv_text(kv_sections_list, filter_text, case_insensitive=True):
    """Filter key-value sections with keys containing filter text."""
    if filter_text is None:
        all_filter_text = None
    elif isinstance(filter_text, str):
        all_filter_text = [filter_text]
    elif isinstance(filter_text, list):
        all_filter_text = filter_text

    filtered_kv_sections = []

    if filter_text is not None:
        for ftext in all_filter_text:
            if case_insensitive:
                this_filtered_kv_sections = [
                    ii for ii in kv_sections_list if ftext in ii["key_text"].lower()
                ]
            else:
                this_filtered_kv_sections = [
                    ii for ii in kv_sections_list if ftext in ii["key_text"]
                ]
            filtered_kv_sections.extend(this_filtered_kv_sections)
    return filtered_kv_sections


def get_value_text(kv_sections_list, filter_text):
    """Filter kry-values on keys and return the first value text"""
    value = None
    filtered_kv_sections = filter_kv_text(kv_sections_list, filter_text)
    if len(filtered_kv_sections):
        value = filtered_kv_sections[0]["value_text"]
    return value


def get_value_data(kv_sections_list, filter_text):
    """Filter key-values on keys and return the first section"""
    value_data = {}
    filtered_kv_sections = filter_kv_text(kv_sections_list, filter_text)
    if len(filtered_kv_sections):
        value_section = filtered_kv_sections[0]
        value_data = {
            "quad": value_section["value_bbox"],
            "raw_text": value_section["value_text"],
            "normalized_text": value_section["value_text"],
            "confidence": value_section["confidence"],
        }
    return value_data


def get_closest_value_text(
    page_kv_sections, filter_text, anchor_section, direction=None
):
    """Filter key-values on keys and get the value text closest to the anchor section in a specific direction."""
    if filter_text is None:
        all_filter_text = None
    elif isinstance(filter_text, str):
        all_filter_text = [filter_text]
    elif isinstance(filter_text, list):
        all_filter_text = filter_text

    filtered_kv_sections = page_kv_sections

    if filter_text is not None:
        filtered_kv_sections = []
        for ftext in all_filter_text:
            filtered_kv_sections.extend(
                [ii for ii in page_kv_sections if ftext in ii["key_text"].lower()]
            )

    closest_section = find_section_closest_to(
        filtered_kv_sections, anchor_section, direction=direction
    )
    value_text = ""
    if closest_section:
        value_text = closest_section["value_text"]
    return value_text


def find_n_words_after_phrase(words, phrase_data, n=1):
    """Given OCR words and a phrase data section, find the next word"""
    next_id = phrase_data["word_ids"][-1] + 1
    selected_words = [w for w in words if w["id"] in range(next_id, next_id + n)]
    return selected_words


def find_n_numbers_after_phrase(words, phrase_data, n=10):
    """Given OCR words and a phrase data section, find the n next numbers"""
    next_id = phrase_data["word_ids"][-1] + 1

    next_numbers = []

    for w in words:
        if w["id"] < next_id:
            continue
        try:
            word_text = w["text"].replace(",", "")
            if not word_text.startswith("0"):
                # the reason to do this
                # is because OCR sometimes mistakes , for .
                # since we are only checking for a number
                # we can replace and check
                # and if number starts with 0, avoid leading zero error
                word_text = word_text.replace(".", "")
            num = float(word_text)
            next_numbers.append(w)
        except:
            pass

        if len(next_numbers) >= n:
            break
    return next_numbers


def find_closest_word(word, word_list):
    """Find word in word_list closest to word"""
    source_centroid = np.array(geometry.get_centroid(word["quad"]))

    distance_list = []
    for dest_word in word_list:
        dquad = dest_word["quad"]
        dcentroid = np.array(geometry.get_centroid(dquad))
        dist_to_source = np.linalg.norm(source_centroid - dcentroid, ord=2)
        distance_list.append(dist_to_source)

    min_idx = np.argmin(distance_list)
    return word_list[min_idx]


def get_n_next_word_extracted_data(words, phrase_data, n=1, numbers=False):
    all_data = []

    if numbers:
        phrase_next_words = find_n_numbers_after_phrase(words, phrase_data, n=1)
    else:
        phrase_next_words = find_n_words_after_phrase(words, phrase_data, n=1)

    for next_word in phrase_next_words:
        all_data.append(
            {
                "raw_text": next_word["text"],
                "normalized_text": next_word["text"],
                "quad": next_word["quad"],
                "confidence": next_word["confidence"],
            }
        )
    return all_data


def get_next_word_after_phrase(
    phrase, ocr_path, ignore_punctuation=True, numbers=False, n=1
):
    extracted_data = {}

    if numbers:
        ignore_punctuation = False

    all_phrase_data = find_in_ocr.find_phrase_in_ocr(
        phrase,
        ocr_path,
        ignore_punctuation=ignore_punctuation,
        same_line=True,
        normalize_distance_to_height=True,
        distance_tolerance=0.1,
        return_all=True,
    )
    words = ocr.get_page_ocr_words(
        ocr_path,
        remove_punctuation=ignore_punctuation,
    )

    if len(all_phrase_data):
        # choose the first phrase found
        phrase_data = all_phrase_data[0]
        phrase_extraction_data = get_n_next_word_extracted_data(
            words, phrase_data, n=n, numbers=numbers
        )
        if len(phrase_extraction_data):
            extracted_data = phrase_extraction_data[0]
    return extracted_data
