# find_in_ocr.py
import copy
import itertools
import pprint
from collections import Counter

import numpy as np

from . import geometry, ocr, utils, text_utils


def get_unique_phrase_combinations(combinations):
    """Filter phrase sequences that only have unique word IDs"""
    unique_combinations = []
    for combination in combinations:
        comb_ids = [w["id"] for w in combination]

        if len(comb_ids) == len(set(comb_ids)):
            unique_combinations.append(combination)
        # uniques, counts = np.unique(comb_ids, return_counts=True)
        # if not any(counts > 1):
        #     unique_combinations.append(combination)
    return unique_combinations


def get_similar_word_angle_combinations(combinations, angle_threshold=15):
    """Filter phrase sequences that only have similar word angles

    Use coefficient of variation? or some measure of stdev?
    CV has weird behavior for negative numbers and between quadrants.
    STDEV is not scale invariant.
    Mean has issues with outliers.

    Or most common (coarse) angle with an angle threshold?

    Be gracious with the angle thresholds here - up to 15 degrees?
    """
    similar_angle_combinations = []
    for combination in combinations:
        # round angles to within some degrees
        # check if they are within some degrees from most common
        comb_angles = [
            utils.round_to_nearest_multiple(w["rotation_angle"], angle_threshold)
            for w in combination
        ]

        if len(set(comb_angles)) == 1:
            similar_angle_combinations.append(combination)

        # most_common = Counter(comb_angles).most_common()[0][0]
        # within_range = [
        #     most_common - angle_threshold * 2 <= ii <= most_common + angle_threshold * 2
        #     for ii in comb_angles
        # ]
        # if all(within_range):
        #     similar_angle_combinations.append(combination)

    return similar_angle_combinations


def get_right_or_bottom_combinations(
    combinations,
    x_tolerance_fraction=(2, 2),
    y_tolerance_fraction=(0.5, 2),
    debug=False,
):
    """Filter phrase sequences where successive words are to the right or below"""

    right_or_bottom_combinations = []

    if debug:
        print(f"Number of Combinations: {len(combinations)}")

    for combination in combinations:
        # tolerance +/- some fraction of the Longest word in phrase
        # first_word_bbox = geometry.quad_ocr_to_rect_std(combination[0]['bbox'])
        # first_word_width = first_word_bbox[2] - first_word_bbox[0]
        # first_word_height = first_word_bbox[3] - first_word_bbox[1]

        # Find longest word for x_tolerance
        word_lengths = [len(ii["text"]) for ii in combination]
        longest_word = combination[np.argmax(word_lengths)]
        # longest_word_bbox = geometry.quad_ocr_to_rect_std(longest_word["quad"])
        longest_word_bbox = longest_word["rect"]
        longest_word_width = longest_word_bbox[2] - longest_word_bbox[0]

        # find tallest word for y_tolerance
        # word_bboxes = [geometry.quad_ocr_to_rect_std(ii["quad"]) for ii in combination]
        word_bboxes = [ii["rect"] for ii in combination]
        word_heights = [ii[3] - ii[1] for ii in word_bboxes]
        tallest_word = combination[np.argmax(word_heights)]
        tallest_word_height = max(word_heights)

        # x tolerance 2x width of first word
        x_tolerance_left_fraction, x_tolerance_right_fraction = x_tolerance_fraction
        x_tolerance_left = x_tolerance_left_fraction * longest_word_width
        x_tolerance_right = x_tolerance_right_fraction * longest_word_width

        # y tolerance up 0.5x height of the first word
        # y tolerance down 2x height
        y_tolerance_up_fraction, y_tolerance_down_fraction = y_tolerance_fraction
        y_tolerance_up = y_tolerance_up_fraction * tallest_word_height
        y_tolerance_down = y_tolerance_down_fraction * tallest_word_height

        combination_words_are_right_or_bottom = []

        # # This block ends up being slower than the for loop below
        # # vectorize the checking operation
        # words_x_coords = np.array([w["centroid"][0] for w in combination])
        # words_y_coords = np.array([w["centroid"][1] for w in combination])

        # # check X thresholds - previous word and first word
        # prev_x_distances = np.diff(words_x_coords)
        # first_x_distances = words_x_coords[1:] - words_x_coords[0]

        # # successive right check
        # successive_x_check = (prev_x_distances <= x_tolerance_right) & (
        #     prev_x_distances >= -x_tolerance_left
        # )

        # first_x_check = (first_x_distances <= x_tolerance_right) & (
        #     first_x_distances >= -x_tolerance_left
        # )
        # # it should pass either tolearnce from previous word or first word
        # x_check = successive_x_check | first_x_check

        # # check Y thresholds - previous word
        # prev_y_distances = np.diff(words_y_coords)
        # y_check = (prev_y_distances <= y_tolerance_down) & (
        #     prev_y_distances >= -y_tolerance_up
        # )

        # if all(x_check) and all(y_check):
        #     right_or_bottom_combinations.append(combination)

        prev_word = combination[0]
        first_word = combination[0]
        for word in combination:
            # check right in range first_word or previous_word
            word_is_right_in_range_first = check_word_is_right_of_other_word(
                word, first_word, tolerance=(x_tolerance_left, x_tolerance_right)
            )
            word_is_right_in_range_prev = check_word_is_right_of_other_word(
                word, prev_word, tolerance=(x_tolerance_left, x_tolerance_right)
            )

            word_is_right_in_range = (
                word_is_right_in_range_first or word_is_right_in_range_prev
            )

            word_is_bottom_in_range = check_word_is_bottom_of_other_word(
                word, prev_word, tolerance=(y_tolerance_up, y_tolerance_down)
            )

            # print(f"prev: {prev_word['text']} --- this: {word['text']}")
            # print(f"right check: {word_is_right_in_range}\nbottom check: {word_is_bottom_in_range}")
            # print()

            if word_is_right_in_range and word_is_bottom_in_range:
                combination_words_are_right_or_bottom.append(True)
            else:
                combination_words_are_right_or_bottom.append(False)

            # do this successively for words
            prev_word = word

        if all(combination_words_are_right_or_bottom):
            right_or_bottom_combinations.append(combination)

    return right_or_bottom_combinations


def check_word_is_right_of_other_word(word, other_word, tolerance=(0, 0)):
    x_tolerance_left, x_tolerance_right = tolerance
    word_is_right_in_range = (
        (other_word["centroid"][0] - x_tolerance_left)
        <= word["centroid"][0]
        <= (other_word["centroid"][0] + x_tolerance_right)
    )
    return word_is_right_in_range


def check_word_is_bottom_of_other_word(word, other_word, tolerance=(0, 0)):
    y_tolerance_up, y_tolerance_down = tolerance
    word_is_bottom_in_range = (
        (other_word["centroid"][1] - y_tolerance_up)
        <= word["centroid"][1]
        <= (other_word["centroid"][1] + y_tolerance_down)
    )
    return word_is_bottom_in_range


def find_successive_distance(point_list):
    """Return the sum of successive distances of the given points in order"""
    distances = []
    for idx in range(len(point_list) - 1):
        this_distance = np.linalg.norm(point_list[idx] - point_list[idx + 1], ord=2)
        distances.append(this_distance)
    return np.sum(distances)


# @utils.pass_by_value
def find_phrase_in_ocr(
    phrase,
    ocr_pkl_path,
    words=None,
    read_as_document=True,
    same_line=False,
    case_insensitive=True,
    ignore_punctuation=False,
    normalize_distance_to_height=True,
    distance_tolerance=0.05,
    return_all=False,
    filter_section=None,
    debug=False,
):
    """Find a given phrase in a page using OCR

    Approach:
    - Find all words (multiple occurrences, if any) from OCR list of words
    - Create all combinations of sequences with multiple occurrences
    - Ensure that the combinations have only unique occurrences of words
    - If read_as_document, filter smartly for sequence to be go only left-->right, top-->bottom
    - If same_line, find only sequences that are in the same line
    - Find successive distances between words in a sequence
    - Find minimum successive distance and everything within some tolerance
    - The final phrase is the first combination
    - If return_all, then return all possible combinations that are within tolerance

    Parameters
    ----------
    phrase : str
        phrase to search for
    ocr_pkl_path : str
        local filepath to OCR pickle file
    words : list of dicts
        if provided, then use these words instead of reading from OCR
    read_as_document : bool
        if True, then only use phrases that go left-->right, top-->bottom within some thresholds
    same_line : bool
        if True, then read the phrase as a single line
    case_insensitive : bool
        if True, then ignore case
    ignore_punctuation : bool
        if True, then ignore punctuation words and punctuation chars in words
    normalize_distance_to_height : bool
        if True, then normalize the distance between words to the height of the phrase
    distance_tolerance : float
        tolerance fraction for phrase distances to use to identify final phrase
    return_all : bool
        if True, then return all possible phrase data
    debug : bool
        if True, then print debug statements
    """

    empty = [] if return_all else {}

    # if both read_as_document and same_line are True, then same_line takes precedence
    if read_as_document and same_line:
        read_as_document = False

    if words is None:
        words = ocr.get_page_ocr_words(
            ocr_pkl_path,
            remove_punctuation=ignore_punctuation,
        )
    else:
        # pass by value
        # do not want to alter the original ref to words that come in
        words = copy.deepcopy(words)

    # fitler_section: find phrase within this section bbox
    # filter words to be within this section if passed
    if filter_section:
        words = geometry.find_intersecting_sections(filter_section, words)
        # word[0] because intersection score is also sent with find_intersecting_sections
        words = [word[0] for word in words]

    # convert everything to lower
    if case_insensitive:
        phrase = phrase.lower()
        for w in words:
            w["text"] = w["text"].lower()

    # ignore punctuation
    if ignore_punctuation:
        # if punctuation is found in search phrase, replace with space
        # treat words before and after punctuation as 2 separate words
        phrase = text_utils.filter_punctuation(phrase, replace_with=" ")
        phrase = " ".join(phrase.split())

    # split phrase into words
    words_in_phrase = text_utils.split_text_into_words(
        phrase, split_punctuation=not ignore_punctuation
    )
    words_in_phrase = [ii.strip() for ii in words_in_phrase]
    words_list = []
    words_options_list = []

    # Get all occurrences of the words from the entire document
    for this_word in words_in_phrase:
        this_word_list = [w for w in words if w["text"] == this_word]
        if not len(this_word_list):
            # If any word not found, then return error
            if debug:
                print(f"ERROR: word {this_word} not found in OCR. No Phrase Found.")
            return empty

        words_list.append(this_word)
        words_options_list.append(this_word_list)

    # Check for number of combinations
    num_combinations = 1
    for word_list in words_options_list:
        num_combinations *= len(word_list)
    if num_combinations > 1e6:
        if debug:
            print(f"Number of Combinations too high: {num_combinations} > 1e6. Exiting.")
        return empty

    # Get all combinations of the phrase sequence naively
    all_phrase_combinations = list(itertools.product(*words_options_list))
    final_phrase_combinations = all_phrase_combinations

    if debug:
        print(f"Start - Number of Combinations: {len(final_phrase_combinations)}")

    # Filter for only unique sequences without repeating word instances
    unique_phrase_combinations = get_unique_phrase_combinations(
        final_phrase_combinations
    )
    final_phrase_combinations = unique_phrase_combinations
    if debug:
        print(
            f"After Unique - Number of Combinations: {len(final_phrase_combinations)}"
        )

    # Filter for same line combinations
    if same_line:
        same_line_combinations = get_right_or_bottom_combinations(
            final_phrase_combinations,
            x_tolerance_fraction=(0, 2),
            y_tolerance_fraction=(0.5, 0.5),
        )
        final_phrase_combinations = same_line_combinations
        if debug:
            print(
                f"After Same Line - Number of Combinations: {len(final_phrase_combinations)}"
            )

    # Filter for right or bottom combinations
    if read_as_document:
        # first, find words that are similar angles
        similar_angle_combinations = get_similar_word_angle_combinations(
            final_phrase_combinations
        )
        if debug:
            print(
                f"After Similar Angle - Number of Combinations: {len(similar_angle_combinations)}"
            )

        # then find words that are right or bottom within meaningful thresholds
        right_or_bottom_phrase_combinations = get_right_or_bottom_combinations(
            similar_angle_combinations
        )
        final_phrase_combinations = right_or_bottom_phrase_combinations
        if debug:
            print(
                f"After Right or Bottom - Number of Combinations: {len(final_phrase_combinations)}"
            )

    if not len(final_phrase_combinations):
        if debug:
            print(f"No Phrase Found, finally.")
        return empty

    # get distances
    final_phrase_combination_distances = []

    for combination in final_phrase_combinations:
        quad_list = [ii["quad"] for ii in combination]
        centroid_list = [
            np.mean(geometry.quad_std_to_quad_points_list(quad), axis=0)
            for quad in quad_list
        ]
        this_combination_distance = find_successive_distance(centroid_list)

        if normalize_distance_to_height:
            combination_phrase_poly = geometry.merge_quads(*quad_list)
            combination_phrase_bbox = geometry.quad_std_to_rect_std(
                combination_phrase_poly
            )
            combination_phrase_height = (
                combination_phrase_bbox[3] - combination_phrase_bbox[1]
            )
            this_combination_distance = (
                this_combination_distance / combination_phrase_height
            )
        final_phrase_combination_distances.append(this_combination_distance)

    if debug:
        print("Distances:", final_phrase_combination_distances)

    # Final combinations are the values within some tolerance of minimum value
    min_val = np.min(final_phrase_combination_distances)

    if (min_val == 0) and (distance_tolerance == np.inf):
        tolerance = distance_tolerance
    else:
        tolerance = distance_tolerance * min_val
    possible_min_indexes = [
        idx
        for idx, ii in enumerate(final_phrase_combination_distances)
        if (min_val - tolerance <= ii <= min_val + tolerance)
    ]

    if debug:
        print(
            f"Possible Min Indexes within {distance_tolerance*100}% tolerance: {possible_min_indexes}"
        )

    # min_idx = np.argmin(final_phrase_combination_distances)
    # pick the first one

    all_phrase_data = []
    for min_idx in possible_min_indexes:
        this_combination = final_phrase_combinations[min_idx]
        this_phrase_poly = geometry.merge_quads(
            *[ii["quad"] for ii in this_combination]
        )
        # get confidence as mean of all word OCR confidences
        this_phrase_confidence = np.mean([ii["confidence"] for ii in this_combination])
        this_phrase_data = {
            "text": " ".join([ii["text"] for ii in this_combination]),
            "quad": this_phrase_poly,
            "centroid": geometry.get_centroid(this_phrase_poly),
            "word_ids": [ii["id"] for ii in this_combination],
            "confidence": this_phrase_confidence,
        }
        all_phrase_data.append(this_phrase_data)

    # sort phrases top to bottom, left to right
    all_phrase_data = sorted(
        all_phrase_data, key=lambda x: (x["centroid"][1], x["centroid"][0])
    )

    if return_all:
        return all_phrase_data

    phrase_data = all_phrase_data[0]
    return phrase_data


def find_words_in_same_line_as_phrase(phrase, ocr_pkl_path, text_height_fraction=1.5):
    """Find words in the same line as the given phrase

    Within a tolerance of median text height
    """
    words = ocr.get_page_ocr_words(
        ocr_pkl_path,
        remove_punctuation=False,
    )

    phrase_data = find_phrase_in_ocr(
        phrase, ocr_pkl_path, case_insensitive=True, same_line=True
    )
    if not phrase_data:
        return []

    phrase_centroid = phrase_data["centroid"]

    phrase_bbox = geometry.quad_std_to_rect_std(phrase_data["quad"])
    phrase_data_height = phrase_bbox[3] - phrase_bbox[1]

    text_height_multiplier = phrase_data_height * text_height_fraction

    selected_words = []
    for w in words:
        if (
            phrase_centroid[1] - text_height_multiplier
            <= w["centroid"][1]
            < phrase_centroid[1] + text_height_multiplier
        ):
            selected_words.append(w)
    selected_words = sorted(selected_words, key=lambda x: x["centroid"][0])
    return selected_words
