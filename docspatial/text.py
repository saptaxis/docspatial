import re
import string
from difflib import SequenceMatcher

import numpy as np
import torch
from nltk.util import ngrams
from sentence_transformers.util import cos_sim
from thefuzz import fuzz
from thefuzz import process as thefuzz_process

from models.embedding import TextEmbedding_MiniLM


def deduplicate_sections_by_text(sections, threshold=0.9):
    unique_sections = []

    for section in sections:
        similarity_scores = []
        for usection in unique_sections:
            this_score = SequenceMatcher(
                None, section["text"], usection["text"]
            ).ratio()
            similarity_scores.append(this_score)
        if not any([score > threshold for score in similarity_scores]):
            unique_sections.append(section)
    return unique_sections


def get_most_dissimilar_text(all_text, num_to_sample, return_idxs=True):
    print(f"Find Dis-similar text: sampling {num_to_sample} from {len(all_text)}")

    if num_to_sample > len(all_text):
        raise ValueError("num_to_sample should be less than total text samples")

    if num_to_sample == len(all_text):
        if return_idxs:
            return all_text, list(range(len(all_text)))
        return all_text

    # return empty if 0
    if num_to_sample == 0:
        if return_idxs:
            return [], []
        return []

    # return random if 1
    if num_to_sample == 1:
        sample_idx = np.random.randint(len(all_text))
        if return_idxs:
            return [all_text[sample_idx]], [sample_idx]
        return [all_text[sample_idx]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed = TextEmbedding_MiniLM(device=device)

    all_text_embeddings = embed.get_embedding(all_text)
    similarity_matrix = cos_sim(all_text_embeddings, all_text_embeddings)

    # start randomly from somewhere
    sampled_idxs = [np.random.randint(len(all_text))]

    for _ in range(num_to_sample - 1):
        # similarity score with existing samples
        sel_sim = similarity_matrix[:, sampled_idxs]
        # calculate mean
        sel_mean = torch.mean(sel_sim, axis=1)
        # ignore selected samples by setting them to 1
        sel_mean[sampled_idxs] = 1
        # set min cosine similarity to find the next idx
        next_idx = torch.argmin(sel_mean).item()
        sampled_idxs.append(next_idx)

    sampled_text = [all_text[idx] for idx in sampled_idxs]
    if return_idxs:
        return sampled_text, sampled_idxs
    return sampled_text


def is_na(value, include_none=False, debug=False):
    actual_na_check = value != value
    none_check = include_none and value is None
    return actual_na_check or none_check


def is_numeric(word):
    allowed_characters = set(string.digits + string.punctuation + " ")
    return all(char in allowed_characters for char in word)


def is_english_or_numeric(word):
    return word.isalnum() and all(char.isascii() for char in word)


def is_english_or_numeric_with_punctuation(word):
    allowed_characters = set(
        string.ascii_letters + string.digits + string.punctuation + " "
    )
    return all(char in allowed_characters for char in word)


def is_naked_punctuation(word):
    return all(char in string.punctuation for char in word)


def is_id_code(s, only_upper=False, quant_number_threshold=10):
    """
    Regex explainer:
    - `^` : Start of the string.
    - `(?=.*[0-9])` : Positive lookahead for any position followed by zero or more characters and then a digit.
    - `(?=.*[a-zA-Z])` : Positive lookahead for any position followed by zero or more characters and then a letter.
    - `([a-zA-Z0-9]+)` : One or more alphanumeric characters.
    - `$` : End of the string.
    """
    pattern = r"^(?=.*[0-9])(?=.*[a-zA-Z])([a-zA-Z0-9]+)$"
    if only_upper:
        pattern = r"^(?=.*[0-9])(?=.*[A-Z])([A-Z0-9]+)$"
    alphanum_id = bool(re.match(pattern, s))

    # uninterrupted number
    not_quant_number = s.isdigit() and len(s) >= quant_number_threshold
    return alphanum_id or not_quant_number


# def is_readable_number(s, quantifiable_number=False):
#     """Specifically checks for numbers that are formatted with commas and decimals.

#     Check for International and Indian formatted numbers.

#     Regex explainer:
#     - `^`: Asserts position at start of the string.
#     - `[+-]?`: Matches an optional plus or minus sign at the beginning.
#     - `(\d+|\d{1,2}(,\d{2})*|\d{1,3}(,\d{2})*(,\d{3})+)`: This is the main group matching either:
#       - `\d+`: A sequence of digits without any commas—handles plain numbers without formatting.
#       - `\d{1,2}(,\d{2})*`: Indian numbering format for numbers less than a lakh (1,00,000), matching 1 or 2 digits followed by zero or more groups of a comma and 2 digits.
#       - `\d{1,3}(,\d{2})*(,\d{3})+`: The typical international numbering format (like 1,234,567) and also covering numbers above 1 lakh in Indian format. It matches 1 to 3 digits, followed by zero or more groups of a comma and 2 digits and at least one group of a comma and 3 digits.
#     - `(\.\d+)?`: Matches an optional decimal part consisting of a period followed by one or more digits.
#     - `$`: Asserts position at the end of the string.
#     """
#     pattern = r"^[+-]?(\d+|\d{1,2}(,\d{2})*|\d{1,3}(,\d{2})*(,\d{3})+)(\.\d+)?$"

#     if quantifiable_number:
#         pattern = (
#             r"^[+-]?(\d{1,2}(,\d{2})+|\d{1,3}(,\d{3})+(\.\d+)?|\d+(,\d{2})*\.\d+)$"
#         )
#     return bool(re.match(pattern, s))


def is_readable_number(text, quantifiable_number=False):
    text = text.strip()

    # remove space from the middle
    text = text.replace(" ", "")

    # check if all valid chars are numbers
    all_numbers = filter_punctuation(text, include_spaces=True).isdigit()

    # split decimal and number separately
    decimal_splits = text.split(".")
    if len(decimal_splits) > 2:
        return False

    # split by decimal
    number = decimal_splits[0]
    decimals = decimal_splits[1] if len(decimal_splits) == 2 else ""

    # comma in decimal
    if decimals and "," in decimals:
        return False

    decimals_found = True if decimals else False
    commas_found = False

    last_group_check = True
    first_group_check = True
    middle_group_check = True

    # if commas, check whether meaningful
    if number and "," in number:
        commas_found = True
        # split by commas
        splits = number.split(",")

        # middle group length - to decide international vs. indian format
        middle_group_length = None
        if len(splits) >= 3:
            middle_group_length = len(splits[1])
            if middle_group_length not in [2, 3]:
                middle_group_length = None

        # last group should be 3
        last_group_check = len(splits[-1]) == 3

        # first group should be 1-2 or 1-3
        first_group_length_options = [1, 2, 3]
        if middle_group_length == 3:
            # international format
            first_group_length_options = [1, 2, 3]
        elif middle_group_length == 2:
            first_group_length_options = [1, 2]

        first_group_check = len(splits[0]) in first_group_length_options

        # middle groups should be 2
        middle_group_check = all([len(s) == middle_group_length for s in splits[1:-1]])

    # print([all_numbers, last_group_check, first_group_check, middle_group_check])

    num_check = all(
        [all_numbers, last_group_check, first_group_check, middle_group_check]
    )

    # quantifiable numbers should have commas or decimals
    if quantifiable_number and num_check:
        if not (commas_found or decimals_found):
            num_check = False

    return num_check


def is_title_case(text):
    # Prepositions and other words typically not capitalized in titles
    exceptions = {
        "a",
        "an",
        "the",
        "at",
        "by",
        "for",
        "in",
        "of",
        "on",
        "to",
        "up",
        "and",
        "as",
        "but",
        "or",
        "nor",
    }
    words = text.split()
    capitalized_words = [
        word for word in words if word.capitalize() == word or word in exceptions
    ]

    if not len(capitalized_words):
        return False

    # Title begins with a capital word and all other words are either exceptions or capitalized
    return words[0].capitalize() == words[0] and len(words) == len(capitalized_words)


def find_string(string_, query, threshold=0.85, lower=False):
    """Find query string on longer string"""
    if lower:
        string_ = string_.lower()
        query = query.lower()

    query_length = len(query.split())
    max_sim_val = 0
    max_sim_string = ""

    for ngram in ngrams(string_.split(), query_length + int(0.2 * query_length)):
        string_ngram = " ".join(ngram)
        similarity = SequenceMatcher(None, string_ngram, query).ratio()
        if similarity > max_sim_val:
            max_sim_val = similarity
            max_sim_string = string_ngram

    if max_sim_val >= threshold:
        return max_sim_string, max_sim_val
    else:
        return None, None


def filter_punctuation(
    str_, include_spaces=False, include_newline=False, replace_with=""
):
    """Remove punctuations from string"""
    trans_table = string.punctuation
    if include_spaces:
        trans_table = trans_table + " "
    if include_newline:
        trans_table += "\n"
    trans_dict = str.maketrans({key: replace_with for key in trans_table})
    return str_.translate(trans_dict)


def remove_leading_punctuations_and_spaces(text):
    return re.sub(r"^[^\w\d]+", "", text)


def remove_trailing_punctuations_and_spaces(text):
    return text.rstrip(string.punctuation + string.whitespace)


def remove_spaces_after_chars(text, chars=["-", "/"]):
    pattern = f"({'|'.join(re.escape(char) for char in chars)})\s+"
    return re.sub(pattern, r"\1", text)


def properly_space_text_with_punctuation(text):
    return re.sub(r"\s([{}])".format(re.escape(string.punctuation)), r"\1", text)


def split_text_into_words(text, split_punctuation=True):
    if split_punctuation:
        words = re.findall(r"\b\w+\b", text)
    else:
        words = text.split()
    return words


def find_phrase(phrase, text):
    """Find phrase in text"""
    text = text.lower()
    phrase = phrase.lower()
    # Escape special regex characters in phrase and replace spaces with \s*
    pattern = re.escape(phrase).replace(r"\ ", r"\s*")
    # Compile regex pattern to find all occurrences of the phrase
    regex = re.compile(pattern)
    # Find all matching phrases in the text
    matches = regex.findall(text)
    return matches


def get_fuzzy_string_match(str1, str2):
    """Return fuzzy string match score between two strings."""
    if str1 is None:
        str1 = ""
    if str2 is None:
        str2 = ""

    str1 = str1.lower()
    str2 = str2.lower()
    match_score = SequenceMatcher(isjunk=None, a=str1, b=str2)
    return match_score.ratio()


def fuzzy_dedupe_list(contains_dupes, threshold=70, scorer=fuzz.token_set_ratio):
    extractor = []

    # iterate over items in *contains_dupes*
    for item in contains_dupes:
        # return all duplicate matches found
        matches = thefuzz_process.extract(
            item, contains_dupes, limit=None, scorer=scorer
        )
        # filter matches based on the threshold
        filtered = [x for x in matches if x[1] > threshold]
        # if there is only 1 item in *filtered*, no duplicates were found so append to *extracted*
        if len(filtered) == 1:
            extractor.append(filtered[0][0])
        elif len(filtered) == 0:
            pass
        else:
            # alpha sort
            filtered = sorted(filtered, key=lambda x: x[0])
            # length sort
            filter_sort = sorted(filtered, key=lambda x: len(x[0]), reverse=True)
            # take first item as our 'canonical example'
            extractor.append(filter_sort[0][0])

    # uniquify *extractor* list
    keys = {}
    for e in extractor:
        keys[e] = 1
    extractor = keys.keys()

    # check that extractor differs from contain_dupes (e.g. duplicates were found)
    # if not, then return the original list
    if len(extractor) == len(contains_dupes):
        return contains_dupes
    else:
        return extractor
