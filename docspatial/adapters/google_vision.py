# google_vision.py

"""Google Cloud Vision -> docspatial standard format.

Walks the DOCUMENT_TEXT_DETECTION full-text-annotation hierarchy of blocks,
paragraphs, words and symbols, and emits standard-format words. Symbol boxes
give per-word font height and width; coordinates are absolute pixels.

Takes a parsed response. Fetching one is the caller's job, so this needs no
Google SDK and no credentials.

Last verified against the Google Vision API in 2024. The response schema may
have moved since; treat it as a reference rather than a maintained integration.
"""
import enum

from .. import geometry, text


def is_legacy_ocr_format(ocr_json):
    if "textAnnotations" in ocr_json.keys():
        return True
    elif "text_annotations" in ocr_json.keys():
        return False

    # could also possibly be an empty OCR json
    return False


def get_ocr_json_key_names(ocr_json):
    legacy = is_legacy_ocr_format(ocr_json)

    # between bounding_box and bounding_poly
    # if using text_annotations, need to access poly_key
    # if using full_text_annotation, need to access quad_key
    if legacy:
        fta_key = "fullTextAnnotation"
        ta_key = "textAnnotations"
        quad_key = "boundingBox"
        poly_key = "boundingPoly"
        dtbreak_key = "detectedBreak"
        type_key = "type"
        lang_key = "detectedLanguages"
    else:
        fta_key = "full_text_annotation"
        ta_key = "text_annotations"
        quad_key = "bounding_box"
        poly_key = "bounding_poly"
        dtbreak_key = "detected_break"
        type_key = "type_"
        lang_key = "detected_languages"

    keys = {
        "fta": fta_key,
        "ta": ta_key,
        "quad": quad_key,
        "poly": poly_key,
        "dtbreak": dtbreak_key,
        "dtbreak_type": type_key,
        "language": lang_key,
    }
    return keys


def is_empty_ocr_json(ocr_json):
    """Check if OCR JSON is empty."""
    key_names = get_ocr_json_key_names(ocr_json)
    ta_key = key_names["ta"]
    if not ocr_json.get(ta_key, []):
        return True
    return False


def get_page_ocr_words(
    ocr_json,
    remove_punctuation=False,
):
    """Convert a Google Vision response into standard-format words.

    ocr_json is the parsed DOCUMENT_TEXT_DETECTION response. Coordinates are
    absolute pixels, so no page size is needed.
    """
    if is_empty_ocr_json(ocr_json):
        return []

    key_names = get_ocr_json_key_names(ocr_json)
    fta_key = key_names["fta"]
    quad_key = key_names["quad"]
    lang_key = key_names["language"]

    page_ocr = ocr_json[fta_key]["pages"][0]
    word_list = []

    for block in page_ocr["blocks"]:
        for paragraph in block["paragraphs"]:
            for word in paragraph["words"]:
                word_text = "".join([symbol["text"] for symbol in word["symbols"]])
                word_confidence = word["confidence"]

                word_vertices = word[quad_key]["vertices"]
                word_centroid = geometry.get_centroid(word_vertices)
                word_rect = geometry.quad_ocr_to_rect_std(word_vertices)
                word_rotation_angle = geometry.calculate_rotation_angle(
                    word_vertices[0], word_vertices[1], nearest_angle=1
                )
                number_of_chars = len(word_text)

                # font height and width
                symbol_heights = [
                    symbol[quad_key]["vertices"][3]["y"]
                    - symbol[quad_key]["vertices"][0]["y"]
                    for symbol in word["symbols"]
                ]
                symbol_widths = [
                    symbol[quad_key]["vertices"][1]["x"]
                    - symbol[quad_key]["vertices"][0]["x"]
                    for symbol in word["symbols"]
                ]
                font_height = (
                    sum(symbol_heights) / len(symbol_heights) if symbol_heights else 0
                )
                font_width = (
                    sum(symbol_widths) / len(symbol_widths) if symbol_widths else 0
                )

                # word language
                word_language = (
                    word.get("property", {})
                    .get(lang_key, [{}])[0]
                    .get("language_code", "unknown")
                )
                if text.is_english_or_numeric(word_text):
                    word_language = "en"
                # REVIEW
                # elif word_language == "unknown" and text.is_naked_punctuation(
                #     word_text
                # ):
                #     word_language = "unknown"

                # remove punction in words
                if remove_punctuation:
                    word_text = text.filter_punctuation(word_text)

                if not len(word_text):
                    continue

                word_list.append(
                    {
                        "text": word_text,
                        "quad": word_vertices,
                        "section_type": "word",
                        "rect": word_rect,
                        "centroid": word_centroid,
                        "confidence": word_confidence,
                        "number_of_chars": number_of_chars,
                        "rotation_angle": word_rotation_angle,
                        "font_height": font_height,
                        "font_width": font_width,
                        "language": word_language,
                    }
                )

    # sort words by centroid rounded to median font height
    word_list = geometry.sort_sections_as_document(word_list, method="centroid")

    # assign IDs to words
    for idx, w in enumerate(word_list):
        w["id"] = idx

    return word_list


class OCRBreakType(enum.IntEnum):
    # unknown break type
    UNKNOWN = 0
    # regular space
    SPACE = 1
    # sure space (very wide)
    SURE_SPACE = 2
    # line-wrapping break
    EOL_SURE_SPACE = 3
    # end-line hyphen that is not present in text
    HYPHEN = 4
    # line break that ends a paragraph
    LINE_BREAK = 5
