# google_vision.py

"""Google Cloud Vision -> docspatial standard format.

Parses Google Vision DOCUMENT_TEXT_DETECTION output (the full-text-annotation
hierarchy of blocks / paragraphs / words / symbols) into the word dicts the
rest of the library expects.

Optional adapter. Requires google-cloud-vision, which is not a core dependency.

Last verified against the Google Vision API in 2024. The response schema may
have moved since; treat it as a working reference rather than a maintained
integration.
"""
import enum
import json
import os
import pickle

import numpy as np

from .. import geometry, live_ocr, text, utils


def run_google_ocr(
    image_path, output_path=None, client=None, force_run=False, latest=False
):
    if client is None:
        raise ValueError(
            "Pass an authenticated google.cloud.vision client. Credential setup "
            "is left to the caller so this adapter stays free of any particular "
            "deployment's auth."
        )

    run_ocr = True
    if output_path:
        try:
            response = get_ocr_json(output_path)
            run_ocr = False
            if not response:
                run_ocr = True
        except Exception as e:
            run_ocr = True

    if force_run:
        run_ocr = True

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if run_ocr:
        from google.cloud import vision

        image_bytes = utils.get_image_bytes(image_path)
        if latest:
            response = client.annotate_image(
                {
                    "image": {"content": image_bytes},
                    "features": [
                        {
                            "type_": vision.Feature.Type.DOCUMENT_TEXT_DETECTION,
                            "model": "builtin/latest",
                        }
                    ],
                }
            )
        else:
            image = vision.Image(content=image_bytes)
            response = client.document_text_detection(image=image)

    if output_path and run_ocr:
        with open(output_path, "wb") as f:
            pickle.dump(response, f)
    return response


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


def get_ocr_json(ocr_path):
    """Load Google OCR output from a pickle or json file.

    Will return the ocr_json as it is if already a dict.

    Parameters
    ----------
    ocr_path : str or dict
        path to ocr file pickle or JSON
    """
    ocr_json = None
    if isinstance(ocr_path, str) and os.path.exists(ocr_path):
        extn = os.path.splitext(ocr_path)[1]
        if extn == ".pkl":
            with open(ocr_path, "rb") as f:
                doc_ocr = pickle.load(f)
            ocr_json = _AnnotateImageResponse.to_dict(doc_ocr)
        elif extn == ".json":
            with open(ocr_path, "r") as f:
                ocr_json = json.load(f)
    elif isinstance(ocr_path, dict):
        ocr_json = ocr_path
    elif isinstance(ocr_path, _AnnotateImageResponse):
        ocr_json = _AnnotateImageResponse.to_dict(ocr_path)

    if ocr_json is None:
        raise Exception("Invalid OCR.")
    return ocr_json


def is_empty_ocr_json(ocr_json):
    """Check if OCR JSON is empty."""
    key_names = get_ocr_json_key_names(ocr_json)
    ta_key = key_names["ta"]
    if not ocr_json.get(ta_key, []):
        return True
    return False


def get_page_ocr_text(ocr_pkl_path, ascii_encode=False, newline_replace=None):
    ocr_json = get_ocr_json(ocr_pkl_path)
    if is_empty_ocr_json(ocr_json):
        return ""

    key_names = get_ocr_json_key_names(ocr_json)
    fta_key = key_names["fta"]

    entire_ocr_text = ocr_json[fta_key]["text"]
    if ascii_encode:
        entire_ocr_text = entire_ocr_text.encode("ascii", "ignore").decode()

    if newline_replace:
        entire_ocr_text = entire_ocr_text.replace("\n", newline_replace)
    return entire_ocr_text


def get_page_ocr_chars(ocr_pkl_path):
    ocr_json = get_ocr_json(ocr_pkl_path)
    if is_empty_ocr_json(ocr_json):
        raise

    key_names = get_ocr_json_key_names(ocr_json)
    fta_key = key_names["fta"]
    quad_key = key_names["quad"]
    lang_key = key_names["language"]

    page_ocr = ocr_json[fta_key]["pages"][0]

    symbol_list = []

    for block in page_ocr["blocks"]:
        for paragraph in block["paragraphs"]:
            for word in paragraph["words"]:
                for symbol in word["symbols"]:
                    symbol_data = {}
                    symbol_data["text"] = symbol["text"]
                    symbol_data["confidence"] = symbol["confidence"]

                    symbol_vertices = symbol[quad_key]["vertices"]
                    symbol_data["quad"] = symbol_vertices
                    symbol_data["centroid"] = geometry.get_centroid(symbol_vertices)
                    symbol_data["rect"] = geometry.quad_ocr_to_rect_std(symbol_vertices)
                    symbol_data["height"] = (
                        symbol_vertices[3]["y"] - symbol_vertices[0]["y"]
                    )
                    symbol_data["width"] = (
                        symbol_vertices[1]["x"] - symbol_vertices[0]["x"]
                    )
                    symbol_data["rotation_angle"] = geometry.calculate_rotation_angle(
                        symbol_vertices[0], symbol_vertices[1], nearest_angle=1
                    )

                    symbol_language = (
                        word.get("property", {})
                        .get(lang_key, [{}])[0]
                        .get("language_code", "unknown")
                    )
                    if text.is_english_or_numeric(symbol["text"]):
                        symbol_language = "en"
                    symbol_data["language"] = symbol_language

                    symbol_list.append(symbol_data)

    symbol_list = geometry.sort_sections_as_document(symbol_list, method="centroid")
    for idx, s in enumerate(symbol_list):
        s["id"] = idx

    return symbol_list


def get_page_ocr_words(
    ocr_pkl_path,
    remove_punctuation=False,
):
    ocr_json = get_ocr_json(ocr_pkl_path)
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


def extract_sections_from_ocr(
    ocr_pkl_path, associate_word_ids=False, associate_words=None
):
    """Extract blocks and paragraph sections from OCR JSON.

    Parameters
    ----------
    ocr_pkl_path : str or dict
        ocr json or pkl path or ocr_json
    words : list of words
        if provided, will assign word IDs to sections
    """

    ocr_json = get_ocr_json(ocr_pkl_path)
    if is_empty_ocr_json(ocr_json):
        return [], []

    # Handle legacy OCR JSONs
    key_names = get_ocr_json_key_names(ocr_json)
    fta_key = key_names["fta"]
    quad_key = key_names["quad"]
    dtbreak_key = key_names["dtbreak"]
    dtbreak_type_key = key_names["dtbreak_type"]

    page_ocr = ocr_json[fta_key]["pages"][0]

    blocks_data = []
    paragraphs_data = []

    for block in page_ocr["blocks"]:
        block_vertices = block[quad_key]["vertices"]
        block_text = ""
        for paragraph in block["paragraphs"]:
            para_vertices = paragraph[quad_key]["vertices"]
            para_text = ""
            line_text = ""
            for word in paragraph["words"]:
                for symbol in word["symbols"]:
                    line_text += symbol["text"]
                    symbol_prop = symbol.get("property")
                    if symbol_prop:
                        detected_break = symbol_prop.get(
                            dtbreak_key, {dtbreak_type_key: 0}
                        )
                        if detected_break[dtbreak_type_key] in [
                            OCRBreakType.SPACE.value,
                            OCRBreakType.SPACE.name,
                        ]:
                            line_text += " "
                        elif detected_break[dtbreak_type_key] in [
                            OCRBreakType.EOL_SURE_SPACE.value,
                            OCRBreakType.EOL_SURE_SPACE.name,
                        ]:
                            # adding new line here instead of space
                            para_text += "\n" + line_text
                            line_text = ""
                        elif detected_break[dtbreak_type_key] in [
                            OCRBreakType.LINE_BREAK.value,
                            OCRBreakType.LINE_BREAK.name,
                        ]:
                            para_text += "\n" + line_text
                            line_text = ""
            paragraphs_data.append(
                {
                    "quad": para_vertices,
                    "centroid": geometry.get_centroid(para_vertices),
                    "text": para_text.strip(),
                    "confidence": paragraph["confidence"],
                    "section_type": "paragraph",
                }
            )
            block_text += para_text
        blocks_data.append(
            {
                "quad": block_vertices,
                "centroid": geometry.get_centroid(block_vertices),
                "text": block_text.strip(),
                "confidence": block["confidence"],
                "section_type": "block",
            }
        )

    blocks_data = geometry.sort_sections_as_document(blocks_data)
    paragraphs_data = geometry.sort_sections_as_document(paragraphs_data)

    # assign IDs
    for idx, section in enumerate(blocks_data):
        section["id"] = idx
    for idx, section in enumerate(paragraphs_data):
        section["id"] = idx

    # get word IDs
    if associate_word_ids and (associate_words is not None):
        for section in blocks_data:
            words_inside = live_ocr.get_words_inside_section(
                section, associate_words, filter_text=section["text"]
            )
            section["word_ids"] = [w["id"] for w in words_inside]
        for section in paragraphs_data:
            words_inside = live_ocr.get_words_inside_section(
                section, associate_words, filter_text=section["text"]
            )
            section["word_ids"] = [w["id"] for w in words_inside]
    return blocks_data, paragraphs_data


def get_median_word_height(ocr_pkl_path):
    ocr_json = get_ocr_json(ocr_pkl_path)
    if is_empty_ocr_json(ocr_json):
        return 0

    key_names = get_ocr_json_key_names(ocr_json)
    ta_key = key_names["ta"]
    poly_key = key_names["poly"]

    all_words_height = []

    for idx, word in enumerate(ocr_json[ta_key][1:]):
        left, top, right, bottom = geometry.quad_ocr_to_rect_std(
            word[poly_key]["vertices"]
        )
        height = bottom - top
        all_words_height.append(height)
    return np.median(all_words_height)


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
