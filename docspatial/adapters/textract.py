# textract.py

"""AWS Textract -> docspatial standard format.

Emits standard-format words from a Textract response. Textract returns
normalised coordinates, so a page size is required to denormalise them; that is
the difference from the Google adapter, which returns absolute pixels.

Takes a parsed response or a path to one. Requires trp to walk it. Fetching a
response is the caller's job, so this needs no boto3 and no credentials.

Last verified against the Textract API in 2024. The response schema may have
moved since; treat it as a reference rather than a maintained integration.
"""
import json
import os
import pickle

from PIL import Image

from .. import geometry


def get_trt_dict(trt_path):
    """Get Textract response from path or JSON or dict

    Parameters
    ----------
    trt_path : str or dict
        Textract response pkl/json path or dict

    """
    trt_dict = None
    if isinstance(trt_path, str) and os.path.exists(trt_path):
        extn = os.path.splitext(trt_path)[1]
        if extn == ".pkl":
            with open(trt_path, "rb") as f:
                trt_dict = pickle.load(f)
        elif extn == ".json":
            with open(trt_path, "r") as f:
                trt_dict = json.load(f)
    elif isinstance(trt_path, dict):
        trt_dict = trt_path

    return trt_dict


def get_trt_words(page_img_path, page_trt_path):
    trt_dict = get_trt_dict(page_trt_path)

    import trp

    document = trp.Document(trt_dict)

    assert len(document.pages) == 1  # written for textract per page
    page = document.pages[0]

    if isinstance(page_img_path, str) and os.path.exists(page_img_path):
        img = Image.open(page_img_path)
        width, height = img.size
    elif isinstance(page_img_path, tuple):
        width, height = page_img_path

    word_list = []
    for line in page.lines:
        for word in line.words:
            text = word.text
            text_type = word.block["TextType"]
            word_confidence = word.block["Confidence"]

            quad = geometry.quad_trp_to_quad_std(word.geometry.polygon, width, height)
            word_vertices = quad
            word_rect = geometry.quad_ocr_to_rect_std(word_vertices)
            word_centroid = geometry.get_centroid(word_vertices)
            word_rotation_angle = geometry.calculate_rotation_angle(
                word_vertices[0], word_vertices[1], nearest_angle=1
            )

            number_of_chars = len(text)

            word_list.append(
                {
                    "text": text,
                    "quad": quad,
                    "section_type": "word",
                    "text_type": text_type,
                    "rect": word_rect,
                    "centroid": word_centroid,
                    "confidence": word_confidence,
                    "number_of_chars": number_of_chars,
                    "rotation_angle": word_rotation_angle,
                    "font_height": 0,
                    "font_width": 0,
                    "language": "unknown",
                }
            )

    # sort words by centroid rounded to median font height
    word_list = geometry.sort_sections_as_document(word_list, method="centroid")

    # assign IDs to words
    for idx, w in enumerate(word_list):
        w["id"] = idx

    return word_list
