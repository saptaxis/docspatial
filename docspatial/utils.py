# utils.py
import copy
import io
import math
import os
from collections import Counter

import numpy as np
from PIL import Image


def pass_by_value(f):
    def _f(*args, **kwargs):
        args_copied = copy.deepcopy(args)
        kwargs_copied = copy.deepcopy(kwargs)
        return f(*args_copied, **kwargs_copied)

    return _f


def get_chunks(lst, chunk_size, pad_value=None):
    return [
        lst[i : i + chunk_size]
        + [pad_value] * (chunk_size - len(lst[i : i + chunk_size]))
        for i in range(0, len(lst), chunk_size)
    ]


def normalize_number(number, base):
    return number / base


def round_to_nearest_multiple(number, base, method="nearest"):
    if base == 0:
        print("round_to_nearest_multiple: Base cannot be 0.")
        return number
    if method == "nearest":
        return base * round(number / base)
    elif method == "up":
        return base * math.ceil(number / base)
    elif method == "down":
        return base * math.floor(number / base)
    else:
        return base * round(number / base)


def get_image_bytes(image):
    """Return image bytes from Image path or PIL or np image."""
    if isinstance(image, str) and os.path.exists(image):
        with open(image, "rb") as f:
            image_bytes = f.read()
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        io_image_bytes = io.BytesIO()
        image.save(io_image_bytes, format="PNG")
        image_bytes = io_image_bytes.getvalue()
    elif isinstance(image, Image.Image):
        io_image_bytes = io.BytesIO()
        image.save(io_image_bytes, format="PNG")
        image_bytes = io_image_bytes.getvalue()
    else:
        raise Exception("Invalid image type.")
    return image_bytes


def get_pil_image(image):
    """Return image bytes from Image path or PIL or np image."""
    pil_image = None
    if isinstance(image, str) and os.path.exists(image):
        pil_image = Image.open(image)
    elif isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        raise Exception("Invalid image type.")
    return pil_image


def deduplicate_list_by_append(keys):
    counter = Counter(keys)
    occurred = {}
    new_keys = []
    for key in keys:
        if counter[key] > 1:  # Check for repetitions
            occurred[key] = occurred.get(key, 0) + 1
            new_key = f"{key} {occurred[key]}" if occurred[key] > 1 else key
        else:
            new_key = key
        new_keys.append(new_key)
    return new_keys


def compare_lists(list1, list2, match_empty=False):
    if (not list1) or (not list2):
        return 0
    if len(list1) != len(list2):
        return 0

    matches = 0
    for l1, l2 in zip(list1, list2):
        this_match = l1 == l2
        # if one of the items is empty, consider it a match
        if match_empty and ((not l1) or (not l2)):
            this_match = True

        if this_match:
            matches += 1
    # matches = sum(l1 == l2 for l1, l2 in zip(list1, list2))
    total = len(list1)
    match_fraction = matches / total
    return match_fraction


def get_file_extension(file_path):
    return os.path.splitext(file_path)[1].lower()[1:]


def get_file_basename(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]
