# geometry

"""
Standardized Representations:

Quadrilateral:
[
    {"x": 0, "y": 0}, # left-top
    {"x": 0, "y": 0}, # right-top
    {"x": 0, "y": 0}, # right-bottom
    {"x": 0, "y": 0}, # left-bottom
]

Rectangle:
[left, top, right, bottom]
"""
import copy
import itertools
import math

import cv2
import numpy as np
import shapely
import torch
import torchvision
from PIL import Image

from . import utils


def get_box_iou(rect1, rect2):
    """Get the IoU of two rectangles"""
    rect1 = torch.tensor([rect1], dtype=torch.float)
    rect2 = torch.tensor([rect2], dtype=torch.float)
    iou = torchvision.ops.boxes.box_iou(rect1, rect2)

    return float(iou.squeeze())


def quad_ocr_to_quad_std(vertices):
    # if isinstance(vertices, google._upb._message.RepeatedCompositeContainer):
    #     return [{"x": v.x, "y": v.y} for v in vertices]
    # else:
    return vertices


def quad_ocr_to_rect_std(vertices):
    # if isinstance(vertices, google._upb._message.RepeatedCompositeContainer):
    #     all_x = [v.x for v in vertices]
    #     all_y = [v.y for v in vertices]
    # else:
    all_x = [v["x"] for v in vertices]
    all_y = [v["y"] for v in vertices]
    left = min(all_x)
    right = max(all_x)
    top = min(all_y)
    bottom = max(all_y)
    return [left, top, right, bottom]


def quad_trp_to_quad_std(x, w, h):
    return [{"x": int(ii.x * w), "y": int(ii.y * h)} for ii in x]


def quad_trt_to_quad_std(x, w, h):
    return [{"x": int(ii["X"] * w), "y": int(ii["Y"] * h)} for ii in x]


def quad_idp_to_quad_std(x):
    return [{"x": ii[0], "y": ii[1]} for ii in x]


def quad_std_to_quad_points_list(x):
    return [[ii["x"], ii["y"]] for ii in x]


def quad_points_list_to_quad_std(x):
    return [{"x": ii[0], "y": ii[1]} for ii in x]


def quad_std_to_rect_std(bbox):
    all_x = [b["x"] for b in bbox]
    all_y = [b["y"] for b in bbox]
    return [int(ii) for ii in [min(all_x), min(all_y), max(all_x), max(all_y)]]


def rect_std_to_quad_std(bbox):
    left, top, right, bottom = bbox
    v1 = (left, top)
    v2 = (right, top)
    v3 = (right, bottom)
    v4 = (left, bottom)
    return [{"x": ii[0], "y": ii[1]} for ii in [v1, v2, v3, v4]]


def rect_std_to_quad_points_list(bbox):
    left, top, right, bottom = bbox
    v1 = [left, top]
    v2 = [right, top]
    v3 = [right, bottom]
    v4 = [left, bottom]
    return [v1, v2, v3, v4]


def rect_cts_to_quad_std(position):
    left = position["left"]
    top = position["top"]
    right = position["right"]
    bottom = position["bottom"]
    bbox = [
        {"x": left, "y": top},
        {"x": right, "y": top},
        {"x": right, "y": bottom},
        {"x": left, "y": bottom},
    ]
    return bbox


def quad_std_to_rect_cts(quad):
    left, top, right, bottom = quad_std_to_rect_std(quad)
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def get_centroid(quad):
    points = quad_std_to_quad_points_list(quad)
    centroid = np.mean(points, axis=0).tolist()
    return centroid


def get_median_height(sections):
    heights = []
    for section in sections:
        section_bbox = quad_std_to_rect_std(section["quad"])
        heights.append(section_bbox[3] - section_bbox[1])
    return np.median(heights)


def words_are_in_same_line(word1, word2, vertical_tolerance_factor=0.8):
    # Calculate vertical tolerance as the average font size of the two words
    average_font_size = (word1["font_height"] + word2["font_height"]) / 2
    # Adjust multiplier based on empirical data
    vertical_tolerance = average_font_size * vertical_tolerance_factor
    return abs(word1["quad"][0]["y"] - word2["quad"][0]["y"]) <= vertical_tolerance


def merge_quads(*quads):
    all_points = list(itertools.chain.from_iterable(quads))
    # if not all_points:
    #     return None

    all_x = [ii["x"] for ii in all_points]
    all_y = [ii["y"] for ii in all_points]

    left = min(all_x)
    right = max(all_x)
    top = min(all_y)
    bottom = max(all_y)

    v1 = {"x": left, "y": top}
    v2 = {"x": right, "y": top}
    v3 = {"x": right, "y": bottom}
    v4 = {"x": left, "y": bottom}

    s_bbox = [v1, v2, v3, v4]
    return s_bbox


def get_merge_sections_confidence(sections, method="mean", ignore_empty=True):
    DEFAULT_NULL_CONFIDENCE = 0.0
    if not len(sections):
        return DEFAULT_NULL_CONFIDENCE

    all_conf = [ii.get("confidence", -1) for ii in sections]
    if ignore_empty:
        all_conf = [ii for ii in all_conf if ii >= 0]

    if not len(all_conf):
        return DEFAULT_NULL_CONFIDENCE

    if method == "mean":
        return np.mean(all_conf)
    elif method == "max":
        return np.max(all_conf)
    elif method == "min":
        return np.min(all_conf)
    elif method == "median":
        return np.median(all_conf)
    else:
        raise Exception("Invalid method. Use 'mean', 'median', 'max', or 'min'.")


def merge_sections(sections, text_delimiter=" "):
    if not len(sections):
        return {}

    all_quads = [ii["quad"] for ii in sections]

    merged_quad = None
    if len(all_quads):
        merged_quad = merge_quads(*all_quads)

    all_text = text_delimiter.join([ii["text"] for ii in sections])

    merged_section = {"text": all_text}
    if merged_quad:
        merged_section["quad"] = merged_quad

    # get confidence
    # mean is not the best way to calculate confidence
    merged_section["confidence"] = get_merge_sections_confidence(
        sections, method="mean"
    )

    return merged_section


def scale_section(section, image_size, x_scale_factor=1.0, y_scale_factor=1.0):
    if x_scale_factor == 1.0 and y_scale_factor == 1.0:
        return section

    image_width, image_height = image_size

    centroid_x, centroid_y = get_centroid(section["quad"])
    vertices = quad_std_to_quad_points_list(section["quad"])

    expanded_vertices = []
    for x, y in vertices:
        scaled_x = centroid_x + (x - centroid_x) * x_scale_factor
        scaled_y = centroid_y + (y - centroid_y) * y_scale_factor
        # clamp to image bounds
        scaled_x = max(0, min(scaled_x, image_width - 1))
        scaled_y = max(0, min(scaled_y, image_height - 1))
        expanded_vertices.append([scaled_x, scaled_y])

    expanded_section_quad = quad_points_list_to_quad_std(expanded_vertices)

    expanded_section = copy.deepcopy(section)
    expanded_section["quad"] = expanded_section_quad

    return expanded_section


def crop_section_from_image(image, section):
    left, top, right, bottom = quad_std_to_rect_std(section["quad"])

    pil_img = False
    if isinstance(image, np.ndarray):
        cropped_image = image[top:bottom, left:right]
    elif isinstance(image, Image.Image):
        pil_img = True
        cropped_image = np.array(image)
        cropped_image = cropped_image[top:bottom, left:right]

    if pil_img:
        return Image.fromarray(cropped_image)

    return cropped_image


def calculate_rotation_angle(point1, point2, nearest_angle=None):
    """
    Calculate rotation angle given two points by computing the slope
    """
    # Calculate the angle between the top edge and the horizontal axis
    dx = point2["x"] - point1["x"]
    dy = point2["y"] - point1["y"]
    angle_rad = math.atan2(dy, dx)

    # Convert radians to degrees
    angle_deg = math.degrees(angle_rad)

    if nearest_angle is not None:
        angle_deg = utils.round_to_nearest_multiple(angle_deg, nearest_angle)

    return angle_deg


def pairwise_distances(points_set1, points_set2):
    """Pair-wise Distances between 2 sets of points"""
    if not isinstance(points_set1, np.ndarray):
        points_set1 = np.array(points_set1)
    if not isinstance(points_set2, np.ndarray):
        points_set2 = np.array(points_set2)
    return np.linalg.norm(points_set1[:, np.newaxis] - points_set2, ord=2, axis=2)


def pairwise_angles(points_set1, points_set2, degrees=False):
    """Pair-wise Angles between 2 sets of points.

    Angles are in radians by default. Set degrees=True to get angles in degrees.
    """
    if not isinstance(points_set1, np.ndarray):
        points_set1 = np.array(points_set1)
    if not isinstance(points_set2, np.ndarray):
        points_set2 = np.array(points_set2)
    points_diff = points_set2[np.newaxis, :, :] - points_set1[:, np.newaxis, :]
    angles_rad = np.arctan2(points_diff[:, :, 1], points_diff[:, :, 0])

    if degrees:
        angles_deg = np.degrees(angles_rad)
        return angles_deg
    return angles_rad


def affine_transform_sections(sections, M):
    """Affine transform sections using affine matrix."""
    if not len(sections):
        return sections

    M = np.array(M)

    if M.shape != (2, 3):
        raise Exception("Expected Affine Matrix of Size 2x3.")

    # keys_to_copy = ["text", "confidence", "rotation_angle", "id"]

    num_sections = len(sections)

    # get all sections quad points -- N, 4, 2
    all_sections_points = np.array(
        [quad_std_to_quad_points_list(ii["quad"]) for ii in sections]
    )

    # convert to -- Nx4, 2
    all_sections_points_flat = all_sections_points.reshape(-1, 2)
    # convert to --Nx4, 3 (for multiplication with affine matrix)
    all_sections_points_flat = np.hstack(
        [all_sections_points_flat, np.ones((all_sections_points_flat.shape[0], 1))]
    )

    # affine transform
    tr_all_sections_points_flat = all_sections_points_flat @ M.T

    # get back to -- N, 4, 2
    tr_all_sections_points = np.array(
        np.array_split(tr_all_sections_points_flat, num_sections)
    ).astype("int")

    # calculate centroid -- N, 2
    tr_all_sections_centroid = np.mean(tr_all_sections_points, axis=1)

    transformed_sections = []
    for idx, section in enumerate(sections):
        tr_section = copy.deepcopy(section)
        # for k in keys_to_copy:
        #     if k in section:
        #         tr_section[k] = section[k]
        tr_section["vertices"] = tr_all_sections_points[idx]
        tr_section["quad"] = quad_points_list_to_quad_std(tr_all_sections_points[idx])
        tr_section["centroid"] = tr_all_sections_centroid[idx]
        transformed_sections.append(tr_section)
    return transformed_sections


def normalize_coordinates_in_section(section, image_width, image_height):
    if not section.get("quad"):
        return

    # assumes section has bbox
    # finds centroid, height, width if not present
    # if "centroid" not in section:
    section["centroid"] = get_centroid(section["quad"])
    # if ("height" not in section) or ("width" not in section):
    section_bbox = quad_std_to_rect_std(section["quad"])
    section["height"] = section_bbox[3] - section_bbox[1]
    section["width"] = section_bbox[2] - section_bbox[0]
    # if "vertices" not in section:
    section["vertices"] = quad_std_to_quad_points_list(section["quad"])

    section["norm_centroid"] = [
        section["centroid"][0] / image_width,
        section["centroid"][1] / image_height,
    ]
    section["norm_height"] = section["height"] / image_height
    section["norm_width"] = section["width"] / image_width
    norm_vertices = np.array(section["vertices"]) / [
        image_width,
        image_height,
    ]
    section["norm_vertices"] = norm_vertices.tolist()


def normalize_coordinates_in_sections(sections, image_width, image_height):
    for section in sections:
        normalize_coordinates_in_section(section, image_width, image_height)


def offset_section_norm_by_page_num(section, page_num, zero_to_n=True, num_pages=None):
    if (not zero_to_n) and (num_pages is None):
        raise Exception("num_pages is required when zero_to_n is False.")

    # offset_section = {k: v for k, v in section.items()}
    offset_section = copy.deepcopy(section)

    norm_keys = [ii for ii in offset_section.keys() if ii.startswith("norm_")]
    # do not raise an Exception
    # if not len(norm_keys):
    #     raise Exception(
    #         "No normalized keys found in section. Run normalize_coordinates_in_section first."
    #     )

    y_offset = int(page_num) - 1

    # should this be zero_to_1? I forget
    if not zero_to_n:
        y_offset = y_offset / num_pages

    # set new page num
    if "page_num" in offset_section:
        offset_section["original_page_num"] = offset_section["page_num"]
    offset_section["page_num"] = page_num

    if not y_offset:
        return offset_section

    # height is relative, so no need to offset
    # norm_keys_to_offset = ["norm_centroid", "norm_vertices"]

    if "norm_vertices" in offset_section:
        offset_section["norm_vertices"] = [
            [v[0], v[1] + y_offset] for v in offset_section["norm_vertices"]
        ]

    if "norm_centroid" in offset_section:
        offset_section["norm_centroid"][1] += y_offset

    return offset_section


def offset_sections_norm_by_page_num(
    sections, page_num, zero_to_n=True, num_pages=None
):
    offset_sections = []
    for section in sections:
        offset_section = offset_section_norm_by_page_num(
            section, page_num, zero_to_n=zero_to_n, num_pages=num_pages
        )
        offset_sections.append(offset_section)
    return offset_sections


def find_point_from_point(source_point, r, theta):
    x = source_point[0] + r * np.cos(theta)
    y = source_point[1] + r * np.sin(theta)
    return [x, y]


def sort_sections_as_document(sections, method="centroid"):
    """Sort sections as a document, top to bottom, left to right"""

    if method not in ["centroid", "top-left"]:
        raise Exception("Invalid method. Use 'centroid' or 'top-left'.")

    if not len(sections):
        return sections

    if method == "centroid":
        median_height = get_median_height(sections)
        for s in sections:
            if "centroid" not in s:
                s["centroid"] = get_centroid(s["quad"])
            s["sort_coords"] = [
                utils.round_to_nearest_multiple(ii, median_height)
                for ii in s["centroid"]
            ]
    elif method == "top-left":
        for s in sections:
            s["sort_coords"] = [s["quad"][0]["x"], s["quad"][0]["y"]]

    sorted_sections = sorted(
        sections,
        key=lambda x: (x["sort_coords"][1], x["sort_coords"][0]),
    )

    for s in sorted_sections:
        s.pop("sort_coords", None)
    return sorted_sections


def get_polygon_iou(poly1, poly2) -> float:
    """
    Get the intersection score (IoU) for two polygons.

    :param poly1: First polygon - shapely.Polygon
    :param poly2: Second polygon.
    :return: Intersection score as a float.
    """
    if not poly1.intersects(poly2):
        return 0.0

    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    iou_score = intersection_area / union_area if union_area else 0

    return iou_score


def _get_rotation_matrix_preserve_image(angle, width, height):
    angle_rad = np.deg2rad(angle)
    final_w = int((height * abs(np.sin(angle_rad))) + (width * abs(np.cos(angle_rad))))
    final_h = int((height * abs(np.cos(angle_rad))) + (width * abs(np.sin(angle_rad))))

    M = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
    M[0, 2] += (final_w / 2) - width // 2
    M[1, 2] += (final_h / 2) - height // 2
    return M, final_w, final_h


def rotate_image(img, angle):
    """Rotate an image, preserving full view of the image"""

    is_np_img = True
    if not isinstance(img, np.ndarray):
        # PIL
        try:
            img = np.array(img)
            is_np_img = False
        except Exception as e:
            print(e)
            print("Input image has to be np.ndarray or PIL Image.")
            return

    h, w = img.shape[:2]

    M, final_w, final_h = _get_rotation_matrix_preserve_image(angle, w, h)

    rot_img = cv2.warpAffine(img, M, (final_w, final_h))

    if not is_np_img:
        # PIL
        return Image.fromarray(rot_img)

    return rot_img


def rotate_points_on_image(quad, angle, image_size):
    """Rotate points on an image, centered on the image, preserving full view of the image

    image_size should be in the original image coordinates
    """
    w, h = image_size

    M, final_w, final_h = _get_rotation_matrix_preserve_image(angle, w, h)

    points = np.array([[ii["x"], ii["y"]] for ii in quad])
    points = np.expand_dims(points, axis=0)
    rot_points = cv2.transform(points, M).squeeze()
    rot_quad = [{"x": int(ii[0]), "y": int(ii[1])} for ii in rot_points]
    return rot_quad


def rotate_sections_on_image(sections, angle, image_size, other_quad_keys_to_rotate=[]):
    """Rotate standard sections on an image, centered on the image,
    preserving full view of the image

    image_size should be in the original image coordinates

    This will rotate the standard 'quad' key. Additional keys can be passed.
    """
    rotated_sections = []
    for section in sections:
        rotated_section = copy.deepcopy(section)
        rotated_section["quad"] = rotate_points_on_image(
            section["quad"], angle, image_size
        )
        if other_quad_keys_to_rotate:
            for other_key in other_quad_keys_to_rotate:
                if other_key not in rotated_section:
                    continue
                rotated_section[other_key] = rotate_points_on_image(
                    section[other_key], angle, image_size
                )
        rotated_sections.append(rotated_section)
    return rotated_sections


# def rotate_points(points_quad, angle, center):
#     cx, cy = center
#     angle_rad = np.deg2rad(angle)
#     rotated_points = [
#         (
#             ((x - cx) * np.cos(angle_rad)) - ((y - cy) * np.sin(angle_rad)) + cx,
#             ((y - cy) * np.cos(angle_rad)) + ((x - cx) * np.sin(angle_rad)) + cy
#         )
#         for x,y in [(ii['x'], ii['y']) for ii in points_quad]
#     ]
#     rotated_quad_std = [{'x': int(ii[0]), 'y': int(ii[1])} for ii in rotated_points]
#     return rotated_quad_std


def find_lines_intersection(line1, line2, img_size=None):
    """Line-Line (infinite) intersection using determinant method

    If img_size is given, then if intersection point lies outside
    it will be considered parallel.

    https://mathworld.wolfram.com/Line-LineIntersection.html
    """
    x1, y1 = [int(ii) for ii in line1[0]]
    x2, y2 = [int(ii) for ii in line1[1]]
    x3, y3 = [int(ii) for ii in line2[0]]
    x4, y4 = [int(ii) for ii in line2[1]]

    p_denom = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))

    if p_denom == 0:
        # parallel or coincident lines
        return []

    px_num = ((x1 * y2 - y1 * x2) * (x3 - x4)) - ((x1 - x2) * (x3 * y4 - y3 * x4))
    py_num = ((x1 * y2 - y1 * x2) * (y3 - y4)) - ((y1 - y2) * (x3 * y4 - y3 * x4))
    px = px_num / p_denom
    py = py_num / p_denom

    if img_size:
        img_w, img_h = img_size
        if (px > img_w) or (py > img_h):
            # parallel line within this page image
            return []

    return [int(px), int(py)]


def get_denormalized_section_from_xywh(x, y, w, h, image_size):
    image_width, image_height = image_size
    x_ = x * image_width
    y_ = y * image_height
    w_ = w * image_width
    h_ = h * image_height

    rect = [x_ - w_ / 2, y_ - h_ / 2, x_ + w_ / 2, y_ + h_ / 2]
    section = {"quad": rect_std_to_quad_std(rect)}
    return section


def get_shapely_polygon_from_section(section):
    if not section.get("quad"):
        return None

    section_vertices = quad_std_to_quad_points_list(section["quad"])
    section_poly = shapely.Polygon(section_vertices)
    return section_poly


def get_shapely_polygons_from_sections(sections):
    section_polygons = []
    for section in sections:
        section_poly = get_shapely_polygon_from_section(section)
        section_polygons.append(section_poly)
    return section_polygons


def find_intersecting_sections(source_section, target_sections):
    source_section_poly = get_shapely_polygon_from_section(source_section)
    target_sections_poly = get_shapely_polygons_from_sections(target_sections)

    intersection_sections = []

    for tidx, tpoly in enumerate(target_sections_poly):
        intrsn = tpoly.intersection(source_section_poly)
        if intrsn.area == 0:
            continue
        intersection_sections.append((target_sections[tidx], intrsn.area))

    intersection_sections = sorted(intersection_sections, key=lambda x: x[1])
    return intersection_sections
