# section_detection.py
import string
from collections import defaultdict
from copy import deepcopy

import cv2
import numpy as np
import PIL
from sklearn.cluster import DBSCAN

from . import geometry, live_ocr, ocr


def extract_lines_from_ocr(words, vertical_tolerance_factor=0.8, straight_words=False):
    words_sorted = geometry.sort_sections_as_document(words)

    # consider only words that are in the correct orientation
    if straight_words:
        words_sorted = [w for w in words_sorted if -3 <= w["rotation_angle"] <= 3]

    # words_sorted = sorted(words, key=lambda w: (w["vertices"][0]["y"], w["vertices"][0]["x"]))
    lines = []
    line_id = 0

    # logger.info(f"Debugging words_sorted: {[word['text'] for word in words_sorted]}")

    while words_sorted:
        start_word = words_sorted.pop(0)
        line = [start_word]
        start_word["line_id"] = line_id
        indices_to_pop = []

        for word in words_sorted[:]:
            if geometry.words_are_in_same_line(line[-1], word):
                # word["line_id"] = line_id
                line.append(word)
                indices_to_pop.append(words_sorted.index(word))
                # words_sorted.remove(word)
                # start_word = word

        for index in sorted(indices_to_pop, reverse=True):
            words_sorted.pop(index)

        lines.append(
            sorted(line, key=lambda w: w["quad"][0]["x"])
        )  # Sort words in line by x-coordinate
        line_id += 1

    lines = [geometry.merge_sections(ii) for ii in lines]
    return lines


def convert_xy_to_nparray(xy):
    return np.array([[c["x"], c["y"]] for c in xy])


def union_vertices(vert1, vert2):
    v1 = [min(vert1[0][0], vert2[0][0]), min(vert1[0][1], vert2[0][1])]
    v2 = [max(vert1[1][0], vert2[1][0]), min(vert1[1][1], vert2[1][1])]
    v3 = [max(vert1[2][0], vert2[2][0]), max(vert1[2][1], vert2[2][1])]
    v4 = [min(vert1[3][0], vert2[3][0]), max(vert1[3][1], vert2[3][1])]
    return np.array([v1, v2, v3, v4])


def get_vertices_as_array(blocks):
    output = np.zeros((len(blocks), 4, 2))
    for i, block in enumerate(blocks):
        output[i] = block["vertices"]
    return output


def is_similar_font(f1, f2):
    if (
        f1["font_style"] != f2["font_style"]
        or f1["bold"] != f2["bold"]
        or f1["italic"] != f2["italic"]
    ):
        return False
    if abs(f1["font_size"] - f2["font_size"]) > 0.1 * f1["font_size"]:
        return False
    return True


def split_long_spans(span):
    # Split a long span into smaller spans based on the presence of ':' or '-'
    # And looking backword to see if the previous span is part of the key, or can be separated
    parts = span["span_text"].split()
    span_list = []
    last_span = parts[-1]
    last_2span = " ".join(parts[-2:])
    # remove punctuation from the last span
    last_span = last_span.strip(string.punctuation)
    last_2span = last_2span.strip(string.punctuation)
    if last_span.lower() in [
        "contact",
        "pan",
        "fax",
        "email",
        "gstin",
        "gstinuid",
        "state",
        "ph",
        "phone",
    ]:
        span_list.append({"span_text": last_span, "vertices": span["vertices"]})
        span_list.append({"span_text": parts[-2], "vertices": span["vertices"]})
    # elif last_2span.lower() in ['contact person', 'contact no', 'contact number', 'contact details', 'contact details:', 'contact details :', 'contact no:', 'contact no :', 'contact number:', 'contact numbe


def merge_spans(spans, text=None):
    # Take a list of spans, and concatenate their text, and do a union of their bboxes
    new_span = deepcopy(spans[0])
    for span in spans[1:]:
        new_span["span_text"] += " " + span["span_text"]
        # print("sum", new_span['vertices'])
        # print("new", span['vertices'])
        new_span["vertices"] = union_vertices(new_span["vertices"], span["vertices"])
    # print("final", new_span)
    # new_span['span_type'] = classify_text(new_span['span_text'])
    if text is not None:
        new_span["raw_text"] = text
    return new_span


def sort_blocks(blocks, method="top-left"):
    def top_left_vertex(block):
        sorted_vertices = sorted(block["vertices"], key=lambda v: (v[1], v[0]))
        return sorted_vertices[0]

    def centroid(block):
        x_coords, y_coords = zip(*block["vertices"])
        centroid_x = sum(x_coords) / len(x_coords)
        centroid_y = sum(y_coords) / len(y_coords)
        return centroid_x, centroid_y

    def reference_point(block):
        if method == "centroid":
            return centroid(block)
        elif method == "top-left":
            return top_left_vertex(block)
        else:
            raise ValueError("Invalid method. Choose 'centroid' or 'top-left'.")

    def y_then_x(block):
        x, y = reference_point(block)
        return (y, x)

    sorted_blocks = sorted(blocks, key=y_then_x)
    # add a new variable "sorted_index" to each block
    for i, block in enumerate(sorted_blocks):
        block["sorted_index"] = i

    return sorted_blocks


def find_directional_distance(
    p, q, v, direction, elimination_threshold=10, is_words=False
):
    if is_words:
        p = get_vertices_as_array([p])[0]
        q = get_vertices_as_array(q)

    p = np.array(p[v[0]])
    q = np.array(q[:, v[1]])
    if direction == "north":
        dist = p[1] - q[:, 1]
        abs_dist = abs(dist)
        dist[abs_dist < elimination_threshold] = np.nan
        dist[p[1] < (q[:, 1] - elimination_threshold)] = -np.inf
    elif direction == "south":
        dist = q[:, 1] - p[1]
        abs_dist = abs(dist)
        dist[abs_dist < elimination_threshold] = np.nan
        dist[p[1] > (q[:, 1] + elimination_threshold)] = -np.inf
    elif direction == "west":
        dist = p[0] - q[:, 0]
        abs_dist = abs(dist)
        dist[abs_dist < elimination_threshold] = np.nan
        dist[p[0] < q[:, 0] - elimination_threshold] = -np.inf
    elif direction == "east":
        dist = q[:, 0] - p[0]
        abs_dist = abs(dist)
        dist[abs_dist < elimination_threshold] = np.nan
        dist[p[0] > q[:, 0] + elimination_threshold] = -np.inf
    return dist


def map_neighbors_new(blocks, threshold_base=3.0):
    block_vertices = get_vertices_as_array(blocks)
    direction_params = {
        "south": {"vertices": (0, 0)},
        "east": {"vertices": (0, 0)},
        "north": {"vertices": (0, 0)},
        "west": {"vertices": (0, 0)},
    }

    for i, block in enumerate(blocks):
        block_height = abs(block["vertices"][3][1] - block["vertices"][0][1])
        block_width = abs(block["vertices"][1][0] - block["vertices"][0][0])
        # set thresholds to eliminate candidates from being neighbors
        ew_threshold = threshold_base * block_width
        ew_ortho_threshold = block_width
        ns_threshold = threshold_base * block_height * 2.0
        ns_ortho_threshold = block_height
        # Find the euclidean distance from this block to all other blocks
        east_distances = find_directional_distance(
            block["vertices"],
            block_vertices,
            direction_params["east"]["vertices"],
            "east",
            block["font_metadata"]["font_width"],
        )
        west_distances = find_directional_distance(
            block["vertices"],
            block_vertices,
            direction_params["west"]["vertices"],
            "west",
            block["font_metadata"]["font_width"],
        )
        north_distances = find_directional_distance(
            block["vertices"],
            block_vertices,
            direction_params["north"]["vertices"],
            "north",
            block["font_metadata"]["font_size"],
        )
        south_distances = find_directional_distance(
            block["vertices"],
            block_vertices,
            direction_params["south"]["vertices"],
            "south",
            block["font_metadata"]["font_size"],
        )
        # eliminate the blocks that are too far away in the direction of choice
        east_distances[east_distances > ew_threshold] = np.inf
        west_distances[west_distances > ew_threshold] = np.inf
        north_distances[north_distances > ns_threshold] = np.inf
        south_distances[south_distances > ns_threshold] = np.inf
        # elininate blocks that are too far away in an orthogonal direction
        ortho_north = north_distances > ns_ortho_threshold
        ortho_south = south_distances > ns_ortho_threshold
        ortho_east = east_distances > ew_ortho_threshold
        ortho_west = west_distances > ew_ortho_threshold
        east_distances[ortho_north] = np.inf
        east_distances[ortho_south] = np.inf
        west_distances[ortho_north] = np.inf
        west_distances[ortho_south] = np.inf
        north_distances[ortho_east] = np.inf
        north_distances[ortho_west] = np.inf
        south_distances[ortho_east] = np.inf
        south_distances[ortho_west] = np.inf
        # finally eliminate the block itself
        east_distances[i] = np.inf
        west_distances[i] = np.inf
        north_distances[i] = np.inf
        south_distances[i] = np.inf
        # eliminate the blocks that are irrelevant
        east_distances[east_distances == -np.inf] = np.inf
        west_distances[west_distances == -np.inf] = np.inf
        north_distances[north_distances == -np.inf] = np.inf
        south_distances[south_distances == -np.inf] = np.inf
        # eliminate the blocks that are nan due to elimination_threshold
        east_distances[np.isnan(east_distances)] = np.inf
        west_distances[np.isnan(west_distances)] = np.inf
        north_distances[np.isnan(north_distances)] = np.inf
        south_distances[np.isnan(south_distances)] = np.inf
        # find the minimum distance
        min_east_distance = east_distances.min()
        min_west_distance = west_distances.min()
        min_north_distance = north_distances.min()
        min_south_distance = south_distances.min()
        # if the minimum distance is inf, then there is no block in that direction
        if np.isinf(min_east_distance):
            block["east_block"] = [np.inf, None]
        else:
            block["east_block"] = [min_east_distance, east_distances.argmin()]
        if np.isinf(min_west_distance):
            block["west_block"] = [np.inf, None]
        else:
            block["west_block"] = [min_west_distance, west_distances.argmin()]
        if np.isinf(min_north_distance):
            block["north_block"] = [np.inf, None]
        else:
            block["north_block"] = [min_north_distance, north_distances.argmin()]
        if np.isinf(min_south_distance):
            block["south_block"] = [np.inf, None]
        else:
            block["south_block"] = [min_south_distance, south_distances.argmin()]
    return blocks


def get_page_ocr_grouped_spans(ocr_pkl_path, page_number=0, threshold_base=3.0):
    ocr_json = ocr.get_ocr_json(ocr_pkl_path)

    key_names = ocr.get_ocr_json_key_names(ocr_json)
    fta_key = key_names["fta"]
    bbox_key = key_names["quad"]

    page_ocr = ocr_json[fta_key]["pages"][page_number]
    new_blocks = []
    new_paragraphs = []
    new_spans = []

    for b, block in enumerate(page_ocr["blocks"]):
        block_text = ""
        block_paras = []
        for paragraph in block["paragraphs"]:
            paragraph_text = ""
            paragraph_spans = []
            font_width_calculator = []
            font_height_calculator = []
            another_para = False
            prev_word_line = None
            for word in paragraph["words"]:
                if len(word["symbols"]) > 0:
                    word_vertices = word[bbox_key]["vertices"]
                    vertices = convert_xy_to_nparray(word_vertices)
                    this_word = ""
                    breaks = 0
                    for symbol in word["symbols"]:
                        this_word += symbol["text"]
                        font_width_calculator.append(
                            symbol[bbox_key]["vertices"][1]["x"]
                            - symbol[bbox_key]["vertices"][0]["x"]
                        )
                        font_height_calculator.append(
                            symbol[bbox_key]["vertices"][3]["y"]
                            - symbol[bbox_key]["vertices"][0]["y"]
                        )
                        if "detected_break" in symbol.keys():
                            breaks += 1
                    font_metadata = {
                        "font_style": None,
                        "font_size": np.mean(font_height_calculator),
                        "font_width": np.mean(font_width_calculator),
                        "bold": False,  # TODO: Add logic to detect bold
                        "italic": False,  # TODO: Add logic to detect italic
                    }
                    span = {
                        "span_text": this_word,
                        "vertices": vertices,
                        "block_no": b,
                        "font_metadata": font_metadata,
                    }
                    new_spans.append(span)

                    # Now check if this word actually belongs to the same paragraph, or is in the next line
                    # If it is in the next line, break the paragraph
                    if prev_word_line is None:
                        prev_word_line = vertices[3][1]
                    else:
                        if vertices[3][1] > prev_word_line + font_metadata["font_size"]:
                            another_para = True
                            prev_word_line = vertices[3][1]
                    if another_para:
                        merged_paragraph_span = merge_spans(
                            paragraph_spans, paragraph_text
                        )
                        new_paragraphs.append(merged_paragraph_span)
                        block_text += paragraph_text + "\n"
                        block_paras.append(merged_paragraph_span)
                        paragraph_text = ""
                        paragraph_spans = []
                        another_para = False

                    paragraph_spans.append(span)
                    paragraph_text += this_word + " "
            merged_paragraph_span = merge_spans(paragraph_spans, paragraph_text)
            new_paragraphs.append(merged_paragraph_span)
            block_text += paragraph_text + "\n"
            block_paras.append(merged_paragraph_span)
        merged_block_span = merge_spans(block_paras, block_text)
        new_blocks.append(merged_block_span)
    sorted_blocks = sort_blocks(new_blocks)
    sorted_blocks = map_neighbors_new(sorted_blocks, threshold_base=threshold_base)

    sorted_blocks = convert_to_standard_format(sorted_blocks)
    new_paragraphs = convert_to_standard_format(new_paragraphs)
    new_spans = convert_to_standard_format(new_spans)
    return sorted_blocks, new_paragraphs, new_spans


def convert_to_standard_format(sections):
    for section in sections:
        section["quad"] = geometry.quad_points_list_to_quad_std(section["vertices"])
        section["centroid"] = geometry.get_centroid(section["quad"])
        section["text"] = (
            section["raw_text"] if "raw_text" in section else section["span_text"]
        )
    return sections


def merge_sections__text_guided_visual_method(
    sections, image, words, lines, debug=False
):
    word_heights = [w["height"] for w in words]
    word_heights_ = np.array(word_heights).reshape(-1, 1)

    # epsilon = min_line_spacing / 2
    epsilon = min(word_heights) / 4
    # epsilon = 1

    db = DBSCAN(eps=epsilon, min_samples=1).fit(word_heights_)

    distinct_word_heights = []
    for cluster_label in np.unique(db.labels_):
        cluster_member_indices = np.where(db.labels_ == cluster_label)[0]
        cluster_member_heights = word_heights_[cluster_member_indices]
        distinct_word_heights.append(np.mean(cluster_member_heights))

    distinct_word_heights.sort()

    if debug:
        print("Distinct Word Heights: ", distinct_word_heights)

    # cluster sections based on word heights
    # One could consider using kNN fit on the core samples from dbscan
    # and then use that to assign sections to clusters
    # here simply use the midpoint between distinct word heights as thresholds

    # find threshold range for cluster belonging
    distinct_word_height_thresholds = {}
    for idx in range(len(distinct_word_heights)):
        if idx == 0:
            before = min(word_heights)
        else:
            before = (distinct_word_heights[idx] + distinct_word_heights[idx - 1]) / 2

        if idx == len(distinct_word_heights) - 1:
            after = max(word_heights)
        else:
            after = (distinct_word_heights[idx] + distinct_word_heights[idx + 1]) / 2

        distinct_word_height_thresholds[distinct_word_heights[idx]] = (before, after)

    if debug:
        print("Distinct Word Height Thresholds: ", distinct_word_height_thresholds)

    # cluster the sections using the determined thresholds
    distinct_word_height_sections = defaultdict(list)

    for section in sections:
        if "word_ids" not in section:
            words_inside = live_ocr.get_words_inside_section(
                section, words, filter_text=section["text"]
            )
            section["word_ids"] = [w["id"] for w in words_inside]

        section_word_heights = [
            w["height"] for w in words if w["id"] in section["word_ids"]
        ]

        # assign section to one of the height clusters
        this_section_cluster_map = {
            k: 0 for k in distinct_word_height_thresholds.keys()
        }
        for cluster_height, height_range in distinct_word_height_thresholds.items():
            section_heights_in_range = [
                height_range[0] <= h <= height_range[1] for h in section_word_heights
            ]
            this_section_cluster_map[cluster_height] = sum(
                section_heights_in_range
            ) / len(section_heights_in_range)
        this_section_cluster = max(
            this_section_cluster_map, key=this_section_cluster_map.get
        )
        distinct_word_height_sections[this_section_cluster].append(section)

    if debug:
        print("Section distribution across word heights:")
        for wh, s in distinct_word_height_sections.items():
            print(wh, ":", len(s))

    # Find min line spacing
    sorted_lines = geometry.sort_sections_as_document(lines)

    # find minumum line spacing
    all_line_spacings = []
    for lidx in range(len(sorted_lines) - 1):
        this_line = sorted_lines[lidx]
        next_line = sorted_lines[lidx + 1]
        # top of this to next line
        line_spacing = next_line["quad"][0]["y"] - this_line["quad"][0]["y"]

        # sometimes, lines algo can give overlapping boxes
        # don't want to include such cases
        if line_spacing > 0:
            all_line_spacings.append(line_spacing)

    min_line_spacing = min(all_line_spacings)

    if debug:
        print("Min Line Spacing detected: ", min_line_spacing)

    # merge visually per section cluster
    image = np.array(image)

    page_final_sections = []

    all_bg = []
    all_dil = []

    for word_height, sections in distinct_word_height_sections.items():
        merged_sections = merge_sections__visual_method(
            image, sections, min_line_spacing, debug=debug
        )
        if debug:
            merged_sections, bg, dilated_img, connected_sections = merged_sections
            all_bg.append(bg)
            all_dil.append(dilated_img)
            print(
                f"Word height: {word_height} -- Found {len(connected_sections)} connected sections."
            )
        page_final_sections.extend(merged_sections)

    page_final_sections = geometry.sort_sections_as_document(page_final_sections)

    if debug:
        return page_final_sections, all_bg, all_dil
    return page_final_sections


def merge_sections__visual_method(
    image, sections, min_line_spacing, text_join_delimiter=" \n ", debug=False
):
    """Visual method to merge sections

    Line spacing is used to merge sections that are close to each other
    by doing a dilation on y axis and then finding connected components
    """
    # PARAMETERS
    # a value of line spacing / 2 and iters 3
    # equals 1.5 times linespacing
    height_threshold = int(min_line_spacing / 2)
    num_iterations = 3

    if isinstance(image, PIL.Image.Image):
        image = np.array(image)

    ###

    # create an empty image
    bg = np.zeros(image.shape[:-1], dtype=np.uint8)

    # mask sections on the image
    for section in sections:
        left, top, right, bottom = geometry.quad_std_to_rect_std(section["quad"])
        bg[top:bottom, left:right] = 255

    # Dilate the image to merge them
    # as a function of linespacing
    kernel_width = 1
    # trying to merge on y, so only on height
    kernel_height = height_threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))

    dilated_img = cv2.dilate(bg, kernel, iterations=num_iterations)

    # Connected components with connectivity 4
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dilated_img, connectivity=4
    )

    # get bboxes
    component_bboxes = stats[1:, :4]
    # to rect xyxy format from xywh
    component_bboxes[:, 2] += component_bboxes[:, 0]
    component_bboxes[:, 3] += component_bboxes[:, 1]

    # create merged sections
    connected_sections = []
    for bbox in component_bboxes:
        connected_sections.append({"quad": geometry.rect_std_to_quad_std(bbox)})

    connected_sections_polygons = geometry.get_shapely_polygons_from_sections(
        connected_sections
    )
    sections_polygons = geometry.get_shapely_polygons_from_sections(sections)

    # map actual text sections inside found connected boxes
    sections_to_merge = defaultdict(list)
    for section, section_poly in zip(sections, sections_polygons):
        # has to map to one of the boxes
        # can use 'in' operation
        # or use the max intersection which is guaranteed to always map a section to something
        intersection_scores = []
        for idx, csection_poly in enumerate(connected_sections_polygons):
            intersection = csection_poly.intersection(section_poly)
            intersection_scores.append(intersection.area)

        sections_to_merge[intersection_scores.index(max(intersection_scores))].append(
            section
        )

    final_sections = [
        geometry.merge_sections(v, text_delimiter=text_join_delimiter)
        for k, v in sections_to_merge.items()
    ]

    if debug:
        return final_sections, bg, dilated_img, connected_sections

    return final_sections


def merge_sections__sentence_method(sections, merge_remaining=False):
    """Merge sections based on sentences"""
    buffer = ""
    buffer_sections = []

    merged_sections = []

    for sidx, section in enumerate(sections):
        # if a section does not contain text, skip the section?
        if not len(section["text"]):
            continue

        buffer += section["text"]
        buffer_sections.append(section)
        buffer = buffer.strip()

        # check the last sentence to see if it has ended properly
        # it is ok, if there are multiple sentences in a section
        is_end_of_sent = buffer[-1] in [".", "!", "?"]

        if is_end_of_sent:
            merged_buffers = geometry.merge_sections(buffer_sections)
            merged_sections.append(merged_buffers)
            buffer = ""
            buffer_sections = []

    if merge_remaining:
        if len(buffer_sections):
            merged_buffers = geometry.merge_sections(buffer_sections)
            merged_sections.append(merged_buffers)
    else:
        # keep whatever remains as is, without merging
        merged_sections.extend(buffer_sections)
    return merged_sections
