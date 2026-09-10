# textract.py

"""AWS Textract -> docspatial standard format.

Parses Textract responses into the standard section format, including layout
blocks, lines, key-value pairs, signatures and tables.

Optional adapter. Requires boto3 and trp, which are not core dependencies.

Last verified against the Textract API in 2024. The response schema may have
moved since; treat it as a working reference rather than a maintained
integration.
"""
import json
import os
import pickle

from PIL import Image

from .. import geometry, live_ocr, text, utils


def initialize_trt_client(profile_name=None):
    """Initialize Textract client"""
    import boto3

    session = boto3.Session(profile_name=profile_name)
    textract = session.client("textract")
    return textract


def run_trt(
    image_path,
    client=None,
    feature_types=["LAYOUT"],
    output_path=None,
    profile_name=None,
    force_run=False,
):
    """Run Textract on image path"""
    run_textract = True
    if output_path:
        try:
            response = get_trt_dict(output_path)
            run_textract = False
            if not response:
                run_textract = True
        except Exception as e:
            run_textract = True

    if force_run:
        run_textract = True

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if run_textract:
        if client is None:
            client = initialize_trt_client(profile_name=profile_name)

        image_bytes = utils.get_image_bytes(image_path)

        if feature_types == ["OCR"]:
            response = client.detect_document_text(Document={"Bytes": image_bytes})
        else:
            response = client.analyze_document(
                Document={"Bytes": image_bytes}, FeatureTypes=feature_types
            )

    if output_path and run_textract:
        with open(output_path, "wb") as f:
            pickle.dump(response, f)
    return response


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


def get_trt_text(page_trt_path, ascii_encode=False, newline_replace=False):
    """Get Textract text from path or JSON or dict

    Parameters
    ----------
    page_trt_path : str or dict
        Textract response pkl/json path or dict

    """
    trt_dict = get_trt_dict(page_trt_path)

    import trp

    document = trp.Document(trt_dict)

    assert len(document.pages) == 1  # written for textract per page
    page = document.pages[0]
    entire_ocr_text = page.text

    if ascii_encode:
        entire_ocr_text = entire_ocr_text.encode("ascii", "ignore").decode()

    if newline_replace:
        entire_ocr_text = entire_ocr_text.replace("\n", newline_replace)

    return entire_ocr_text


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


def extract_trt_layout_sections(
    page_img_path,
    page_trt_path,
    do_live_ocr=True,
    live_ocr_words=None,
    associate_word_ids=False,
    associate_words=None,
):
    trt_dict = get_trt_dict(page_trt_path)

    # page_img_path can be image path or size tuple
    if isinstance(page_img_path, str) and os.path.exists(page_img_path):
        img = Image.open(page_img_path)
        width, height = img.size
    elif isinstance(page_img_path, tuple):
        width, height = page_img_path

    layout_sections = []

    layout_blocks = [b for b in trt_dict["Blocks"] if "LAYOUT" in b["BlockType"]]

    for block in layout_blocks:
        this_section = {
            "quad": geometry.quad_trt_to_quad_std(
                block["Geometry"]["Polygon"], width, height
            ),
            "section_type": block["BlockType"],
            "confidence": block["Confidence"] / 100,
        }

        if do_live_ocr and (live_ocr_words is not None):
            this_section = live_ocr.update_section_with_liveocr(
                this_section, live_ocr_words
            )

        if associate_word_ids and (associate_words is not None):
            words_inside = live_ocr.get_words_inside_section(
                this_section, associate_words, filter_text=None
            )
            this_section["word_ids"] = [w["id"] for w in words_inside]

        layout_sections.append(this_section)

    layout_sections = geometry.sort_sections_as_document(layout_sections)

    # add ID to sections
    for idx, s in enumerate(layout_sections):
        s["id"] = idx
    return layout_sections


def extract_trt_line_sections(
    page_img_path, page_trt_path, associate_word_ids=False, associate_words=None
):
    trt_dict = get_trt_dict(page_trt_path)

    # page_img_path can be image path or size tuple
    if isinstance(page_img_path, str) and os.path.exists(page_img_path):
        img = Image.open(page_img_path)
        width, height = img.size
    elif isinstance(page_img_path, tuple):
        width, height = page_img_path

    import trp

    document = trp.Document(trt_dict)
    assert len(document.pages) == 1  # written for textract per page
    page = document.pages[0]

    line_sections = []

    for line in page.lines:
        this_section = {
            "quad": geometry.quad_trp_to_quad_std(line.geometry.polygon, width, height),
            "text": line.text,
            "confidence": line.confidence / 100,
            "section_type": "line",
        }

        if associate_word_ids and (associate_words is not None):
            words_inside = live_ocr.get_words_inside_section(
                this_section, associate_words, filter_text=this_section["text"]
            )
            this_section["word_ids"] = [w["id"] for w in words_inside]
        line_sections.append(this_section)

    # sections_data = sorted(sections_data, key=lambda x: x["centroid"][1])
    line_sections = geometry.sort_sections_as_document(line_sections)

    # add ID to sections
    for idx, s in enumerate(line_sections):
        s["id"] = idx

    return line_sections


def extract_trt_kv_sections(
    page_img_path,
    page_trt_path,
    associate_word_ids=False,
    associate_words=None,
    remove_empty=True,
):
    """Extract FORM fields as key-values from Textract response

    Parameters
    ----------
    page_img_path : str or tuple
        page image path or size tuple (width, height)
    page_trt_path : str
        page textract response pkl path
    words: list of ocr word sections
        if provided, will be used to extract OCR word IDs inside each section
    remove_empty : bool
        remove if key or value text is empty

    """
    trt_dict = get_trt_dict(page_trt_path)

    import trp

    document = trp.Document(trt_dict)

    assert len(document.pages) == 1  # written for textract per page
    page = document.pages[0]

    # page_img_path can be image path or size tuple
    if isinstance(page_img_path, str) and os.path.exists(page_img_path):
        img = Image.open(page_img_path)
        width, height = img.size
    elif isinstance(page_img_path, tuple):
        width, height = page_img_path

    sections_data = []

    for field in page.form.fields:
        if field.value is None:
            continue
        if remove_empty:
            if (not len(field.value.text)) or (not len(field.key.text)):
                continue
        # print("Key: {}, Value: {}".format(field.key, field.value))
        k_quad = geometry.quad_trp_to_quad_std(
            field.key.geometry.polygon, width, height
        )
        v_quad = geometry.quad_trp_to_quad_std(
            field.value.geometry.polygon, width, height
        )
        # s_quad = geometry.merge_quads(k_quad, v_quad)
        # s_text = field.key.text + " - " + field.value.text

        key_text = text.remove_trailing_punctuations_and_spaces(field.key.text)

        this_section = {
            "quad": v_quad,
            "text": field.value.text,
            "section_type": "key-value",
            "key_text": key_text,
            "value_text": field.value.text,
            "key_quad": k_quad,
            "value_quad": v_quad,
            # "key_value_quad": s_quad,
            "confidence": field.confidence / 100,
            "centroid": geometry.get_centroid(v_quad),
        }

        if associate_word_ids and (associate_words is not None):
            # get all words in the region
            # filter words so that only the words actually in the values are kept
            words_inside = live_ocr.get_words_inside_section(
                this_section, associate_words, filter_text=this_section["text"]
            )
            this_section["word_ids"] = [w["id"] for w in words_inside]

            # above is same as value word IDs
            this_section["value_word_ids"] = this_section["word_ids"]

            # get key word ids
            key_words_inside = live_ocr.get_words_inside_section(
                {"quad": k_quad}, associate_words, filter_text=this_section["key_text"]
            )
            this_section["key_word_ids"] = [w["id"] for w in key_words_inside]

        sections_data.append(this_section)

    # sections_data = sorted(sections_data, key=lambda x: x["centroid"][1])
    sections_data = geometry.sort_sections_as_document(sections_data)

    # add ID to sections
    for idx, s in enumerate(sections_data):
        s["id"] = idx

    return sections_data


def extract_trt_signature_sections(
    page_img_path,
    page_trt_path,
    associate_word_ids=False,
    associate_words=None,
    kv_sections=None,
    page_num=None,
):
    trt_dict = get_trt_dict(page_trt_path)
    import trp

    document = trp.Document(trt_dict)

    assert len(document.pages) == 1

    page = document.pages[0]

    # page_img_path can be image path or size tuple
    if isinstance(page_img_path, str) and os.path.exists(page_img_path):
        img = Image.open(page_img_path)
        width, height = img.size
    elif isinstance(page_img_path, tuple):
        width, height = page_img_path

    signature_sections = []
    for b in page.blocks:
        if b["BlockType"] == "SIGNATURE":
            quad = geometry.quad_trt_to_quad_std(
                b["Geometry"]["Polygon"], width, height
            )
            print(f"Signature Block: {quad}")

            this_section = {
                "idx": len(signature_sections),
                "quad": quad,
                "text": "",
                "section_type": "signature",
                "key_text": None,
                "value_text": "",
                "key_quad": None,
                "value_quad": quad,
                "confidence": b["Confidence"] / 100,
                "centroid": geometry.get_centroid(quad),
            }

            # do live ocr on the bbox and get the signature text
            if associate_word_ids and (associate_words is not None):
                signature_text_section = live_ocr.get_text_section_inside_section(
                    this_section,
                    associate_words,
                    uniform_words=False,
                    keep_original_quad=True,
                )

                text = signature_text_section["text"]
                this_section["text"] = text
                this_section["value_text"] = text
                print(f"live ocr text found: {text}")

            # if key value sections has an overlapping value quad, then assign key as signature key
            is_key_found = False
            if kv_sections is not None:
                for section in kv_sections:
                    value_quad = section["value_quad"]
                    value_box = geometry.quad_std_to_rect_std(value_quad)
                    sign_box = geometry.quad_std_to_rect_std(quad)
                    if geometry.get_box_iou(value_box, sign_box) >= 0.8:
                        is_key_found = True
                        this_section["key_text"] = section["key_text"]
                        this_section["key_quad"] = section["key_quad"]

            if not is_key_found:
                if page_num is not None:
                    this_section[
                        "key_text"
                    ] = f"Signature - {page_num}_{len(signature_sections) + 1}"
                else:
                    this_section[
                        "key_text"
                    ] = f"Signature - {len(signature_sections) + 1}"

            signature_sections.append(this_section)
    return signature_sections


def extract_trt_table_sections(
    page_img_path, page_trt_path, associate_word_ids=False, associate_words=None
):
    """Extract table sections from Textract Response

    Header names are not properly handled/tested.

    Parameters
    ----------
    page_img_path : str or tuple
        Page image path ot size tuple (width, height)
    page_trt_path : str
        Page textract response pkl path
    words: list of ocr word sections
        if provided, will be used to extract OCR word IDs inside each cell section

    """
    trt_dict = get_trt_dict(page_trt_path)

    import trp

    document = trp.Document(trt_dict)

    assert len(document.pages) == 1  # written for textract per page
    page = document.pages[0]

    # page_img_path can be image path or size tuple
    if isinstance(page_img_path, str) and os.path.exists(page_img_path):
        img = Image.open(page_img_path)
        width, height = img.size
    elif isinstance(page_img_path, tuple):
        width, height = page_img_path

    sections_data = []

    for idx, table in enumerate(page.tables):
        table_bbox = geometry.quad_trp_to_quad_std(
            table.geometry.polygon, width, height
        )
        cell_sections_data = get_trt_table_cell_sections(
            table,
            width,
            height,
            associate_words=associate_words,
            associate_word_ids=associate_word_ids,
        )

        # row count will include header row if present
        n_rows, n_cols = get_trt_table_num_rows_cols(table)

        # this returns a list of list
        # VERIFY: why using the first one?
        table_header_field_names = table.get_header_field_names()
        if len(table_header_field_names):
            table_header_field_names = table_header_field_names[0]

        table_df = get_table_as_df(table)

        # if table has headers, reassign df with headers as the first row
        if table_header_field_names:
            table_headers = table_df.iloc[0]
            # deduplicate table headers
            table_headers = utils.deduplicate_list_by_append(table_headers)
            table_df = table_df[1:]
            table_df.columns = table_headers

        sections_data.append(
            {
                "quad": table_bbox,
                "text": "table",
                "num_rows": n_rows,
                "num_cols": n_cols,
                "header_field_names": table_header_field_names,
                "cell_sections": cell_sections_data,
                "table_idx": idx,
                "section_type": "table",
                "confidence": table.confidence / 100,
                "df": table_df,
                "table_id": table.id,
            }
        )
    return sections_data


def get_trt_table_cell_sections(
    trt_table,
    width,
    height,
    merge_cells=True,
    associate_word_ids=False,
    associate_words=None,
):
    """Get cell sections data from Textract Respone Parser table

    Parameters
    ----------
    trt_table : trp.Table
        Textract response parser Table
    width : int
        image width
    height : int
        image height
    merge_cells: bool
        merge cells to get collective text
    words: list of ocr word sections
        if provided, will be used to extract OCR word IDs inside each cell section
    """
    cell_sections_data = []

    if merge_cells:
        merged_cell_text_map = get_merged_cells_text_map(trt_table)
    else:
        merged_cell_text_map = {}

    for ridx, row in enumerate(trt_table.rows):
        for cidx, cell in enumerate(row.cells):
            # print("Table[{}][{}] = {} --- {}".format(r, c, cell.text, cell.confidence))
            cell_bbox = geometry.quad_trp_to_quad_std(
                cell.geometry.polygon, width, height
            )
            # cell_text = cell.text
            cell_text = (
                merged_cell_text_map[cell.id]
                if cell.id in merged_cell_text_map
                else cell.mergedText
            )
            this_section = {
                "quad": cell_bbox,
                "text": cell_text,
                "section_type": "table-cell",
                "confidence": cell.confidence / 100,
                "row_idx": ridx,
                "col_idx": cidx,
                "cell_type": cell.entityTypes,
            }

            if associate_word_ids and (associate_words is not None):
                # REVIEW: This behavior is a bit weird if merge_cells is True
                # because the cell text contains more than what is inside the cell bounding box
                # live_ocr will only get the words inside the given section
                words_inside = live_ocr.get_words_inside_section(
                    this_section, associate_words, filter_text=cell_text
                )
                this_section["word_ids"] = [w["id"] for w in words_inside]

            cell_sections_data.append(this_section)
    return cell_sections_data


def get_trt_table_num_rows_cols(trt_table):
    """Get number of rows and columns from TRP table

    Parameters
    ----------
    trt_table : trp.Table
        Textract Response Parser Table

    """
    n_rows = len(trt_table.rows)
    # n_cols = 0
    n_cols_options = []
    for row in trt_table.rows:
        this_n_col = len(row.cells)
        n_cols_options.append(this_n_col)
    n_cols = max(n_cols_options)

    return n_rows, n_cols


def get_table_as_df(table, merge_cells=True):
    """Get TRT table as pandas DataFrame"""
    if merge_cells:
        merged_cell_text_map = get_merged_cells_text_map(table)
    else:
        merged_cell_text_map = {}

    rows = serialize_rows_to_text(table.rows, merged_cell_text_map=merged_cell_text_map)
    import pandas as pd

    df = pd.DataFrame(rows)
    return df


def serialize_rows_to_text(rows, merged_cell_text_map={}):
    rows_text = []
    for row in rows:
        # rows_text.append([ii.mergedText for ii in row.cells])
        rows_text.append(
            [
                merged_cell_text_map[ii.id]
                if ii.id in merged_cell_text_map
                else ii.mergedText
                for ii in row.cells
            ]
        )
    return rows_text


def get_merged_cells_text_map(table):
    """Get mappping of merged cells to their collective text"""
    merged_cell_text_map = {}

    merged_groups = []
    merged_groups_flat = []

    for idx, mcell in enumerate(table.merged_cells):
        this_group = {"Ids": [], "merged_text": None}
        relationships = mcell.block.get("Relationships", [])
        child_reln = [ii for ii in relationships if ii["Type"] == "CHILD"]
        # choose first?
        child_ids = child_reln[0]["Ids"]
        this_group["Ids"].extend(child_ids)
        merged_groups_flat.extend(child_ids)
        merged_groups.append(this_group)

    # get cell text
    cell_text_map = {}

    for row in table.rows:
        for cell in row.cells:
            if cell.id in merged_groups_flat:
                cell_text_map[cell.id] = cell.text

    # get merged text
    for idx, group in enumerate(merged_groups):
        this_merged_text = ""
        for id_ in group["Ids"]:
            this_merged_text += cell_text_map[id_]
        merged_groups[idx]["merged_text"] = this_merged_text.strip()

    # get final cell to merged text map
    for mgrp in merged_groups:
        for id_ in mgrp["Ids"]:
            merged_cell_text_map[id_] = mgrp["merged_text"]

    return merged_cell_text_map


### BRING BACK LATER IF NEEDED
# def is_key_value_table(table_section_data):
#     """Check if table section can be extracted as table of key-values

#     Conditions for parsing a table as key-values
#         - ?? table has no header - REMOVED
#         - table has confidence >= 95%
#         - table as 2 cols

#     Parameters
#     ----------
#     table_section_data : dict
#         Table sections data (standard format)

#     """
#     # group 1
#     condition1 = not table_section_data["header_field_names"]
#     condition2 = table_section_data["confidence"] >= 0.8
#     condition3 = table_section_data["num_cols"] == 2

#     group1_op = all([condition2, condition3])

#     # group 2
#     condition1 = not table_section_data["header_field_names"]
#     condition2 = table_section_data["confidence"] >= 0.8
#     condition3 = table_section_data["num_cols"] == 4

#     group2_op = all([condition2, condition3])

#     return group1_op or group2_op


# def get_key_value_sections_from_table(table_section_data, merge_rows=False):
#     """Get key-values sections from Table

#     Parameters
#     ----------
#     table_section_data : dict
#         Table section data (standard format)

#     """
#     if not is_key_value_table(table_section_data):
#         return None

#     sections_data = []

#     cell_index_map = np.zeros(
#         (table_section_data["num_rows"], table_section_data["num_cols"]), dtype=int
#     )

#     for idx, section in enumerate(table_section_data["cell_sections"]):
#         cell_index_map[section["row_idx"]][section["col_idx"]] = idx

#     for idx, row in enumerate(cell_index_map):
#         k_section_idx = row[0]
#         v_section_idx = row[1]

#         k_section = table_section_data["cell_sections"][k_section_idx]
#         v_section = table_section_data["cell_sections"][v_section_idx]

#         merge_this_row_up = False
#         if merge_rows and (not k_section["text"]) and len(sections_data):
#             # Merge up, if:
#             # parameter is passed to merge
#             # section text is empty
#             # and there already exists something upstream to merge with
#             merge_this_row_up = True

#         if merge_this_row_up:
#             # pop and modify the section to include
#             this_section = sections_data.pop()
#             this_section["quad"] = geometry.merge_quads(
#                 this_section["quad"], k_section["quad"], v_section["quad"]
#             )
#             this_section["value"] = this_section["value"] + "\n" + v_section["text"]
#             this_section["text"] = this_section["key"] + " - " + this_section["value"]
#             this_section["key_bbox"] = geometry.merge_quads(
#                 this_section["key_bbox"], k_section["quad"]
#             )
#             this_section["value_bbox"] = geometry.merge_quads(
#                 this_section["value_bbox"], v_section["quad"]
#             )
#         else:
#             # merge key-value
#             s_bbox = geometry.merge_quads(k_section["quad"], v_section["quad"])
#             s_text = k_section["text"] + " - " + v_section["text"]

#             this_section = {
#                 "quad": s_bbox,
#                 "text": s_text,
#                 "section_type": "key-value",
#                 "key": k_section["text"],
#                 "value": v_section["text"],
#                 "key_bbox": k_section["quad"],
#                 "value_bbox": v_section["quad"],
#                 "confidence": np.mean(
#                     [
#                         k_section["confidence"],
#                         v_section["confidence"],
#                     ]
#                 ),
#             }
#         sections_data.append(this_section)

#     return sections_data


# def split_4col_table_into_2_key_value_tables(table_section):
#     cell_sections = table_section["cell_sections"]
#     table1_cell_sections = []
#     table2_cell_sections = []

#     for cell in cell_sections:
#         if cell["col_idx"] in [0, 1]:
#             table1_cell_sections.append(cell)
#         else:
#             # modify cell index
#             mod_cell = cell.copy()
#             mod_cell["col_idx"] -= 2
#             table2_cell_sections.append(mod_cell)

#     table1_sections_data = table_section.copy()
#     table1_sections_data["num_cols"] = int(table1_sections_data["num_cols"] / 2)
#     table1_sections_data["cell_sections"] = table1_cell_sections
#     table1_sections_data["quad"] = geometry.merge_quads(
#         *[ii["quad"] for ii in table1_cell_sections]
#     )

#     table2_sections_data = table_section.copy()
#     table2_sections_data["num_cols"] = int(table2_sections_data["num_cols"] / 2)
#     table2_sections_data["cell_sections"] = table2_cell_sections

#     table2_sections_data["quad"] = geometry.merge_quads(
#         *[ii["quad"] for ii in table2_cell_sections]
#     )

#     return [table1_sections_data, table2_sections_data]


# def get_key_value_sections_from_all_tables(table_sections_data, merge_rows=False):
#     """Get key-values sections from list of table sections

#     Parameters
#     ----------
#     table_sections_data : dict
#         List of table sections (standard format)

#     """
#     all_kv_sections_data = []

#     for table_section in table_sections_data:
#         if table_section["section_type"] != "table":
#             continue
#         this_kv_sections_data = get_key_value_sections_from_table(
#             table_section, merge_rows=merge_rows
#         )
#         all_kv_sections_data.extend(this_kv_sections_data)
#     return all_kv_sections_data
