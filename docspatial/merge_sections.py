import shapely

from . import geometry


def filter_non_intersecting_sections(sections, sections_to_check):
    """Filter sections that do not intersect with sections_to_check

    Parameters
    ----------
    sections : list of sections
        sections that need to be filtered
    sections_to_check : list of sections
        sections that need to be kept, to use for filtering
    """

    polygons_to_check = []
    for sec in sections_to_check:
        sec_vertices = geometry.quad_std_to_quad_points_list(sec["quad"])
        sec_polygon = shapely.Polygon(sec_vertices)
        polygons_to_check.append(sec_polygon)

    sections_filtered = []
    for sec in sections:
        sec_vertices = geometry.quad_std_to_quad_points_list(sec["quad"])
        sec_polygon = shapely.Polygon(sec_vertices)

        intersects = False
        for this_polygon in polygons_to_check:
            if sec_polygon.intersects(this_polygon):
                intersects = True
                break

        if not intersects:
            sections_filtered.append(sec)

    return sections_filtered


def merge_ocr_trt_sections(ocr_sections, kv_sections, table_sections=None):
    """Merge OCR, Key-Value and Table sections together and filter out non-intersecting sections

    - Filter high confidence tables
    - 2 column table with no headers is not a table
    - Filter key-values not within any tables (often, textract returns key-values inside tables)
    - Filter OCR sections not within any tables or key-values

    Parameters
    ----------
    ocr_sections : list of sections
        ocr detected sections
    kv_sections : list of sections
        key-value sections
    table_sections : list of sections
        table sections
    """

    # filter tables
    table_sections_filtered = []

    if table_sections is not None:
        for table_section in table_sections:
            # if confidence less than 90%, not a table
            if table_section["confidence"] < 0.9:
                continue
            # if 2 columns and no header field names, not a table
            # it is likely a neatly formatted list of key-values
            if (table_section["num_cols"] == 2) and (
                not table_section["header_field_names"]
            ):
                continue
            table_sections_filtered.append(table_section)

    # filter key-values from tables
    kv_sections_filtered = filter_non_intersecting_sections(
        kv_sections, table_sections_filtered
    )

    # filter OCR paragraphs from tables and key-values
    # merge key and value bbox together for polygon checks
    merged_kv_sections = []
    for kvsec in kv_sections_filtered:
        merged_quad = geometry.merge_quads(kvsec["key_quad"], kvsec["value_quad"])
        merged_kv_sections.append({"quad": merged_quad})

    ocr_sections_filtered = filter_non_intersecting_sections(
        ocr_sections, table_sections_filtered + merged_kv_sections
    )

    if table_sections is None:
        return ocr_sections_filtered, kv_sections_filtered

    return ocr_sections_filtered, kv_sections_filtered, table_sections_filtered
