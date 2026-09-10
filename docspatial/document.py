# document.py

"""Page and multi-page document objects over word-level OCR output.

A thin object layer over the functional modules, for when you are working with
a whole page or document rather than calling the spatial functions directly.
It carries the sequence that normally has to be got right by hand: measure the
page's skew, rotate words and image together, crop to the document bounds while
translating every section with the crop, normalize, then query.

Bring your own words. The original of this class loaded them from a particular
OCR provider and job layout; here the page takes words in the standard format
and the engine is somebody else's problem (see docspatial.adapters).
"""
from collections import defaultdict

import numpy as np
from PIL import Image

from . import geometry, phrase_search, sections as section_detection, viz


class DocumentPage:
    def __init__(
        self,
        words,
        image=None,
        name=None,
        page_num=None,
        lines=None,
        paragraphs=None,
        blocks=None,
        auto_deskew=False,
        bounded=False,
        normalize_sections=False,
    ):
        self.name = name
        self.page_num = page_num

        self.words = geometry.prepare_words(words)
        self.lines = lines if lines is not None else []
        self.paragraphs = paragraphs if paragraphs is not None else []
        self.blocks = blocks if blocks is not None else []

        if not self.lines and self.words:
            self.lines = section_detection.extract_lines_from_ocr(self.words)

        self.load_image_data(image)

        # keep the untouched originals — rotation and bounding always operate
        # from these, so the operations do not compound
        self._original_words = list(self.words)
        self._original_lines = list(self.lines)
        self._original_paragraphs = list(self.paragraphs)
        self._original_blocks = list(self.blocks)

        self.page_already_rotated = False
        self.rotation_angle = self.find_document_rotation(
            nearest_angle=1, check_in_original=True
        )
        if auto_deskew:
            self.rotate_page()

        self.page_already_bounded = False
        self.document_bbox = self.get_word_based_document_bounding_box()
        if bounded:
            self.bound_page()

        if normalize_sections:
            self.normalize_all_sections()

    def __str__(self):
        return f"DocumentPage `{self.name}` with {len(self.words)} words."

    def __repr__(self):
        return self.__str__()

    @property
    def _section_objects(self):
        return {
            "words": self.words,
            "lines": self.lines,
            "paragraphs": self.paragraphs,
            "blocks": self.blocks,
        }

    def load_image_data(self, image=None):
        """Load image data."""
        self.image = self.image_width = self.image_height = None
        if image is not None:
            if isinstance(image, str):
                image = Image.open(image)
            self.image = image
            self.image_width, self.image_height = self.image.size
        else:
            # no image: fall back to the extent of the words, which is all the
            # rotation and normalization maths actually needs
            bounds = self.get_word_based_document_bounding_box()
            if bounds:
                self.image_width, self.image_height = bounds[2], bounds[3]

        self._original_image = self.image
        self._original_image_width = self.image_width
        self._original_image_height = self.image_height

    def normalize_all_sections(self):
        for section_type, sections in self._section_objects.items():
            geometry.normalize_coordinates_in_sections(
                sections, self.image_width, self.image_height
            )

    def get_word_based_document_bounding_box(self):
        """Get document bounds based on word bounding boxes."""
        if not self.words:
            return None
        all_bboxes = [geometry.quad_std_to_rect_std(ii["quad"]) for ii in self.words]

        if not len(all_bboxes):
            return []

        left_most = min([ii[0] for ii in all_bboxes])
        top_most = min([ii[1] for ii in all_bboxes])
        right_most = max([ii[2] for ii in all_bboxes])
        bottom_most = max([ii[3] for ii in all_bboxes])

        bounds_rect = [left_most, top_most, right_most, bottom_most]
        return bounds_rect

    def find_document_rotation(self, nearest_angle=1, check_in_original=False):
        """Find document rotation to the nearest angle

        Calculate the angle of all words and return the most common angle.

        Vertices are expected in clockwise order starting from the top-left,
        so calculating the slope between the first 2 points gives the angle.

        This can calculate current rotation angle or original rotation angle.

        NOTE ON SIGN — the returned angle is what you rotate BY to deskew, not
        its negative. See geometry.calculate_rotation_angle.
        """
        if not self.words:
            return None

        if check_in_original:
            words = self._original_words
        else:
            words = self.words

        # method 1
        # calculate angle of every word and take the most common
        # all_angles = []
        # for idx, word in enumerate(words):
        #     word_rot_angle = word.get("rotation_angle", 0)
        #     if word_rot_angle is None:
        #         word_rot_angle = geometry.calculate_rotation_angle(
        #             word["quad"][0], word["quad"][1]
        #         )
        #     all_angles.append(word_rot_angle)

        # if nearest_angle:
        #     all_angles = [
        #         utils.round_to_nearest_multiple(ii, nearest_angle) for ii in all_angles
        #     ]

        # rot_angle = Counter(all_angles).most_common()[0][0]

        # method 2
        # calculate the angle of the top 10 longest words and take the median
        # longest words based on geometric length
        longest_words = sorted(
            words, key=lambda x: x["rect"][2] - x["rect"][0], reverse=True
        )[:10]
        rot_angle = np.median([w["rotation_angle"] for w in longest_words])
        return rot_angle

    def _rotate_page_image(self, image, rotation_angle):
        """Rotate page image by the rotation angle."""
        return geometry.rotate_image(image, rotation_angle)

    def _rotate_sections(
        self, sections, rotation_angle, image_size, other_quad_keys_to_rotate=[]
    ):
        """Rotate OCR sections by the rotation angle."""
        return geometry.rotate_sections_on_image(
            sections,
            rotation_angle,
            image_size,
            other_quad_keys_to_rotate=other_quad_keys_to_rotate,
        )

    def rotate_page(self, nearest_angle=3, force_run=False):
        """Rotate page image and OCR data by the rotation angle.

        Operate on _original data always.
        """
        if not self.words:
            print("No word data found.")
            return

        if (not force_run) and self.page_already_rotated:
            print("Page already rotated.")
            return

        # rotate page will operate with _original data
        self.rotation_angle = self.find_document_rotation(
            nearest_angle=nearest_angle, check_in_original=True
        )

        if self.rotation_angle != 0:
            image_size = (self._original_image_width, self._original_image_height)
            if self._original_image is not None:
                self.image = self._rotate_page_image(
                    self._original_image, self.rotation_angle
                )
                self.image_width, self.image_height = self.image.size

            self.words = self._rotate_sections(
                self._original_words, self.rotation_angle, image_size
            )
            self.lines = self._rotate_sections(
                self._original_lines, self.rotation_angle, image_size
            )
            self.blocks = self._rotate_sections(
                self._original_blocks, self.rotation_angle, image_size
            )
            self.paragraphs = self._rotate_sections(
                self._original_paragraphs, self.rotation_angle, image_size
            )

        self.document_bbox = self.get_word_based_document_bounding_box()
        self.page_already_rotated = True

    def bound_page(self):
        """Bound page image and OCR data by the document bounds.

        When there are multiple crops in an Image, and we want to bound the page
        then certain words, paragraphs, sections will be outside the bounds.
        And will have to be filtered out.
        """
        if not self.words:
            print("No word data found.")
            return

        if self.page_already_bounded:
            print("Page already bounded.")
            return

        if not self.document_bbox:
            print("No document bounds found.")
            return

        if self.image is not None:
            self.image = self.image.crop(self.document_bbox)
            self.image_width, self.image_height = self.image.size
        else:
            left, top, right, bottom = self.document_bbox
            self.image_width, self.image_height = right - left, bottom - top

        # Treat Cropping as Translation
        # first point coordinates of document bounds will be the translation values
        first_x, first_y = self.document_bbox[:2]
        M = np.array([[1, 0, -first_x], [0, 1, -first_y]])

        self.words = geometry.affine_transform_sections(self.words, M)
        self.lines = geometry.affine_transform_sections(self.lines, M)
        self.blocks = geometry.affine_transform_sections(self.blocks, M)
        self.paragraphs = geometry.affine_transform_sections(self.paragraphs, M)

        self.normalize_all_sections()
        self.page_already_bounded = True

    def find_phrase(
        self,
        phrase,
        read_as_document=True,
        same_line=False,
        case_insensitive=True,
        debug=False,
    ):
        """Find phrase in OCR text."""
        if not self.words:
            print("No word data found.")
            return None

        # ignore_punctuation is not an option
        # it will always be False
        # if this has to be an option, then it will have to be managed
        # from the begginning of this class, by getting OCR words without punctuation
        # along with rotations and bounding
        all_phrase_data = phrase_search.find_phrase(
            phrase,
            self.words,
            read_as_document=read_as_document,
            same_line=same_line,
            case_insensitive=case_insensitive,
            ignore_punctuation=False,
            distance_tolerance=np.inf,
            return_all=True,
            debug=debug,
        )

        if not all_phrase_data:
            return all_phrase_data

        for pdata in all_phrase_data:
            geometry.normalize_coordinates_in_section(
                pdata, self.image_width, self.image_height
            )

        return all_phrase_data

    def find_all_phrases(
        self,
        phrases,
        read_as_document=True,
        same_line=False,
        case_insensitive=True,
        debug=False,
    ):
        all_phrases_data = {}
        for phrase in phrases:
            this_phrase_data = self.find_phrase(
                phrase,
                read_as_document=read_as_document,
                same_line=same_line,
                case_insensitive=case_insensitive,
                debug=debug,
            )
            all_phrases_data[phrase] = this_phrase_data
        return all_phrases_data

    def visualize_sections(
        self,
        sections,
        draw_text=False,
        text_key=None,
        text_inside=False,
        bbox_color="blue",
        bbox_width=2,
        font_size=15,
    ):
        """Visualize sections.

        Passed sections can be a list of sections or a section type.
        """
        section_type_options = self._section_objects.keys()

        if isinstance(sections, str):
            section_type = sections
            sections = self._section_objects.get(section_type)
            if sections is None:
                print(f"No {section_type} found.")
                print(f"Available section types: {section_type_options}")
                return None
        elif isinstance(sections, list):
            pass
        else:
            print("sections have to be a section_type or a list of sections.")
            return None

        return viz.visualize_sections_data(
            sections,
            self.image,
            draw_text=draw_text,
            text_key=text_key,
            text_inside=text_inside,
            bbox_color=bbox_color,
            bbox_width=bbox_width,
            font_size=font_size,
        )


class Document:
    def __init__(self, pages, name=None, merge_method=None):
        """A document is an ordered list of DocumentPage objects."""
        self.pages = pages
        self.name = name
        self.merge_method = merge_method

        # supported merge methods
        supported_merge_methods = ["zero_to_n", "zero_to_1"]
        if (self.merge_method is not None) and (
            self.merge_method not in supported_merge_methods
        ):
            print("WARNING: Unsupported merge method. No merging beyond flattening.")
            self.merge_method = None

        self._initialize_pages()

    def __str__(self):
        return f"Document `{self.name}` with {self.num_pages} pages."

    def __repr__(self):
        return self.__str__()

    def _initialize_pages(self):
        for idx, page in enumerate(self.pages):
            if page.page_num is None:
                page.page_num = idx + 1

        self.num_pages = len(self.pages)
        self.page_nums = [p.page_num for p in self.pages]
        self.page_num_to_index = {p.page_num: i for i, p in enumerate(self.pages)}
        self.page_index_to_num = {i: p.page_num for i, p in enumerate(self.pages)}

    def _validate_page_nums(self, page_nums):
        for page_num in page_nums:
            if page_num not in self.page_num_to_index:
                raise ValueError(f"Page number {page_num} not in document.")

    def find_phrase(
        self,
        phrase,
        read_as_document=True,
        same_line=False,
        case_insensitive=True,
        page_nums=[],
        debug=False,
    ):
        self._validate_page_nums(page_nums)

        all_page_phrase_data = {}

        if not page_nums:
            page_nums = self.page_nums
        for page_num in page_nums:
            page_idx = self.page_num_to_index[page_num]
            this_page_found_phrases = self.pages[page_idx].find_phrase(
                phrase,
                read_as_document=read_as_document,
                same_line=same_line,
                case_insensitive=case_insensitive,
                debug=debug,
            )
            all_page_phrase_data[page_num] = this_page_found_phrases
        return all_page_phrase_data

    def find_all_phrases(
        self,
        phrases,
        read_as_document=True,
        same_line=False,
        case_insensitive=True,
        page_nums=[],
        debug=False,
    ):
        self._validate_page_nums(page_nums)

        all_phrases_all_pages_data = {k: {} for k in phrases}

        if not page_nums:
            page_nums = self.page_nums
        for page_num in page_nums:
            page_idx = self.page_num_to_index[page_num]
            this_page_all_phrases_data = self.pages[page_idx].find_all_phrases(
                phrases,
                read_as_document=read_as_document,
                same_line=same_line,
                case_insensitive=case_insensitive,
                debug=debug,
            )

            for this_phrase, this_phrase_data in this_page_all_phrases_data.items():
                all_phrases_all_pages_data[this_phrase][page_num] = this_phrase_data

        return all_phrases_all_pages_data

    def merge_sections_across_pages(self, sections_dict, merge_method=None):
        """Merge sections across pages.

        sections_dict: dict
            A dictionary with page numbers as keys and sections as values.
            Example:
                {
                    '1': [section1, section2, ...],
                    '2': [section1, section2, ...],
                    ...
                }
        """
        page_nums = sections_dict.keys()
        self._validate_page_nums(page_nums)

        if merge_method is None:
            merge_method = self.merge_method

        if merge_method == "zero_to_n":
            zero_to_n = True
        else:
            zero_to_n = False

        flat_sections = []
        for page_num, page_sections in sections_dict.items():
            if self.merge_method is not None:
                page_sections_offset = geometry.offset_sections_norm_by_page_num(
                    page_sections,
                    page_num,
                    zero_to_n=zero_to_n,
                    num_pages=self.num_pages,
                )
                flat_sections.extend(page_sections_offset)
            else:
                # add page_num to sections
                for s in page_sections:
                    s["page_num"] = page_num
                flat_sections.extend(page_sections)

        return flat_sections

    def get_section_from_merged_pages_xywh(self, x, y, w, h, merge_method="zero_to_n"):
        """Get section from merged page coordinates"""

        if merge_method != "zero_to_n":
            raise NotImplementedError(
                "Only `zero_to_n` merge method is supported for now."
            )
        page_num = int(y)
        y -= page_num
        page = self.pages[page_num]
        section = geometry.get_denormalized_section_from_xywh(
            x, y, w, h, (page.image_width, page.image_height)
        )
        section["page_num"] = self.page_index_to_num[page_num]
        return section

    def visualize_sections(
        self,
        sections_dict={},
        draw_text=False,
        text_key=None,
        text_inside=False,
        bbox_color="blue",
        bbox_width=2,
        font_size=15,
    ):
        viz_pages = [p.image for p in self.pages]

        if isinstance(sections_dict, str):
            section_type = sections_dict
            sections_dict = {
                page_num: getattr(
                    self.pages[self.page_num_to_index[page_num]], section_type
                )
                for page_num in self.page_nums
            }
        elif isinstance(sections_dict, list):
            # list of sections with page_num inside
            sections_list = sections_dict
            sections_dict = defaultdict(list)
            for section in sections_list:
                sections_dict[section["page_num"]].append(section)

        if sections_dict:
            for page_num, page_sections in sections_dict.items():
                page_idx = self.page_num_to_index[page_num]
                viz_pages[page_idx] = self.pages[page_idx].visualize_sections(
                    page_sections,
                    draw_text=draw_text,
                    text_key=text_key,
                    text_inside=text_inside,
                    bbox_color=bbox_color,
                    bbox_width=bbox_width,
                    font_size=font_size,
                )
        grid_viz = viz.visualize_images_in_grid(viz_pages)
        return grid_viz
