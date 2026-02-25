import numpy as np
import PIL
from models import geometry
from models.image_utils import resize_image_keep_aspect
from PIL import Image, ImageDraw, ImageFont

from meta.utils import get_chunks


def visualize_images_in_grid(
    images,
    image_size=(1024, 768),
    images_per_row="auto",
    return_pil=True,
    add_separator=True,
):
    if isinstance(images_per_row, str) and images_per_row.lower() == "auto":
        images_per_row = min(len(images), 4)

    if not len(images):
        return None

    chunked_images = get_chunks(images, images_per_row, pad_value=None)

    all_row_images = []
    for row_images in chunked_images:
        row_images_ = []
        for iidx, img in enumerate(row_images):
            if isinstance(img, PIL.Image.Image):
                img = np.array(img)

            if img is None:
                img = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
            else:
                # if grayscale
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, axis=-1)
                img = resize_image_keep_aspect(img, image_size)
            row_images_.append(img)
            if add_separator and (iidx < len(row_images) - 1):
                row_images_.append(np.zeros((image_size[0], 10, 3), dtype=np.uint8))
        row_images_ = np.hstack(row_images_)
        all_row_images.append(row_images_)

    grid_image = np.vstack(all_row_images)

    if return_pil:
        grid_image = Image.fromarray(grid_image)
    return grid_image


def get_pil_colors(return_hex=False):
    available_colors = [
        c for c in PIL.ImageColor.colormap.keys() if "white" not in c.lower()
    ]
    remove_colors = ["mintcream", "snow"]
    available_colors = [ii for ii in available_colors if ii not in remove_colors]

    if return_hex:
        available_colors = {c: PIL.ImageColor.colormap[c] for c in available_colors}
    return available_colors


def draw_field_on_image(
    image,
    bbox,
    field_name=None,
    bbox_color="blue",
    bbox_width=2,
    text_color="red",
    text_inside=False,
    font_size=15,
):
    """
    Issues:

    Font size and placement is constant here.
    It should be inferred dynamically based on bbox/image size
    Text direction is not figured out yet. It assumes image is not rotated.
    """

    all_x = [ii["x"] for ii in bbox]
    all_y = [ii["y"] for ii in bbox]

    v1 = (min(all_x), min(all_y))
    v3 = (max(all_x), max(all_y))

    # draw as polygon
    quad_points = geometry.quad_std_to_quad_points_list(bbox)
    quad_points = [tuple(ii) for ii in quad_points]

    draw = ImageDraw.Draw(image)

    # draw.rectangle((v1, v3), None, bbox_color, width=2)
    draw.polygon(quad_points, None, bbox_color, width=bbox_width)

    if field_name:
        if text_inside:
            v_text = (v1[0], v1[1] + 20)
        else:
            v_text = (v1[0], v1[1] - 20)
        fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", font_size)
        draw.text(v_text, field_name, font=fnt, fill=text_color)


def visualize_sections_data(
    sections_data,
    img_path,
    draw_text=None,
    text_key=None,
    bbox_color="blue",
    bbox_width=2,
    text_color="red",
    text_inside=False,
    font_size=15,
):
    """Visualize list of sections data on image

    Parameters
    ----------
    sections_data : list
        list of sections data (as shown below in example usage)
    img_path : str or PIL.Image
        image local file path or PIL image object
    draw_text : bool
        if True, also draw text on image
    bbox_color : str
        color for bounding box around sections (PIL.ImageColor.colormap)
    text_color : str
        color for the drawn text (PIL.ImageColor.colormap)
    text_inside : bool
        if True, draw text inside the bounding box

    Example:
    --------
    pil_img_path = "/some/path/to/image.jpg"
    # or pil_img_path = Image.open("/some/path/to/image.jpg")
    sections_data = [
        {
            "text": "machine",
            "quad": [
                {"x": 259, "y": 832},
                {"x": 333, "y": 831},
                {"x": 333, "y": 843},
                {"x": 259, "y": 844},
            ],
        },
        {
            "text": "shop",
            "quad": [
                {"x": 340, "y": 831},
                {"x": 384, "y": 831},
                {"x": 384, "y": 843},
                {"x": 340, "y": 843},
            ],
        }
    ]
    visualize_pil_img = visualize_sections_data(sections_data, pil_img_path)
    """
    image = get_image(img_path)

    if text_key is None:
        text_key = "text"

    for section in sections_data:
        if not section['quad']:
            continue
        field_name = None
        if draw_text:
            field_name = section.get(text_key, "")
        draw_field_on_image(
            image,
            section["quad"],
            field_name=field_name,
            bbox_color=bbox_color,
            bbox_width=bbox_width,
            text_color=text_color,
            text_inside=text_inside,
            font_size=font_size,
        )
    return image


def visualize_points(points, img_path, color="red", radius=3):
    """Visualize list of points on image

    Parameters
    ----------
    points : list
        list of points (as shown below in example usage)
    img_path : str or PIL.Image
        image local file path or PIL image object
    color : str
        color for the drawn points (PIL.ImageColor.colormap)

    Example:
    --------
    pil_img_path = "/some/path/to/image.jpg"
    # or pil_img_path = Image.open("/some/path/to/image.jpg")
    points = [
        [20, 40], [30, 50], [40, 60]
    ]
    visualize_pil_img = visualize_points(points, pil_img_path)
    """
    image = get_image(img_path)

    draw = ImageDraw.Draw(image)
    for point in points:
        x, y = point
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius), fill=color, outline=color
        )
    return image


def get_image(img_path):
    if isinstance(img_path, str):
        image = Image.open(img_path).convert("RGB")
    else:
        image = img_path.copy()
    return image
