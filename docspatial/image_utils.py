import cv2
import numpy as np
from PIL import Image


def resize_image_keep_aspect(
    img,
    shape=None,
    max_dim=None,
    pad_color=(0, 0, 0),
    return_info=False,
    interpolation=None,
    return_pil=False,
):
    """
    Resize image keeping aspect ratio
    img, pad_color to have same bgr / rgb
    """
    if isinstance(img, str):
        img = cv2.imread(img)
    elif isinstance(img, Image.Image):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if len(img.shape) == 3:
        H, W, ch = img.shape
    elif len(img.shape) == 2:
        ch = None
        H, W = img.shape

    is_resize = True
    if shape is not None:
        final_H, final_W = shape
    elif max_dim is not None:
        if max(H, W) > max_dim:  # if larger than max_dim
            if H > W:
                final_H = max_dim
                final_W = int(W * (max_dim / H))
            else:
                final_W = max_dim
                final_H = int(H * (max_dim / W))
        else:  # if already smaller than max_dim
            final_H, final_W = H, W
            is_resize = False
    else:
        final_H, final_W = H, W
        is_resize = False

    if not is_resize:
        if return_pil:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if return_info:
            info = {
                "h_ratio": 1,
                "w_ratio": 1,
                "min_ratio": 1,
                "x1": 0,
                "y1": 0,
                "x2": W,
                "y2": H,
            }
            return img, info
        else:
            return img

    h_ratio = final_H / float(H)
    w_ratio = final_W / float(W)

    min_ratio = min(h_ratio, w_ratio)

    intrp = interpolation if interpolation is not None else cv2.INTER_AREA
    resized_img = cv2.resize(
        img, (0, 0), fx=min_ratio, fy=min_ratio, interpolation=intrp
    )
    if len(img.shape) == 3:
        resized_img_H, resized_img_W, _ = resized_img.shape
    elif len(img.shape) == 2:
        resized_img_H, resized_img_W = resized_img.shape

    if ch is None:
        final_image = np.ones((final_H, final_W), dtype=np.uint8)
        # final_image = pad_color * final_image
    else:
        final_image = np.ones((final_H, final_W, ch), dtype=np.uint8)
        final_image = pad_color * final_image
    final_image = final_image.astype(np.uint8)

    start_xidx = int((final_W - resized_img_W) / 2)
    start_yidx = int((final_H - resized_img_H) / 2)
    final_image[
        start_yidx : start_yidx + resized_img_H, start_xidx : start_xidx + resized_img_W
    ] = resized_img

    # create mask with added background
    mask = np.ones((final_H, final_W), dtype=np.uint8) * 255
    mask[
        start_yidx : start_yidx + resized_img_H, start_xidx : start_xidx + resized_img_W
    ] = 0

    if return_pil:
        final_image = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))

    if return_info:
        info = {
            "h_ratio": h_ratio,
            "w_ratio": w_ratio,
            "min_ratio": min_ratio,
            "x1": start_xidx,
            "y1": start_yidx,
            "x2": start_xidx + resized_img_W,
            "y2": start_yidx + resized_img_H,
        }
        return final_image, info
    else:
        return final_image
