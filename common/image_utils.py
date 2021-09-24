import numpy as np
from typing import Iterable, Optional, Tuple
import typeguard
import torch
from PIL import Image, ImageCms
from logging import getLogger
# 3rd party:
import io
import os
import matplotlib.pyplot as plt
from .data_classes import TensorBatch


def im_concatenate(images: Iterable[np.ndarray], axis=1, pad_value: float = 1.0) -> np.ndarray:
    """
    im_concatenate images in a row such that are centered aligned in the axis provided
    :param images: list of images np.ndarray WxHxC
    :param axis: if 0 then result is column if 1 then results a row
    :param pad_value: padding value [0..1]
    :return: np.ndarray padded with pad of all images
    """
    assert axis in [0, 1], 'axis should be 0 or 1'
    assert 0. <= pad_value <= 1., 'pad_value should be in [0,1]'
    typeguard.check_argument_types()
    pivot = abs(1 - axis)
    n_ch = images[0].shape[-1]

    total_axis = sum([i.shape[axis] for i in images])
    max_pivot = max([i.shape[pivot] for i in images])
    center = max_pivot // 2

    shape = np.zeros(3, dtype=int)
    shape[axis] = total_axis
    shape[pivot] = max_pivot
    shape[2] = n_ch

    canvas = np.ones(shape, dtype=np.float32) * pad_value

    accumulator_axis = 0
    for img in images:
        cur_axis = img.shape[axis]
        cur_pivot = img.shape[pivot]
        min_pivot = center - cur_pivot // 2

        idx = np.zeros((2, 2), dtype=int)
        idx[axis, :] = [accumulator_axis, accumulator_axis + cur_axis]
        idx[pivot, :] = [min_pivot, min_pivot + cur_pivot]

        canvas[idx[0, 0]:idx[0, 1], idx[1, 0]: idx[1, 1], :] = img
        accumulator_axis += cur_axis

    return canvas


def square_grid_im_concat(images: Iterable[np.ndarray], pad_value: float = 1.0) -> np.ndarray:
    n_imgs = len(images)
    n_sq = int(np.floor(np.sqrt(n_imgs)))
    n_full = n_sq ** 2
    if n_full > n_imgs:
        images_list = []
        for ind in range(n_full):
            if ind < n_imgs:
                images_list.append(images[ind])
            else:
                images_list.append(np.ones_like(images[0]) * pad_value)
    else:
        images_list = images
    img_grid = [im_concatenate(images_list[i: i + n_sq], pad_value=pad_value, axis=0)
                for i in range(0, n_sq, n_full-n_sq)]

    img_grid = im_concatenate(img_grid, pad_value=pad_value, axis=1)

    return img_grid


def im_load(image_path: str,
         as_batch: bool = False,
         as_srgb: bool = True,
         as_float32: bool = True) -> np.ndarray:
    """
    Loads an image from a file.
    :param image_path: path to an image file.
    :param resize: required size for the output image - tuple (H, W).
    :param as_batch: whether or not to expand the first (batch) dimension.
    :param as_srgb: whether or not to convert to SRGB format.
    :param as_float32: whether or not to convert to float32 precision
    :param rotate: whether or not rotate the image
    in range [0,1], otherwise uint8 in range [0, 255]
    :return: array of shape (H, W, C) or (1, H, W, C).
    """

    typeguard.check_argument_types()
    # assert types.is_image(image_path), f'Unsupported image format in given file: {image_path}'
    image_pil = Image.open(image_path)

    # if rotate:
    #     image_pil = _rotate_pil_if_needed(image_pil)

    if as_srgb:
        image_pil = to_srgb(image_pil)

    image_pil = image_pil.convert('RGB')

    image_np = from_pil(image_pil, convert_to_float32=as_float32)

    # if resize:
        # image_np = im_resize(image_np, (resize[1], resize[0]))

    if as_batch:
        image_np = image_np[None]

    return image_np


def im_save(image: np.ndarray, output_path: str, quality: int = 100) -> None:
    """
    Saved an image to a file.
    :param image: The image to save.
    :param output_path: The path to save the image in.
    :param quality: The save quality: 0 (worst quality) - 100 (best quality).
    :return: None
    """

    typeguard.check_argument_types()
    # assert types.is_image(output_path), f'Unsupported image format in output file: {output_path}'
    pil_image = to_pil(image)
    pil_image.save(output_path, quality=quality)


def to_pil(image: np.ndarray) -> Image.Image:
    """
    Converts an np.ndarray image to PIL.Image.
    :param image: The image to convert.
    :return: The converted PIL image.
    """

    typeguard.check_argument_types()
    assert len(image.shape) in (2, 3), \
        f'Invalid image shape: {image.shape}. Shape must be of length 2 (greyscale) or 3 (RGB).'

    image_uint8 = to_uint8(image)
    image_uint8 = np.squeeze(image_uint8)  # for grayscale

    if len(image_uint8.shape) == 3:
        image_pil = Image.fromarray(image_uint8, 'RGB')
    else:
        image_pil = Image.fromarray(image_uint8, 'L')

    return image_pil


def from_pil(pil_image: Image.Image, convert_to_float32: bool = True) -> np.ndarray:
    """
    Converts a PIL.Image to an np.ndarray image.
    :param pil_image: A PIL.Image to convert.
    :param convert_to_float32: Whether or not to return the result as a float32 image in the range [0, 1].
    :return: The converted np.ndarray image.
    """

    typeguard.check_argument_types()
    # noinspection PyTypeChecker
    np_image = np.array(pil_image)
    if convert_to_float32:
        np_image = to_float32(np_image)

    return np_image


def to_float32(image: np.ndarray) -> np.ndarray:
    """
    Converts a uint8 image to a float32 image in the range [0, 1].
    :param image: The image to convert.
    :return: The converted float32 image.
    """

    typeguard.check_argument_types()
    if image.dtype == np.float32:
        return image

    image_float32 = np.asarray(image, dtype='float32') / 255.0
    return image_float32


def to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Converts a float32 image to a uint8 image in the range [0, 255].
    :param image: The image to convert.
    :return: The converted uint8 image.
    """

    typeguard.check_argument_types()
    if image.dtype == np.uint8:
        return image

    image_uint8 = np.asarray(image * 255, dtype='uint8')
    return image_uint8


def to_jpeg_bytes(image: np.ndarray, quality: int = 100) -> bytes:
    """
    Encodes an image as JPEG and returns its bytes object.
    :param image an image supported by array_to_img (H, W, C)
    :param quality: The save quality: 0 (worst quality) - 100 (best quality).
    :return: a bytes object of the
    """

    typeguard.check_argument_types()
    pil_image = to_pil(image)
    image_as_bytes = io.BytesIO()
    if quality is None:
        pil_image.save(image_as_bytes, format='JPEG')
    else:
        pil_image.save(image_as_bytes, format='JPEG', quality=quality)
    return image_as_bytes.getvalue()


def to_srgb(pil_image: Image.Image) -> Image.Image:
    """
    Converts a PIL image to sRGB color space (if possible).
    :param pil_image: The PIL image to convert.
    :return: The SRGB converted image.
    """

    typeguard.check_argument_types()
    if _is_srgb(pil_image):
        return pil_image

    icc = pil_image.info.get('icc_profile', '')
    if icc:
        io_handle = io.BytesIO(icc)  # virtual file
        src_profile = ImageCms.ImageCmsProfile(io_handle)
        dst_profile = ImageCms.createProfile('sRGB')
        try:
            pil_image = ImageCms.profileToProfile(pil_image, src_profile, dst_profile, outputMode='RGB')
        except ImageCms.PyCMSError as e:
            getLogger('image_conversions').warning(f'could not convert color profile: {e}')

    return pil_image


def _is_srgb(pil_image: Image.Image) -> bool:
    """
    Checks whether or not a given PIL image is in the sRGB color space.
    :param pil_image: The PIL image to check.
    :return: True iff the given PIL image is in the sRGB color space.
    """

    typeguard.check_argument_types()
    profile = pil_image.info.get('icc_profile', '')
    if profile == '':
        return True  # Assume sRGB by default

    return b'sRGB' in profile


def plot_losses_metrics(losses: TensorBatch, name: str, out_path_savefig: Optional[str] = None):
    metric_names = []
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for metric_name, val in losses.as_dict().items():
        if val is not None:
            ax.plot(val.detach().cpu().numpy().ravel())
            metric_names.append(metric_name)
    ax.legend(metric_names)
    plt.title(name)
    if out_path_savefig is not None:
        fig.savefig(os.path.join(out_path_savefig))
    else:
        plt.show()


def fft2_from_rfft(x_rfft: torch.Tensor, size2d: Tuple[int, int]) -> torch.Tensor:
    fft_freq = (torch.fft.fftfreq(size2d[1])*size2d[1]).numpy().astype(int)
    rftt_freq = (torch.fft.rfftfreq(size2d[1])*size2d[1]).numpy().astype(int)
    torch_device = x_rfft.device
    none_neq_ind = rftt_freq[fft_freq[fft_freq >= 0]]
    neg_ind = rftt_freq[-1*fft_freq[fft_freq < 0]]
    x_fft_pos_freq = torch.index_select(x_rfft, -1, torch.tensor(none_neq_ind, device=torch_device))
    x_fft_neg_freq = torch.index_select(x_rfft, -1, torch.tensor(neg_ind, device=torch_device))
    y_neg_ind = [0] + list(range(size2d[0]-1, 0, -1))
    x_fft_neg_freq = torch.index_select(x_fft_neg_freq, -2, torch.tensor(y_neg_ind, device=torch_device))
    if torch.is_complex(x_rfft):
        x_fft_neg_freq.imag = -1 * x_fft_neg_freq.imag
    x_fft = torch.cat([x_fft_pos_freq, x_fft_neg_freq], -1)
    return x_fft
