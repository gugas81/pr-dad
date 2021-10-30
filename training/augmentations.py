
import random
from typing import Tuple, Sequence
import torch
from torch import Tensor


def adjust_gamma(img: Tensor, gamma: float, gain: float = 1) -> Tensor:
    """Perform gamma correction on an image.

    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:

        I_out = 255 * gain * ((I_in / 255) ** gamma)

    See https://en.wikipedia.org/wiki/Gamma_correction for more details.

    Args:
        img (np.ndarray): CV Image to be adjusted.
        gamma (float): Non negative real number. gamma larger than 1 make the
            shadows darker, while gamma smaller than 1 make dark regions
            lighter.
        gain (float): The constant multiplier.
    """
#     if not _is_numpy_image(img):
#         raise TypeError('img should be CV Image. Got {}'.format(type(img)))

#     if gamma < 0:
#         raise ValueError('Gamma should be a non-negative real number')

    img_corrected = img * 0.5 + 0.5
    img_corrected = 1.0 * gain * (img_corrected ** gamma)
    img_corrected = (img_corrected - 0.5) * 2.0
    return img_corrected


class RandomGammaCorrection(object):
    def __init__(self, gamma_range: Tuple[float, float] = (0.5, 2)):
        # super().__init__()
        self._gamma_range = gamma_range
        # self.eval()
        # self.to(device='cpu')

    def __call__(self, img: Tensor) -> Tensor:
        gamma_rnd = random.uniform(self._gamma_range[0], self._gamma_range[1])
        return adjust_gamma(img, gamma_rnd)

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += f'(random_gamma_range=({self._gamma_range[0]}, {self._gamma_range[1]}))'
        return format_string


class RandomAddGaussianNoise(object):
    def __init__(self, noise_factor: float = 0.1, rand_factor: bool = False):
        self._factor = noise_factor
        self._rand_factor = rand_factor

    def __call__(self, img: Tensor) -> Tensor:
        noise_factor = random.uniform(0, self._factor) if self._rand_factor else self._factor
        noise = noise_factor * torch.randn_like(img)
        noised_img = img + noise
        return noised_img

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += f'(noise_factor={self._factor}, use_rnd_factor={self._rand_factor})'
        return format_string


class ClipValue(object):
    def __init__(self, clip_values: Sequence[float] = (-1.0, 1.0)):
        self._clip_values = clip_values

    def __call__(self, img: Tensor) -> Tensor:
        return img.clip(min=self._clip_values[0], max=self._clip_values[1])

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += f'(range=[{self._clip_values[0]}:{self._clip_values[1]}])'
        return format_string
