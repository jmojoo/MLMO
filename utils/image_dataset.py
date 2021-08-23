import os

import cv2
try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e
import six
import numpy as np
from chainercv import transforms

import chainer
from chainer.dataset import dataset_mixin


def _read_image_as_array(path, dtype):
    image = cv2.imread(path)
    return image


def _postprocess_image(image):
    if image.ndim == 2:
        # image is greyscale
        image = image[..., None]
    return image


class Transform(object):

    def __init__(self, args, augment=False, ensemble=False):
        self.extractor = args.extractor
        self.augment = augment
        self.dataset = args.dataset
        self.ensemble = ensemble
        if 'alexnet' in self.extractor:
            self.size = (227, 227)
        elif 'darknet-19' in self.extractor:
            self.size = (320, 320)
        elif 'vgg16' in self.extractor:
            self.size = (224, 224)

    def __call__(self, in_data):

        img = in_data
        img = img.astype(np.float32)
        # if img.max() > 1:
        #     img /= 255.0
        if img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
            if img.shape[2] == 1:
                img = np.broadcast_to(img, (img.shape[0], img.shape[1], 3))
        if 'alexnet' in self.extractor or 'vgg16' in self.extractor:
            img = cv2.resize(img, (256, 256))

        if 'alexnet' in self.extractor:
            if img.max() <= 1:
                img *= 255.0

            # img = img[:, :, ::-1]  # RGB -> BGR # unnecessary if loaded using opencv
            mean_bgr = np.array([104, 117, 123], dtype=np.float32)
            img -= mean_bgr
        elif 'darknet-19' in self.extractor:
            if img.max() > 1:
                img /= 255.0
        elif 'vgg16' in self.extractor:
            if img.max() <= 1:
                img *= 255.0

            mean_bgr = np.array([103.939, 116.779, 123.68], dtype=np.float32)
            img -= mean_bgr

        if img.shape[2] == 3:
            img = np.transpose(img, (2, 0, 1))

        augment = chainer.global_config.train and self.augment
        if augment:
            if np.random.randint(2):
                img = transforms.random_flip(
                    img, x_random=True, y_random=False)

        if self.ensemble:
            img = transforms.ten_crop(img, self.size)
        else:
            img = transforms.resize(img, self.size)
        #
        return img


class ImageDataset(dataset_mixin.DatasetMixin):

    """Dataset of images built from a list of paths to image files.

    This dataset reads an external image file on every call of the
    :meth:`__getitem__` operator. The paths to the image to retrieve is given
    as either a list of strings or a text file that contains paths in distinct
    lines.

    Each image is automatically converted to arrays of shape
    ``channels, height, width``, where ``channels`` represents the number of
    channels in each pixel (e.g., 1 for grey-scale images, and 3 for RGB-color
    images).

    .. note::
       **This dataset requires the Pillow package being installed.** In order
       to use this dataset, install Pillow (e.g. by using the command ``pip
       install Pillow``). Be careful to prepare appropriate libraries for image
       formats you want to use (e.g. libpng for PNG images, and libjpeg for JPG
       images).

    .. warning::
       **You are responsible for preprocessing the images before feeding them
       to a model.** For example, if your dataset contains both RGB and
       grayscale images, make sure that you convert them to the same format.
       Otherwise you will get errors because the input dimensions are different
       for RGB and grayscale images.

    Args:
        paths (str or list of strs): If it is a string, it is a path to a text
            file that contains paths to images in distinct lines. If it is a
            list of paths, the ``i``-th element represents the path to the
            ``i``-th image. In both cases, each path is a relative one from the
            root path given by another argument.
        root (str): Root directory to retrieve images from.
        dtype: Data type of resulting image arrays. ``chainer.config.dtype`` is
            used by default (see :ref:`configuration`).

    """

    def __init__(self, paths, root='.', dtype=None):
        _check_pillow_availability()
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths
        self._root = root
        self._dtype = chainer.get_dtype(dtype)

    def __len__(self):
        return len(self._paths)

    def get_path(self, i):
        return self._paths[i]

    def get_example(self, i):
        path = os.path.join(self._root, self._paths[i])
        image = _read_image_as_array(path, self._dtype)

        return _postprocess_image(image)


def _check_pillow_availability():
    if not available:
        raise ImportError('PIL cannot be loaded. Install Pillow!\n'
                          'The actual import error is as follows:\n' +
                          str(_import_error))
