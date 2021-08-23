import collections
import os
import sys

import numpy
try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e

import chainer
from chainer.dataset.convert import concat_examples
from chainer.dataset import download
from chainer import function
from chainer.functions.activation.relu import relu
from chainer.functions.activation.tanh import tanh
from chainer.functions.activation.softmax import softmax
from chainer.functions.array.reshape import reshape
from chainer.functions.math.sum import sum
from chainer.functions.noise.dropout import dropout
from chainer.functions import max_pooling_2d
from chainer.functions import unpooling_2d
from chainer.functions.pooling import average_pooling_2d
from chainer.functions import max_pooling_2d
from chainer.initializers import constant
from chainer.initializers import normal
from chainer import link
from chainer.links.connection.convolution_2d import Convolution2D
from chainer.links.connection.linear import Linear
from chainer import ChainList
from chainer.links import BatchNormalization, PReLU
from chainer.serializers import npz
from chainer.utils import argument
from chainer.utils import imgproc
from chainer.variable import Variable
from chainer.functions import concat


class VGGLayers(link.Chain):

    """A pre-trained CNN model provided by VGG team.
    You can use ``VGG16Layers`` or ``VGG19Layers`` for concrete
    implementations. During initialization, this chain model
    automatically downloads the pre-trained caffemodel, convert to
    another chainer model, stores it on your local directory,
    and initializes all the parameters with it.
    This model would be useful when you want to extract a semantic
    feature vector from a given image, or fine-tune the model
    on a different dataset.
    Note that these pre-trained models are released under Creative Commons
    Attribution License.
    If you want to manually convert the pre-trained caffemodel to a chainer
    model that can be specified in the constructor,
    please use ``convert_caffemodel_to_npz`` classmethod instead.
    See: K. Simonyan and A. Zisserman, `Very Deep Convolutional Networks
    for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`_
    Args:
        pretrained_model (str): the destination of the pre-trained
            chainer model serialized as a ``.npz`` file.
            If this argument is specified as ``auto``,
            it automatically downloads the caffemodel from the internet.
            Note that in this case the converted chainer model is stored
            on ``$CHAINER_DATASET_ROOT/pfnet/chainer/models`` directory,
            where ``$CHAINER_DATASET_ROOT`` is set as
            ``$HOME/.chainer/dataset`` unless you specify another value
            as a environment variable. The converted chainer model is
            automatically used from the second time.
            If the argument is specified as ``None``, all the parameters
            are not initialized by the pre-trained model, but the default
            initializer used in the original paper, i.e.,
            ``chainer.initializers.Normal(scale=0.01)``.
        n_layers (int): The number of layers of this model. It should be
            either 16 or 19.
    Attributes:
        available_layers (list of str): The list of available layer names
            used by ``forward`` and ``extract`` methods.
    """

    def __init__(self, pretrained_model='auto', n_layers=16):
        super(VGGLayers, self).__init__()
        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            init = constant.Zero()
            kwargs = {'initialW': init, 'initial_bias': init}
        else:
            # employ default initializers used in the original paper
            kwargs = {
                # 'initialW': normal.Normal(0.01),
                'initialW': normal.LeCunNormal(),
                'initial_bias': constant.Zero(),
            }

        if n_layers not in [16, 19]:
            raise ValueError(
                'The n_layers argument should be either 16 or 19, '
                'but {} was given.'.format(n_layers)
            )

        with self.init_scope():
            self.conv1_1 = Convolution2D(3, 64, 3, 1, 1, **kwargs)
            self.conv1_2 = Convolution2D(64, 64, 3, 1, 1, **kwargs)
            self.conv2_1 = Convolution2D(64, 128, 3, 1, 1, **kwargs)
            self.conv2_2 = Convolution2D(128, 128, 3, 1, 1, **kwargs)
            self.conv3_1 = Convolution2D(128, 256, 3, 1, 1, **kwargs)
            self.conv3_2 = Convolution2D(256, 256, 3, 1, 1, **kwargs)
            self.conv3_3 = Convolution2D(256, 256, 3, 1, 1, **kwargs)
            self.conv4_1 = Convolution2D(256, 512, 3, 1, 1, **kwargs)
            self.conv4_2 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv4_3 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_1 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_2 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_3 = Convolution2D(512, 512, 3, 1, 1, **kwargs)

            # self.fc = Linear(100)
            self.fc6 = Linear(512 * 7 * 7, 4096, **kwargs)
            self.fc7 = Linear(4096, 4096, **kwargs)
            # self.fc8 = Linear(4096, 1000, **kwargs)
            # self.out = Linear(None, n_class)

        if pretrained_model == 'auto':
            _retrieve(
                'VGG_ILSVRC_16_layers.npz',
                'https://www.robots.ox.ac.uk/%7Evgg/software/very_deep/'
                'caffe/VGG_ILSVRC_16_layers.caffemodel',
                self)
        elif pretrained_model:
            npz.load_npz(pretrained_model, self, strict=False)

    @property
    def functions(self):
        # This class will not be used directly.
        raise NotImplementedError

    @property
    def available_layers(self):
        return list(self.functions.keys())

    @classmethod
    def convert_caffemodel_to_npz(cls, path_caffemodel, path_npz):
        """Converts a pre-trained caffemodel to a chainer model.
        Args:
            path_caffemodel (str): Path of the pre-trained caffemodel.
            path_npz (str): Path of the converted chainer model.
        """

        # As CaffeFunction uses shortcut symbols,
        # we import CaffeFunction here.
        from chainer.links.caffe.caffe_function import CaffeFunction
        caffemodel = CaffeFunction(path_caffemodel)
        npz.save_npz(path_npz, caffemodel, compression=False)

    def __call__(self, x, layers=None, **kwargs):
        """forward(self, x, layers=['prob'])
        Computes all the feature maps specified by ``layers``.
        Args:
            x (~chainer.Variable): Input variable. It should be prepared by
                ``prepare`` function.
            layers (list of str): The list of layer names you want to extract.
                If ``None``, 'prob' will be used as layers.
        Returns:
            Dictionary of ~chainer.Variable: A dictionary in which
            the key contains the layer and the value contains the
            corresponding feature map variable.
        """

        if layers is None:
            layers = ['prob']

        if kwargs:
            argument.check_unexpected_kwargs(
                kwargs, test='test argument is not supported anymore. '
                'Use chainer.using_config'
            )
            argument.assert_kwargs_empty(kwargs)

        # print(self.xp.min(x), self.xp.max(x))
        # print(x[(x > 0) * (x < 1)][:50])
        # exit()
        # b = self.conv1_1.b.array
        # print(self.xp.min(b), self.xp.max(b))
        # exit()

        h = x
        # l = [2, 2, 3, 3, 3]
        with chainer.no_backprop_mode():
            h = self.block(h, 1, 2)
            # print(self.xp.min(h.array), self.xp.max(h.array))
            # exit()
            h = _max_pooling_2d(h)

            h = self.block(h, 2, 2)
            h = _max_pooling_2d(h)

            h = self.block(h, 3, 3)
            h = _max_pooling_2d(h)

            h = self.block(h, 4, 3)
            h = _max_pooling_2d(h)

            h = self.block(h, 5, 3)
            h_conv = _max_pooling_2d(h)

        h = relu(self.fc6(h_conv))
        #
        h = relu(self.fc7(h))
        # h = average_pooling_2d.average_pooling_2d(h, h.shape[2])

        # h = self.out(h)

        # print(h)
        # print(self.xp.min(h.array), self.xp.max(h.array))
        # exit()

        return h_conv, h

    def predict(self, images, oversample=True):
        """Computes all the probabilities of given images.
        Args:
            images (iterable of PIL.Image or numpy.ndarray): Input images.
                When you specify a color image as a :class:`numpy.ndarray`,
                make sure that color order is RGB.
            oversample (bool): If ``True``, it averages results across
                center, corners, and mirrors. Otherwise, it uses only the
                center.
        Returns:
            ~chainer.Variable: Output that contains the class probabilities
            of given images.
        """
        print("Boy getting called over here")
        input("Press enter to continue; ")
        # x = concat_examples([prepare(img, size=(256, 256)) for img in images])
        x = images
        if oversample:
            x = imgproc.oversample(x, crop_dims=(224, 224))
        else:
            x = x[:, :, 16:240, 16:240]
        # Use no_backprop_mode to reduce memory consumption
        with function.no_backprop_mode(), chainer.using_config('train', False):
            x = Variable(self.xp.asarray(x))
            y = softmax(self.forward(x))
            if oversample:
                n = len(y) // 10
                y_shape = y.shape[1:]
                y = reshape(y, (n, 10) + y_shape)
                y = sum(y, axis=1) / 10
        return y


class VGG16Layers(VGGLayers):

    """A pre-trained CNN model with 16 layers provided by VGG team.
    During initialization, this chain model automatically downloads
    the pre-trained caffemodel, convert to another chainer model,
    stores it on your local directory, and initializes all the parameters
    with it. This model would be useful when you want to extract a semantic
    feature vector from a given image, or fine-tune the model
    on a different dataset.
    Note that this pre-trained model is released under Creative Commons
    Attribution License.
    If you want to manually convert the pre-trained caffemodel to a chainer
    model that can be specified in the constructor,
    please use ``convert_caffemodel_to_npz`` classmethod instead.
    See: K. Simonyan and A. Zisserman, `Very Deep Convolutional Networks
    for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`_
    Args:
        pretrained_model (str): the destination of the pre-trained
            chainer model serialized as a ``.npz`` file.
            If this argument is specified as ``auto``,
            it automatically downloads the caffemodel from the internet.
            Note that in this case the converted chainer model is stored
            on ``$CHAINER_DATASET_ROOT/pfnet/chainer/models`` directory,
            where ``$CHAINER_DATASET_ROOT`` is set as
            ``$HOME/.chainer/dataset`` unless you specify another value
            as a environment variable. The converted chainer model is
            automatically used from the second time.
            If the argument is specified as ``None``, all the parameters
            are not initialized by the pre-trained model, but the default
            initializer used in the original paper, i.e.,
            ``chainer.initializers.Normal(scale=0.01)``.
    Attributes:
        available_layers (list of str): The list of available layer names
            used by ``forward`` and ``extract`` methods.
    """

    def __init__(self, pretrained_model='auto', n_class=None):
        super(VGG16Layers, self).__init__(pretrained_model, 16)

    @property
    def functions(self):
        return collections.OrderedDict([
            ('conv1_1', [self.conv1_1, relu]),
            ('conv1_2', [self.conv1_2, relu]),
            ('conv2_1', [self.conv2_1, relu]),
            ('conv2_2', [self.conv2_2, relu]),
            ('conv3_1', [self.conv3_1, relu]),
            ('conv3_2', [self.conv3_2, relu]),
            ('conv3_3', [self.conv3_3, relu]),
            ('conv4_1', [self.conv4_1, relu]),
            ('conv4_2', [self.conv4_2, relu]),
            ('conv4_3', [self.conv4_3, relu]),
            ('conv5_1', [self.conv5_1, relu]),
            ('conv5_2', [self.conv5_2, relu]),
            ('conv5_3', [self.conv5_3, relu]),
        ])

    def block(self, x, i, size):
        for l in range(1, size + 1):
            layer = 'conv{}_{}'.format(i, l)
            for f in range(len(self.functions[layer])):
                x = self.functions[layer][f](x)
        return x


def prepare(image, size=(224, 224)):
    """Converts the given image to the numpy array for VGG models.
    Note that you have to call this method before ``forward``
    because the pre-trained vgg model requires to resize the given image,
    covert the RGB to the BGR, subtract the mean,
    and permute the dimensions before calling.
    Args:
        image (PIL.Image or numpy.ndarray): Input image.
            If an input is ``numpy.ndarray``, its shape must be
            ``(height, width)``, ``(height, width, channels)``,
            or ``(channels, height, width)``, and
            the order of the channels must be RGB.
        size (pair of ints): Size of converted images.
            If ``None``, the given image is not resized.
    Returns:
        numpy.ndarray: The converted output array.
    """

    if not available:
        raise ImportError('PIL cannot be loaded. Install Pillow!\n'
                          'The actual import error is as follows:\n' +
                          str(_import_error))
    dtype = chainer.get_dtype()
    if isinstance(image, numpy.ndarray):
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image[0, :, :]
            elif image.shape[0] == 3:
                image = image.transpose((1, 2, 0))
        image = Image.fromarray(image.astype(numpy.uint8))
    image = image.convert('RGB')
    if size:
        image = image.resize(size)
    image = numpy.asarray(image, dtype=dtype)
    image = image[:, :, ::-1]
    image -= numpy.array(
        [103.939, 116.779, 123.68], dtype=dtype)
    image = image.transpose((2, 0, 1))
    return image


def _max_pooling_2d(x):
    return max_pooling_2d(x, ksize=2)


def _make_npz(path_npz, url, model):
    path_caffemodel = download.cached_download(url)
    sys.stderr.write(
        'Now loading caffemodel (usually it may take few minutes)\n')
    sys.stderr.flush()
    VGGLayers.convert_caffemodel_to_npz(path_caffemodel, path_npz)
    npz.load_npz(path_npz, model, strict=False)
    return model


def _retrieve(name, url, model):
    root = download.get_dataset_directory('pfnet/chainer/models/')
    path = os.path.join(root, name)
    return download.cache_or_load_file(
        path, lambda path: _make_npz(path, url, model),
        lambda path: npz.load_npz(path, model, strict=False))