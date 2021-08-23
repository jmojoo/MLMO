import chainer
import chainer.functions as F
import chainer.links as L
from math import ceil


class AlexNet(chainer.Chain):

    insize = 227

    def __init__(self):
        super(AlexNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 96, 11, stride=4)
            self.conv2 = L.Convolution2D(96, 256, 5, pad=2)
            self.conv3 = L.Convolution2D(256, 384, 3, pad=1)
            self.conv4 = L.Convolution2D(384, 384, 3, pad=1)
            self.conv5 = L.Convolution2D(384, 256, 3, pad=1)
            self.fc6 = L.Linear(256 * 6 * 6, 4096)
            self.fc7 = L.Linear(4096, 4096)

    def __call__(self, x, t=None):
        with chainer.no_backprop_mode():
            h = self.conv1(x)
            h = F.max_pooling_2d(F.relu(
                F.local_response_normalization(h)), 3, stride=2)
            h = F.max_pooling_2d(F.relu(
                F.local_response_normalization(self.conv2(h))), 3, stride=2)
            h = F.relu(self.conv3(h))
            h = F.relu(self.conv4(h))
            h = F.relu(self.conv5(h))
            stage1 = F.max_pooling_2d(h, 3, stride=2)

        h = F.dropout(F.relu(self.fc6(stage1)), ratio=0.5)
        h = F.dropout(F.relu(self.fc7(h)), ratio=0.5)

        return stage1, h
