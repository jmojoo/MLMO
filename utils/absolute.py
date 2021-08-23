from chainer import function_node
from chainer.utils import type_check
from chainer import utils
import numpy
from chainer.backends import cuda


class Absolute(function_node.FunctionNode):
    @property
    def label(self):
        return '|_|'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        self.retain_inputs((0,))
        return utils.force_array(abs(x[0])),

    def backward(self, indexes, grad_outputs):
        x = self.get_retained_inputs()[0]
        return AbsoluteGrad(x.data).apply(grad_outputs)


class AbsoluteGrad(function_node.FunctionNode):

    def __init__(self, x):
        super(AbsoluteGrad, self).__init__()
        self.x = x

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('gy',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, inputs):
        return utils.force_array(numpy.sign(self.x) + 1e-7 * inputs[0]),

    def forward_gpu(self, inputs):
        gx0 = cuda.elementwise(
            'T x0, T gy', 'T gx0',
            'gx0 = ((x0 > 0) - (x0 < 0) + 1e-7) * gy',
            'abs_bwd')(self.x, inputs[0])
        return gx0,

    def backward(self, indexes, grad_outputs):
        return AbsoluteGrad(self.x).apply(grad_outputs)


def absolute(self):
    """Element-wise absolute.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Absolute().apply((self,))[0]