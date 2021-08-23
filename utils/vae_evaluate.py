import copy
import numpy as np
import six
import cupy
import math

from chainer import cuda
from chainer import function
from chainer import reporter
import chainer.training.extensions
import sys

from chainercv.utils import apply_to_iterator
import time

xp = cupy


class VAEEvaluator(chainer.training.extensions.Evaluator):

    default_name = 'val'
    priority = chainer.training.PRIORITY_EDITOR

    def __init__(self, val_iterator, target, args, device=None):
        super(VAEEvaluator, self).__init__(
            val_iterator, target, device=device)
        self.iterator = val_iterator
        self.target = target

    def evaluate(self):
        sys.stdout.write("validating...\r")
        val_it = self.iterator
        target = self.target

        if hasattr(val_it, 'reset'):
            val_it.reset()
        else:
            val_it = copy.copy(val_it)

        N = len(val_it.dataset)
        total_batch = math.ceil(N / val_it.batch_size)
        count = 0
        rec_sum = 0
        latent_sum = 0
        # start = time.time()
        for batch in val_it:
            with chainer.using_config('train', False), \
                 chainer.no_backprop_mode():
                batch = xp.stack(batch)
                rec_loss, l_loss = target(xp.array(batch))

                rec_sum += xp.sum(rec_loss.array)
                latent_sum += xp.sum(l_loss.array)

            sys.stdout.write(
                "computing losses embeddings...%d/%d\r" % (count, total_batch))
            sys.stdout.flush()
            count += 1
            # start = time.time()

        rec_loss = rec_sum / total_batch
        latent_loss = latent_sum / total_batch

        report = {'rec_loss': rec_loss,
                  'latent_loss': latent_loss}
        observation = {}
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation