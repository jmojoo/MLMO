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


class RetrievalEvaluator(chainer.training.extensions.Evaluator):

    trigger = 1, 'epoch'
    default_name = 'val'
    priority = chainer.training.PRIORITY_EDITOR

    def __init__(
            self, query_iterator, query_codes, query_labels,
            database_iterator, database_codes, db_labels,
            target, args, device=None, hash=False):
        super(RetrievalEvaluator, self).__init__(
            query_iterator, target, device=device)
        self._iterators = {'query': query_iterator,
                           'database': database_iterator}
        self.database_codes = database_codes
        self.query_codes = query_codes
        self.query_labels = query_labels
        self.db_labels = db_labels
        self.R = args.R
        self.stage = args.stage
        self.code_batchsize = args.code_batch_n
        self.output_dim = args.output_dim
        self.hash = hash

    def evaluate(self):
        sys.stdout.write("validating...\r")
        query_it = self._iterators['query']
        database_it = self._iterators['database']
        target = self._targets['main']

        if hasattr(query_it, 'reset'):
            query_it.reset()
            q_it = query_it
        else:
            q_it = copy.copy(query_it)

        if hasattr(database_it, 'reset'):
            database_it.reset()
            db_it = database_it
        else:
            db_it = copy.copy(database_it)

        db_embed = []
        q_embed = []

        total_batch = math.ceil(len(db_it.dataset)/db_it.batch_size)
        count = 0
        running_sum = 0
        # start = time.time()
        for batch in db_it:
            with chainer.using_config('train', False):
                out = target.get_embedding_test(batch)

            # if db_it.dataset._transform.ensemble:
            #     out = xp.mean(out.reshape(-1, 10, self.output_dim), axis=1)
            db_embed.append(out)

            count += 1
            # running_sum += 1 / (time.time() - start)
            # etime = running_sum / count
            etime = 0
            sys.stdout.write(
                "computing database embeddings...%d/%d, %f/iter\r" % (count, total_batch, etime))
            sys.stdout.flush()
            # start = time.time()

        total_batch = math.ceil(len(q_it.dataset) / q_it.batch_size)
        count = 0
        for batch in q_it:
            with chainer.using_config('train', False):
                out = target.get_embedding_test(batch)

            # if q_it.dataset._transform.ensemble:
            #     out = xp.mean(out.reshape(-1, 10, self.output_dim), axis=1)
            q_embed.append(out)

            count += 1
            sys.stdout.write(
                "computing query embeddings...%d/%d\r" % (count, total_batch))
            sys.stdout.flush()

        # if db_it.dataset._transform.ensemble:
        db_embed = xp.concatenate(db_embed, axis=0)
        # else:
        #     db_embed = xp.stack(db_embed)

        # if q_it.dataset._transform.ensemble:
        q_embed = xp.concatenate(q_embed, axis=0)
        # else:
        #     q_embed = xp.stack(q_embed)
        if q_it.dataset._transform.ensemble:
            db_embed = xp.mean(db_embed.reshape(-1, 10, self.output_dim), axis=1)
            q_embed = xp.mean(q_embed.reshape(-1, 10, self.output_dim), axis=1)

        # scale = 1 / (1 + xp.exp(-target.scale.array))
        db_embed = cuda.to_cpu(db_embed)
        q_embed = q_embed
        # if self.stage == 2:
        #     beta = 1 / (1 + np.exp(-cuda.to_cpu(target.beta.array)))
        #     inds = np.argsort(-beta)
        #     db_embed = db_embed[:, inds[:64]]
        #     q_embed = q_embed[:, inds[:64]]

        center_backup = target.C.array.copy()
        print("initializing centers for database embeddings...", end='\r')
        target.initialize_centers(db_embed)
        print("updating database codes and centers...", end='\r')

        target.update_codes_and_centers((db_embed, self.database_codes))
        target.update_codes_batch((q_embed, self.query_codes), self.code_batchsize)

        db_embed = xp.array(db_embed, dtype=xp.float32)
        d_codes = self.database_codes
        db_labels = self.db_labels
        q_labels = self.query_labels
        q_codes = self.query_codes

        print("calculating mAP...", end='\r')
        if self.hash:
            q_embed = xp.sign(q_embed)
            db_embed = xp.sign(db_embed)
        mAP_feature = mAP(q_embed, q_labels,
                          db_embed, db_labels, 'inner_product', self.R)

        mAP_at_feat = mAP_at(q_embed, q_labels,
                          db_embed, db_labels, 'inner_product', self.R, 0.5)

        WAP_feature = WAP(q_embed, q_labels,
                          db_embed, db_labels, 'inner_product', self.R)

        C = target.C.array[:, :db_embed.shape[-1]]
        if self.stage == 1:
            db_reconstr = xp.dot(d_codes, C)
            q_reconstr = xp.dot(q_codes, C)
        else:
            C = xp.split(C, 2, axis=1)
            d_c = xp.split(d_codes, 2, axis=1)
            q_c = xp.split(q_codes, 2, axis=1)
            db_reconstr = xp.concatenate(
                (xp.dot(d_c[0], C[0]), xp.dot(d_c[1], C[1])), axis=1
            )
            q_reconstr = xp.concatenate(
                (xp.dot(q_c[0], C[0]), xp.dot(q_c[1], C[1])), axis=1
            )

        mAP_AQD = mAP(q_embed, q_labels,
                      db_reconstr, db_labels,
                      'inner_product', self.R)

        mAP_05 = mAP_at(q_embed, q_labels,
                      db_reconstr, db_labels,
                      'inner_product', self.R, 0.5)

        WAP_AQD = WAP(q_embed, q_labels,
                      db_reconstr, db_labels,
                      'inner_product', self.R)

        # mAP_SQD = mAP(q_reconstr, q_labels,
        #               db_reconstr, db_labels,
        #               'inner_product', self.R)

        report = {'mAP_feat': mAP_feature,
                  'mAP_AQD': mAP_AQD,
                  'mAP@0.5': mAP_05,
                  'mAP@.5feat': mAP_at_feat,
                  'WAP_feat': WAP_feature,
                  'WAP': WAP_AQD}
        observation = {}
        with reporter.report_scope(observation):
            reporter.report(report, target)
        target.C.array = center_backup
        return observation


def mAP(q_vectors, q_labels, db_vectors, db_labels, dist_func, R):
    APs = []
    positions = xp.arange(1, db_labels.shape[0] + 1).astype(xp.float32)[:R]

    for q_v, q_l in six.moves.zip(q_vectors, q_labels):
        q_l = q_l.copy()

        q_l = xp.sign(q_l)
        q_l[q_l == 0] = -1

        dist = distance(xp, q_v, db_vectors, dist_func=dist_func)
        order = xp.argsort(dist)
        lbl_ordered = db_labels[order[:R]]

        lbl_ordered = xp.sign(lbl_ordered)

        indicator = xp.any(lbl_ordered == q_l, axis=1).astype(xp.int32)
        sum_rel = xp.cumsum(indicator)
        cut_off_precision = sum_rel / positions
        rel = xp.sum(indicator)
        if rel != 0:
            AP = xp.sum(
                cut_off_precision * indicator) / rel
            APs.append(AP.astype(xp.float32))

    del positions

    APs = xp.asarray(APs, dtype=xp.float32)
    return np.mean(APs.get())


def mAP_at(q_vectors, q_labels, db_vectors, db_labels, dist_func, R, thresh):
    APs = []
    positions = xp.arange(1, R + 1).astype(xp.float32)

    for q_v, q_l in six.moves.zip(q_vectors, q_labels):

        dist = distance(xp, q_v, db_vectors, dist_func=dist_func)
        order = xp.argsort(dist)
        lbl_ordered = db_labels[order[:R]]

        intersection = xp.sum(xp.minimum(q_l, lbl_ordered), axis=1)
        union = xp.sum(xp.maximum(q_l, lbl_ordered), axis=1)
        iou = intersection / union

        # iou = iou[xp.where(iou > 0)[0]]
        # positions = positions[:iou.shape[0]]

        indicator = (iou > thresh).astype(xp.int32)

        sum_rel = xp.cumsum(indicator)
        cut_off_precision = sum_rel / positions
        rel = xp.sum(indicator)
        if rel != 0:
            AP = xp.sum(
                cut_off_precision * indicator) / rel
            APs.append(AP)

    del positions
    APs = xp.asarray(APs, dtype=xp.float32)
    return np.mean(APs.get())


def WAP(q_vectors, q_labels, db_vectors, db_labels, dist_func, R):
    APs = []
    positions = xp.arange(1, R + 1).astype(xp.float32)

    for q_v, q_l in six.moves.zip(q_vectors, q_labels):
        q_l = q_l.copy()

        q_l = xp.sign(q_l)
        q_l[q_l == 0] = -1

        dist = distance(xp, q_v, db_vectors, dist_func=dist_func)
        order = xp.argsort(dist)
        lbl_ordered = db_labels[order[:R]]

        lbl_ordered = xp.sign(lbl_ordered)

        indicator = xp.any(lbl_ordered == q_l, axis=1).astype(xp.int32)
        shared = (lbl_ordered == q_l).astype(xp.int32)
        shared = xp.sum(shared, axis=1)
        cum_shared = xp.cumsum(shared)
        ACG = cum_shared / positions
        rel = xp.sum(indicator)
        if rel != 0:
            AP = xp.sum(
                ACG * indicator) / rel
            APs.append(AP)

    del positions
    APs = xp.asarray(APs, dtype=xp.float32)
    return np.mean(APs.get())


def distance(backend, x, y=None, pair=False, dist_func='inner_product'):
    if y is None:
        y = x
    if dist_func == 'inner_product':
        return inner_product(backend, x, y, pair)
    if pair:
        x = backend.expand_dims(x, 1)
        y = backend.expand_dims(y, 0)
    if dist_func == 'euclid2':
        return euclid2(backend, x, y)


def euclid2(backend, x, y):
    return backend.sum(backend.square(x - y), axis=-1)


def inner_product(backend, x, y, pair=False):
    if pair:
        return - backend.inner(x, y)
    else:
        return - backend.sum(x * y, axis=-1)


