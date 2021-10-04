import chainer
from chainer import links as L, initializers
from chainer import functions as F
from models import base
import numpy as np


class MLMO(base.Base):
    def __init__(self, extractor, args):
        super(MLMO, self).__init__(args)

        self.coord_scale = args.coord_scale
        self.rn_output_dim = args.rn_output_dim
        self.rn_alpha = args.rn_alpha
        with self.init_scope():
            self.extractor = extractor
            self.g_ = G_theta(256)
            self.f_ = F_theta(1024, self.rn_output_dim)
            self.out = L.Linear(self.output_dim - self.rn_output_dim)
            self.bnorm = L.BatchRenormalization(1024)
            self.C = chainer.Parameter(self.xp.random.uniform(-1, 1,
                (self.subspace_n * self.subcenter_n, self.output_dim)
            ).astype(self.xp.float32))

    def RN(self, x, s2):
        n, c, h, w = x.shape
        spatial_area = h * w
        x = F.reshape(x, (n, c, spatial_area))

        h_hh = F.expand_dims(x, 2)
        h_ww = F.expand_dims(x, 3)

        h_hh = F.repeat(h_hh, spatial_area, axis=2)
        h_ww = F.repeat(h_ww, spatial_area, axis=3)

        s = F.reshape(s2, (n, s2.shape[1], 1, 1))
        s = F.broadcast_to(s, (n, s.shape[1], h_ww.shape[2], h_ww.shape[2]))
        h = F.concat((h_hh, h_ww, s), axis=1)

        h = F.transpose(h, (0, 2, 3, 1))
        ind = np.triu_indices(spatial_area)
        h = F.reshape(h, (n, spatial_area, spatial_area, -1))[:, ind[0], ind[1], :]
        triu = h.shape[1]

        h = F.reshape(h, (n * triu, -1))
        h = self.g_(h)

        h = F.reshape(h, (n, triu, -1))
        h = F.sum(h, axis=1)

        h = self.bnorm(h)

        h = F.concat((h, s2))
        h = self.f_(h)

        return h

    def get_embedding_train(self, x):
        x, s = self.extractor(x)
        s = F.tanh(self.out(s))
        if self.stage == 1:
            h = s
        else:
            coord = self.coord_scale * self.get_coord(x.shape[0], x.shape[2])
            x = F.concat((x, coord), axis=1)
            h = self.RN(x, s)
            h = F.tanh(h)
            h = F.concat((h, s), axis=1)
        return h

    def get_embedding_test(self, x):
        if x[0].ndim > 3:
            x = self.xp.vstack([self.xp.array(a) for a in x])
        else:
            x = self.xp.array(x)
        with chainer.no_backprop_mode(),\
                chainer.using_config('type_check', False):
            x, s2 = self.extractor(x)
            coord = self.coord_scale * self.get_coord(x.shape[0], x.shape[2])  # *100 for AlexNet
            x = F.concat((x, coord), axis=1)
            s = self.out(s2)
            s2 = F.tanh(s)

            if self.stage == 1:
                x = s2
            elif self.stage == 2:
                x = self.RN(x, s)
                x = F.tanh(x)
                x = F.concat((x, s2), axis=1)

        return x.array

    def mine_triplets(self, x, labels):
        x = self.get_embedding_train(x)
        b = x.shape[0]
        x = x.array
        unique_labels, counts = self.xp.unique(labels, return_counts=True)
        inds = self.xp.arange(len(labels)).astype(self.xp.int32)

        label1, label2 = self.xp.meshgrid(labels, labels)[::-1]
        ind1, ind2 = self.xp.meshgrid(inds, inds)[::-1]
        label1 = label1.reshape(b * b)
        label2 = label2.reshape(b * b)
        ind1 = ind1.reshape(b * b)
        ind2 = ind2.reshape(b * b)
        label_diff = label1 - label2
        match = label_diff == 0
        mismatch = label_diff != 0
        distance = self.xp.sum(self.xp.square(x[ind1] - x[ind2]),
                               axis=1)

        chosen = []
        anchors = []
        positives = []
        negatives = []
        max_positive = 3
        for cat in unique_labels:
            cat_inds = self.xp.where(labels == cat)[0]
            for i, cat_ind in enumerate(cat_inds):
                if i >= max_positive:
                    break

                if cat_ind in chosen:
                    anchors.append(chosen.index(cat_ind))
                else:
                    chosen.append(cat_ind)
                    anchors.append(len(chosen) - 1)

                # anchors.append(cat_ind)
                is_current_ind = ind1 == cat_ind
                p_inds = ind2[self.xp.where(
                    is_current_ind * match
                )]
                n_inds = ind2[self.xp.where(
                    is_current_ind * mismatch
                )]
                max_pos = p_inds[self.xp.argmax(distance[p_inds])]

                if max_pos in chosen:
                    positives.append(chosen.index(max_pos))
                else:
                    chosen.append(max_pos)
                    positives.append(len(chosen) - 1)

                min_neg = n_inds[self.xp.argmin(distance[n_inds])]
                if min_neg in chosen:
                    negatives.append(chosen.index(min_neg))
                else:
                    chosen.append(min_neg)
                    negatives.append(len(chosen) - 1)

        del x, distance, ind1, ind2, label1, label2

        return chosen, anchors, positives, negatives

    def __call__(self, anchor, positive, negative, codes, inds, embed, sim=None):
        x = self.xp.concatenate((anchor, positive, negative), axis=0)
        out = self.get_embedding_train(x)

        anchor, positive, negative = F.split_axis(out, 3, axis=0)

        pos_dist = F.square(anchor - positive)
        pos_dist_all = F.sum(pos_dist, axis=-1)

        neg_dist = F.square(anchor - negative)
        neg_dist_all = F.sum(neg_dist, axis=-1)

        zero = self.xp.zeros((1,), dtype=self.xp.float32)

        triplet_loss = F.maximum(pos_dist_all - neg_dist_all + sim * self.margin, zero)

        triplet_loss = F.mean(triplet_loss, axis=0)

        codes = self.xp.concatenate(codes, axis=0)
        quantization_loss = self.quantization_loss(out, codes)

        return triplet_loss, quantization_loss

    def get_coord(self, batch_size, feat_size):
        coord = self.xp.arange(feat_size).astype(self.xp.float32)
        y = coord.reshape(1, feat_size)
        y = self.xp.repeat(y, feat_size, axis=0)

        x = coord.reshape(feat_size, 1)
        x = self.xp.repeat(x, feat_size, axis=1)

        coord = self.xp.stack((y, x), axis=0)
        coord = 2 * (coord / (feat_size - 1)) - 1
        coord = self.xp.expand_dims(coord, 0)

        coord = self.xp.broadcast_to(coord, (batch_size, 2, feat_size, feat_size))

        return coord


class G_theta(chainer.Chain):
    def __init__(self, outsize):
        super(G_theta, self).__init__()
        init_l1 = initializers.Normal(0.001)
        init = initializers.Normal(0.001)
        with self.init_scope():
            self.h1 = L.Linear(outsize, initialW=None)
            self.h2 = L.Linear(outsize, initialW=None)
            self.h3 = L.Linear(outsize * 4, initialW=init)

    def __call__(self, x):
        h = F.relu(self.h1(x))
        h = F.relu(self.h2(h))
        out = self.h3(h)

        return out


class F_theta(chainer.Chain):
    def __init__(self, insize, output_dim):
        super(F_theta, self).__init__()
        init = initializers.Normal(0.0001)
        with self.init_scope():
            self.h1 = L.Linear(insize, initialW=None)
            self.h2 = L.Linear(insize, initialW=None)
            self.out = L.Linear(output_dim, initialW=None)

    def __call__(self, x):
        h1 = F.dropout(F.relu(self.h1(x)), ratio=0.5)
        h2 = F.dropout(F.relu(self.h2(h1)), ratio=0.5)
        out = self.out(h2)
        self.kl_loss = 0

        return out
