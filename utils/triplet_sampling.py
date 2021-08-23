from chainer.dataset import DatasetMixin
from utils.retrieval import distance
import math
import numpy as np
from chainer import cuda


class TripletDataset(DatasetMixin):
    def __init__(self, train_img, train_embed, train_codes, args):
        super(TripletDataset, self).__init__()
        self.xp = np
        self.g = args.num_groups
        self.min_triplets = args.min_triplets
        self.stage = args.stage
        self.dist = []
        self.margin = args.margin
        self.triplets = []
        self.similarity = []
        self.dataset = train_img
        self.codes = train_codes
        self.n_samples = len(train_embed)
        thresh = 0.5
        self.thresh = thresh
        self._perm = self.xp.arange(self.n_samples).astype(self.xp.int32)
        np.random.shuffle(self._perm)

    def __len__(self):
        return len(self.triplets)

    def get_example(self, i):
        idx = self.triplets[i]
        idx = [self._perm[a] for a in idx]
        triplet = (self.dataset[idx[0]], self.dataset[idx[1]], self.dataset[idx[2]],
                   self.codes[idx[0]], self.codes[idx[1]], self.codes[idx[2]],
                   idx[0], idx[1], idx[2], self.similarity[i])
        return triplet

    def update_triplets(self, output, lbl):
        """
        :param output: train image embeddings
        :param lbl: train image labels
        """
        if not isinstance(output, np.ndarray):
            output = cuda.to_cpu(output)

        n_part = self.g
        n_samples = self.n_samples
        np.random.shuffle(self._perm)
        embedding = output[self._perm[:n_samples]]
        labels = lbl[self._perm[:n_samples]].astype(np.int32)
        n_samples_per_part = int(math.ceil(n_samples / n_part))

        similarity = []
        triplets = []

        dist = None
        for i in range(n_part):
            start = n_samples_per_part * i
            end = min(n_samples_per_part * (i + 1), n_samples)
            dist = distance(self.xp, embedding[start:end], pair=True, dist_func='euclid2')

            for idx_anchor in range(0, end - start):
                label_anchor = np.copy(labels[idx_anchor + start, :])
                lbl_g = labels[start:end]

                intersection = np.sum(np.minimum(lbl_g, label_anchor), axis=1)
                union = np.sum(np.maximum(lbl_g, label_anchor), axis=1)
                iou = intersection / (union + 1e-7)

                all_pos = np.where(iou > self.thresh)[0]
                all_neg = np.where(iou == 0)[0]
                all_pos_iou = iou[all_pos]

                for idx_pos, pos_iou in zip(all_pos, all_pos_iou):
                    if idx_pos == idx_anchor:
                        continue

                    cond = dist[idx_anchor, all_neg] - dist[idx_anchor, idx_pos] < pos_iou * self.margin
                    selected_neg = all_neg[np.where(cond)[0]]

                    if selected_neg.shape[0] > 0:
                        idx_neg = np.random.choice(selected_neg)
                        triplets.append(
                            (idx_anchor + start, idx_pos + start, idx_neg + start))
                        similarity.append((pos_iou - iou[idx_neg]).astype(np.float32))

        if len(triplets) < self.min_triplets and self.g > 1:
            print(
                "Number of triplets below {}. Halving number of groups to {}".format(
                    self.min_triplets, self.g // 2))
            self.g //= 2
            del similarity
            del triplets
            del dist
            self.update_triplets(output, lbl)
            return

        self.triplets = np.array(triplets)
        self.similarity = np.array(similarity)

        perm = np.arange(len(self.triplets))
        np.random.shuffle(perm)

        # assert
        labels = np.sign(labels)
        anchor = labels[self.triplets[:, 0]]
        mapper = lambda anchor, other: np.any(anchor * (anchor == other), -1)
        assert (np.all(mapper(anchor, labels[self.triplets[:, 1]])))
        assert (np.all(np.invert(anchor, labels[self.triplets[:, 2]])))
        return


