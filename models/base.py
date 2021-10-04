import chainer
from chainer import links as L, initializers
from chainer import functions as F
from chainer import cuda
from sklearn.cluster import MiniBatchKMeans
from math import ceil
import numpy as np
import random


class Base(chainer.Chain):
    def __init__(self, args):
        super(Base, self).__init__()
        self.margin = args.margin
        self.output_dim = args.output_dim
        self.subspace_n = args.subspace
        self.subcenter_n = args.subcenter
        self.hard_feat = args.hard_feat
        self.max_iter_b = args.max_iter_b
        self.max_iter_Cb = args.max_iter_Cb
        self.code_batch_n = args.code_batch_n
        self.val_batchsize = args.val_batchsize
        self.batchsize = args.train_batchsize * 3
        self.stage = args.stage
        self.check_hard = True
        self.val = False

    def get_embedding_train(self, x):
        raise NotImplementedError

    def get_embedding_test(self, x):
        raise NotImplementedError

    def initialize_centers(self, img_output):
        if self.stage == 1:
            img_outputs = [img_output]
        else:
            img_outputs = np.split(img_output, 2, axis=1)

        C_init = []
        for img_output in img_outputs:
            all_output = cuda.to_cpu(img_output)
            output_dim = img_output.shape[-1]
            C_ = np.zeros([self.subspace_n * self.subcenter_n, img_output.shape[-1]])
            for i in range(self.subspace_n):
                start = i * int(output_dim / self.subspace_n)
                end = (i + 1) * int(output_dim / self.subspace_n)
                to_fit = all_output[:, start:end]
                kmeans = MiniBatchKMeans(n_clusters=self.subcenter_n).fit(to_fit)
                centers = kmeans.cluster_centers_
                C_[i * self.subcenter_n: (i + 1) * self.subcenter_n, start:end] = centers
            C_init.append(C_)
        if self.stage == 1:
            C_init = C_init[0]
        else:
            C_init = np.concatenate(C_init, axis=1)

        self.C.array[:, :C_init.shape[-1]] = self.xp.array(C_init, dtype=self.xp.float32)

    def update_centers(self, datasets):
        if self.stage == 1:
            embeddings, codes = datasets
            embeddings = [self.xp.array(embeddings)]
            codes = [self.xp.array(codes)]
        else:
            embeddings = self.xp.split(self.xp.array(datasets[0]), 2, axis=1)
            codes = self.xp.split(self.xp.array(datasets[1]), 2, axis=1)

        i = 0
        for U, h in zip(embeddings, codes):
            curr_dim = U.shape[-1]
            # h = self.xp.array(datasets[1])
            # U = self.xp.array(datasets[0])
            smallResidual = self.xp.eye(self.subcenter_n * self.subspace_n,
                                     dtype=self.xp.float32) * 0.001
            Uh = self.xp.matmul(self.xp.transpose(h), U)
            hh = self.xp.add(self.xp.matmul(self.xp.transpose(h), h), smallResidual)
            centers = self.xp.matmul(self.xp.linalg.inv(hh), Uh)
            C_sums = self.xp.sum(self.xp.square(centers), axis=1)
            C_non_zeros_ids = self.xp.where(C_sums >= 1e-8)[0]
            self.C.array[C_non_zeros_ids, i:i + curr_dim] = centers[C_non_zeros_ids, :]
            i += curr_dim

    def update_codes(self, output, code):
        codes = []
        if self.stage == 1:
            outputs = [output]
            codes_ = [code]
            Cs = [self.C.array]
        else:
            outputs = self.xp.split(output, 2, axis=1)
            codes_ = self.xp.split(code, 2, axis=1)
            Cs = self.xp.split(self.C.array, 2, axis=1)

        comp = ['fc', 'fc']
        for output, code, C, comp in zip(outputs, codes_, Cs, comp):
            code = self.xp.zeros(code.shape, dtype=self.xp.float32)
            out_dim = output.shape[-1]
            for iterate in range(self.max_iter_b):
                sub_list = list(range(self.subspace_n))
                random.shuffle(sub_list)
                for m in sub_list:
                    ICM_b_m = code[:, m * self.subcenter_n: (m + 1) * self.subcenter_n]
                    start = m * self.subcenter_n
                    ICM_C_m = C[start: start + self.subcenter_n, :out_dim]
                    residual = output - self.xp.matmul(code, C[:, :out_dim]) + self.xp.matmul(
                        ICM_b_m, ICM_C_m)
                    res_expand = self.xp.expand_dims(residual, 1)
                    ICM_m_expand = self.xp.expand_dims(ICM_C_m, 0)
                    ICM_loss = self.xp.sum(self.xp.square(res_expand - ICM_m_expand), axis=2)

                    if comp == 'rn':
                        best_center_idx = self.xp.argsort(ICM_loss, axis=1)[:, :3]
                        dim0 = self.xp.arange(best_center_idx.shape[0]).reshape(best_center_idx.shape[0], 1)
                        best_centers = self.xp.zeros((best_center_idx.shape[0], self.subcenter_n))
                        best_centers[dim0, best_center_idx] = 1
                        code[:, m * self.subcenter_n: (m + 1) * self.subcenter_n] = best_centers
                    else:
                        best_centers = self.xp.argmin(ICM_loss, axis=1)
                        best_centers_one_hot = self.xp.eye(self.subcenter_n)[best_centers]
                        code[:, m * self.subcenter_n: (m + 1) * self.subcenter_n] = best_centers_one_hot
            codes.append(code)
        if self.stage == 1:
            code = codes[0]
        else:
            code = self.xp.concatenate(codes, axis=1)
        return code

    def update_codes_batch(self, datasets, batchsize):
        num_batch = int(ceil(len(datasets[1]) / batchsize))
        is_embedding_on_gpu = isinstance(datasets[0], self.xp.ndarray)
        is_code_on_gpu = isinstance(datasets[1], self.xp.ndarray)
        for i in range(num_batch):
            # embedding, codes = zip(*batch)
            # embedding = self.xp.array(embedding, dtype=self.xp.float32)
            start = i * batchsize
            end = start + batchsize
            embedding = datasets[0][start:end]
            if not is_embedding_on_gpu:
                embedding = self.xp.array(embedding, dtype=self.xp.float32)
            codes = datasets[1][start:end]
            if not is_code_on_gpu:
                codes = self.xp.array(codes, dtype=self.xp.float32)

            new_codes = self.update_codes(embedding, codes)
            if not is_code_on_gpu:
                new_codes = cuda.to_cpu(new_codes)
            datasets[1][start:end] = new_codes

    def update_codes_and_centers(self, datasets):
        for i in range(self.max_iter_Cb):
            self.update_codes_batch(datasets, self.code_batch_n)
            self.update_centers(datasets)

    def update_embedding(self, imgs, embeddings):
        # iterator = chainer.iterators.MultithreadIterator(
        #     imgs, self.batchsize, repeat=False, shuffle=False
        # )
        num_batch = int(ceil(len(imgs) / self.batchsize))
        is_embedding_on_gpu = isinstance(embeddings, self.xp.ndarray)
        for i in range(num_batch):
            # size = len(X)
            start = i * self.batchsize
            end = start + self.batchsize
            X = imgs[start: end]
            new_embedding = self.get_embedding_test(X)
            if not is_embedding_on_gpu:
                new_embedding = cuda.to_cpu(new_embedding)
            embeddings[start:end] = new_embedding

    def quantization_loss(self, embeddings, codes):
        if self.stage == 1:
            e = [embeddings]
            c = [codes]
            Cs = [self.C.array]
        else:
            e = F.split_axis(embeddings, 2, axis=1)
            c = self.xp.split(codes, 2, axis=1)
            Cs = self.xp.split(self.C.array, 2, axis=1)
        q_loss = 0
        for embeddings, codes, C in zip(e, c, Cs):
            l = F.sum(
                F.square(embeddings - F.matmul(codes, C[:, :embeddings.shape[-1]]
                )), -1
            )
            q_loss += F.mean(l)
        return q_loss
