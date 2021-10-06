from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import chainer
from chainer.datasets import TransformDataset
from chainer.optimizer_hooks import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions
from utils.retrieval import RetrievalEvaluator
from utils.triplet_sampling import TripletDataset
from utils.momentum_sgd import MomentumSGD
from utils.multiprocess_iterator import MultiprocessIterator

import json

from models.mlmo import MLMO
from models.extractors import AlexNet
from models.extractors import vgg16
from utils import datasets
from utils.extensions import UpdateTrainData
from utils.extensions import LearningRateStep
from utils.image_dataset import Transform
import cupy

import os
import sys


class TripletTrainChain(chainer.Chain):
    def __init__(self, model, q_lambda):
        super(TripletTrainChain, self).__init__()
        self.train_embed = None
        with self.init_scope():
            self.model = model
            self.q_lambda = q_lambda

    def __call__(self, a, p, n, a_b, p_b, n_b,
                 a_ind, p_ind, n_ind, sim):
        with chainer.using_config('type_check', False), \
             chainer.using_config('use_cudnn', 'auto'):
            r_loss, q_loss = self.model(a, p, n, (a_b, p_b, n_b),
                                      (a_ind, p_ind, n_ind), self.train_embed, sim)

            chainer.reporter.report(
                {'rank_loss': r_loss, 'q_loss': q_loss}, self)
            return r_loss + self.q_lambda * q_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-batchsize', type=int, default=128)
    parser.add_argument('--val-batchsize', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=80)
    parser.add_argument('--steps', type=int, nargs='*', default=[30000, 60000])
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--model', choices=('mlmo',), default='mlmo')
    parser.add_argument('--out', default=None)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--extractor', choices=('alexnet', 'vgg16'))
    parser.add_argument('--dataset', choices=('cifar10','nuswide', 'coco', 'nuswide_21'))
    parser.add_argument('--trainstage', choices=('val', 'final'), default='final')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--margin', type=float, default=30)
    parser.add_argument('--q_lambda', type=float, default=0.001)
    parser.add_argument('--output-dim', type=int, default=128)
    parser.add_argument('--rn-output-dim', type=int, default=64)
    parser.add_argument('--stage', type=int, choices=(1, 2), default=1)
    parser.add_argument('--rn-alpha', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--subspace', type=int, default=2)
    parser.add_argument('--subcenter', type=int, default=256)
    parser.add_argument('--hard-feat', action='store_true')
    parser.add_argument('--max-iter-b', default=3, type=int)
    parser.add_argument('--max-iter-Cb', default=2, type=int)
    parser.add_argument('--code_batch_n', default=500, type=int)
    parser.add_argument('--min-triplets', default=60000, type=int)
    parser.add_argument('--num-groups', default=50, type=int)
    parser.add_argument('--coord_scale', default=50, type=int)
    parser.add_argument('--weight-decay', default=0.0005, type=float)
    args = parser.parse_args()

    extractors = {'alexnet': AlexNet,
                  'vgg16': vgg16}
    models = {'mlmo': MLMO}
    dset = {'nuswide': datasets.nuswide, 'nuswide_21': datasets.nuswide_21, 'coco': datasets.coco, }
    Rs = {'cifar10': 54000, 'nuswide': 5000, 'nuswide_21': 5000, 'coco': 5000}
    # nuswide 21 is the version of NUSWIDE with oly 21 most common classes
    args.R = Rs[args.dataset]
    extractor = extractors[args.extractor]()
    if args.pretrained is not None:
        chainer.serializers.load_npz(args.pretrained, extractor, strict=False)

    model = models[args.model](extractor, args)

    if args.out is None:
        args.out = os.path.join('results', args.model, args.dataset, str(args.gamma))

    train_chain = TripletTrainChain(model, args.q_lambda)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        xp = cupy
    else:
        xp = np

    optimizer = MomentumSGD(lr=args.lr)
    optimizer.setup(train_chain)
    if args.weight_decay > 0:
        optimizer.add_hook(WeightDecay(args.weight_decay))

    if args.stage == 1:
        if args.model == 'mlmo':
            embed_dim = args.output_dim - args.rn_output_dim
        else:
            embed_dim = args.output_dim
        code_dim = args.subspace * args.subcenter
    else:
        code_dim = 2 * args.subspace * args.subcenter
        embed_dim = args.output_dim

    data = dset[args.dataset]()

    if args.trainstage == 'val':
        # split train data into train and val
        ridx = np.random.permutation(len(data['train/label']))
        train_idx = ridx[:9000]
        val_idx = ridx[9000:]

        train_img = TransformDataset(
            data['train/img'][train_idx],
            Transform(args, augment=True)
        )
        train_label = data['train/label'][train_idx]

        query_img = TransformDataset(
            data['train/img'][val_idx],
            Transform(args, ensemble=True)
        )
        query_label = xp.array(data['train/label'][val_idx], dtype=np.int32)

        database_img = query_img
        database_label = query_label
    else:
        train_img = TransformDataset(
            data['train/img'],
            Transform(args, augment=True)
        )
        train_label = data['train/label']

        query_img = TransformDataset(
            data['queries/img'],
            Transform(args, ensemble=True)
        )
        query_label = xp.array(data['queries/label'], dtype=np.int32)

        database_img = TransformDataset(
            data['database/img'],
            Transform(args, ensemble=True)
        )
        database_label = xp.array(data['database/label'], dtype=np.int32)

    train_codes = np.zeros((len(train_label), code_dim), dtype=np.float32)
    train_embed = np.zeros((len(train_label), embed_dim), dtype=np.float32)

    train_chain.train_embed = train_embed

    query_codes = xp.zeros((len(query_label), code_dim), dtype=np.float32)
    database_codes = xp.zeros((len(database_label), code_dim), dtype=np.float32)

    train = TripletDataset(train_img, train_embed, train_codes, args)

    train_iter = MultiprocessIterator(
        train, args.train_batchsize, shuffle=False, shared_mem=3859140,
        maxtasksperchild=args.train_batchsize, n_prefetch=4)

    query_iter = MultiprocessIterator(
        query_img, args.val_batchsize, repeat=False, n_prefetch=2,
        shuffle=False)

    db_iter = MultiprocessIterator(
        database_img, args.val_batchsize, repeat=False, n_prefetch=2,
        shuffle=False)

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(
        updater, (args.epoch, 'epoch'), args.out)

    scales = [0.1] * len(args.steps)
    trainer.extend(LearningRateStep(args.steps, scales),
                  trigger=(1, 'iteration'))

    trainer.extend(
        extensions.snapshot(),
        trigger=(10, 'epoch'))
    trainer.extend(
        extensions.snapshot_object(model, 'model_epoch_{.updater.epoch}'),
        trigger=(80, 'epoch'))

    trainer.extend(
        RetrievalEvaluator(query_iter, query_codes, query_label,
                           db_iter, database_codes, database_label,
                           model, args, device=args.gpu),
        trigger=(40, 'epoch'))

    trainer.extend(
        UpdateTrainData(train_iter, model, train_img,
                        train_embed, train_codes, train_label, train),
        trigger=(1, 'epoch')
    )

    log_interval = 1, 'epoch'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'lr',
         'main/rank_loss', 'main/q_loss', 'val/main/mAP_feat',
         'val/main/mAP_AQD', 'val/main/mAP@.5feat', 'val/main/mAP@0.5']),
        trigger=log_interval)

    trainer.extend(extensions.PlotReport(
        ('main/triplet_loss',),
        filename='loss.png', marker=None, trigger=(1000, 'iteration')))

    trainer.extend(extensions.ProgressBar(update_interval=1))

    if args.resume:
        serializers.load_npz(args.resume, trainer, strict=False)

    sys.stdout.write("updating embeddings...\r")
    with chainer.using_config('train', True):
        model.update_embedding(train_img, train_embed)
    sys.stdout.write("initializing centers...\r")
    model.initialize_centers(train_embed)
    sys.stdout.write("updating codes and centers...\r")
    model.update_codes_and_centers((train_embed, train_codes))
    sys.stdout.write("creating triplets...\r")
    sys.stdout.flush()
    train.update_triplets(train_embed, train_label)

    for name, param in model.namedparams():
        if "out" in name or 'f_' in name or 'g_' in name:
            param.update_rule.hyperparam.lr *= 10
        if name[-1] == 'b':
            param.update_rule.hyperparam.lr *= 2

    config = vars(args)
    config_json = json.dumps(config, indent=4)
    path = args.out
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, 'config.json'), 'w') as f:
        print(config_json, file=f)

    model.cleargrads()
    trainer.run()


if __name__ == '__main__':
    main()