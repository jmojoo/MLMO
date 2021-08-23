from train_net import Transform
import chainer
import numpy as np
import cupy

from models.dtq_rn import DTQ_RN
from models.dtq import DTQ
from models.dtq_rn_pyramid import DTQPyramidRN
from models.dtq_srl import DTQ_SRL
from models.extractors import AlexNet_conv5
from models.extractors import AlexNet_fc7
from models.extractors.vgg16 import VGG16Layers
from utils.retrieval import mAP, WAP, mAP_at
from utils import datasets
from utils.multiprocess_iterator import MultiprocessIterator
from labels import nuswide_labels

from chainer.datasets import TransformDataset
from chainer import cuda

import json
import os
import argparse
import math
import sys


class DictToObject:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


def evaluate(q_it, db_it, q_labels, db_labels,
             q_codes, d_codes, model1, R, config, db_embed=None, q_embed=None):

    xp = cupy
    sys.stdout.write("validating...\r")

    db_embed1 = []
    q_embed1 = []
    output_dim1 = config.output_dim
    half_dim = output_dim1 // 2
    print(output_dim1)

    total_batch = math.ceil(len(db_it.dataset) / db_it.batch_size)

    count = 0
    for batch in db_it:
        with chainer.using_config('train', False), \
             chainer.using_config('type_check', False):
            out1 = model1.get_embedding_test(batch)
        out1 = xp.mean(out1.reshape(-1, 10, output_dim1), axis=1)
        # print(out1.shape)
        # print(out1[0, :64])
        # exit()
        db_embed1.append(out1)

        count += 1
        sys.stdout.write(
            "computing database embeddings...%d/%d\r" % (count, total_batch))
        sys.stdout.flush()

    total_batch = math.ceil(len(q_it.dataset) / q_it.batch_size)
    count = 0
    for batch in q_it:
        with chainer.using_config('train', False):
            out1 = model1.get_embedding_test(batch)
        out1 = xp.mean(out1.reshape(-1, 10, output_dim1), axis=1)
        q_embed1.append(out1)

        count += 1
        sys.stdout.write(
            "computing query embeddings...%d/%d\r" % (count, total_batch))
        sys.stdout.flush()

    db_embed = cuda.to_cpu(xp.concatenate(db_embed1, axis=0))

    q_embed = xp.concatenate(q_embed1, axis=0)

    # db_embed = xp.mean(db_embed.reshape(-1, 10, output_dim1), axis=1)
    # q_embed = xp.mean(q_embed.reshape(-1, 10, output_dim1), axis=1)

    print("initializing centers for database embeddings...", end='\r')
    model1.initialize_centers(db_embed)
    print("updating database codes and centers...", end='\r')

    model1.update_codes_and_centers((db_embed, d_codes))
    model1.update_codes_batch((q_embed, q_codes), config.code_batch_n)

    db_embed = xp.array(db_embed, dtype=xp.float32)
    d_codes = d_codes
    db_labels = db_labels
    q_labels = q_labels
    q_codes = q_codes

    print("calculating mAP...", end='\r')


    if config.part == "rn":
        q_feat = q_embed[:, :half_dim]
        db_feat = db_embed[:, :half_dim]
    elif config.part == "fc":
        q_feat = q_embed[:, half_dim:]
        db_feat = db_embed[:, half_dim:]
    else:
        q_feat = q_embed
        db_feat = db_embed

        # q_feat[:, :half_dim] *= .5
        # db_feat[:, :half_dim] *= .5

    mAP_feature = mAP(q_feat, q_labels,
                      db_feat, db_labels, 'inner_product', R)

    mAP_05_feature = mAP_at(q_feat, q_labels,
                    db_feat, db_labels,
                    'inner_product', R, 0.5)

    WAP_feature = WAP(q_feat, q_labels,
                      db_feat, db_labels, 'inner_product', R)

    C = model1.C.array[:, :db_embed.shape[-1]]
    if config.stage == 1:
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

    if config.stage == 2:
        if config.part == 'rn':
            q_embed = q_embed[:, :half_dim]
            db_reconstr = db_reconstr[:, :half_dim]
        elif config.part == 'fc':
            q_embed = q_embed[:, half_dim:]
            db_reconstr = db_reconstr[:, half_dim:]

    mAP_AQD = mAP(q_embed, q_labels,
                  db_reconstr, db_labels,
                  'inner_product', R)

    mAP_05 = mAP_at(q_embed, q_labels,
                  db_reconstr, db_labels,
                  'inner_product', R, 0.5)

    WAP_AQD = WAP(q_embed, q_labels,
                  db_reconstr, db_labels,
                  'inner_product', R)

    print()
    print('mAP_feat: {}\n'
          'mAP_AQD: {}\n'
          'mAP@0.5: {}\n'
          'mAP@0.5_feat: {}\n'
          'WAP_feat: {}\n'
          'WAP_AQD: {}'.format(mAP_feature, mAP_AQD, mAP_05,
                               mAP_05_feature, WAP_feature, WAP_AQD))

    return db_reconstr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config1', type=str)
    parser.add_argument('--pretrained1', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--part', choices=('all', 'rn', 'fc'), default='all')
    parser.add_argument('--subspace', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    subspace = args.subspace
    with open(args.config1) as json_file:
        config1 = DictToObject(json.load(json_file))

    config1.part = args.part

    extractors = {'alexnet_conv5': AlexNet_conv5,
                  'alexnet_fc7': AlexNet_fc7,
                  'vgg16': VGG16Layers}
    models = {'dtq': DTQ, 'dtq_rn': DTQ_RN, 'dtq_prn': DTQPyramidRN, 'dtq_srl': DTQ_SRL}
    dset = {'nuswide': datasets.nuswide, 'nuswide_21': datasets.nuswide_21,
            'coco': datasets.coco, 'cifar10': datasets.cifar10,
            'holidays': datasets.holidays}
    extractor1 = extractors[config1.extractor]()

    model1 = models[config1.model](extractor1, config1)
    chainer.serializers.load_npz(args.pretrained1, model1, strict=True)

    config1.subspace = subspace
    model1.C.array = cupy.random.uniform(-1, 1,
                (config1.subspace * config1.subcenter, config1.output_dim)
            ).astype(cupy.float32)
    model1.C = model1.C[:config1.subspace * config1.subcenter]
    model1.subspace_n = subspace

    Rs = {'cifar10': 54000, 'nuswide': 5000, 'nuswide_21': 5000, 'coco': 5000}
    R = Rs[config1.dataset]

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model1.to_gpu()
        xp = cupy
    else:
        xp = np

    config1.dataset = args.dataset

    if config1.stage == 1:
        if config1.model != 'dtq':
            config1.output_dim = config1.output_dim - config1.rn_output_dim
        code_dim = config1.subspace * config1.subcenter
    else:
        code_dim = 2 * config1.subspace * config1.subcenter
    data = dset[config1.dataset]()

    query_img = TransformDataset(
        data['queries/img'],
        Transform(config1, ensemble=True)
    )
    query_label = xp.array(data['queries/label'], dtype=np.int32)
    query_codes = xp.zeros((len(query_label), code_dim), dtype=np.float32)

    database_img = TransformDataset(
        data['database/img'],
        Transform(config1, ensemble=True)
    )
    database_label = xp.array(data['database/label'], dtype=np.int32)
    database_codes = xp.zeros((len(database_label), code_dim), dtype=np.float32)

    query_iter = MultiprocessIterator(
        query_img, config1.val_batchsize, repeat=False, n_prefetch=2,
        shuffle=False)

    db_iter = MultiprocessIterator(
        database_img, config1.val_batchsize, repeat=False, n_prefetch=2,
        shuffle=False)

    db_embed = evaluate(query_iter, db_iter, query_label, database_label,
             query_codes, database_codes, model1, R, config1)
    db_embed = cuda.to_cpu(db_embed)
    savedir = "db_embed"
    mod_name = args.pretrained1.split("/")[1]
    mod_name = mod_name.split("_")[-1]
    np.save(os.path.join(savedir, "db_{}_{}_{}_{}_{}.npy".format(
        mod_name, config1.extractor, args.dataset, args.part, subspace)), db_embed)


if __name__ == '__main__':
    main()

