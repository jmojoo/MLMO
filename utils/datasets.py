from matplotlib import pyplot

import chainer
import sys
import random
import numpy as np
from chainer import datasets
from chainer.datasets import ConcatenatedDataset
# from chainer.datasets import ImageDataset
from utils.image_dataset import ImageDataset
import os

def get_data(dataset, gamma=0, include_train=True):
    queries = [[], []]
    database = [[], []]
    train = [[], []]
    with open('data/{}/data_dir.txt'.format(dataset), 'r') as f:
        img_dir = f.readline()

    # print("here")
    # for file in os.listdir(os.path.join(img_dir, 'images')):
    #     print(file)
    #     exit()

    print("reading query data...")
    with open('data/{}/test.txt'.format(dataset), 'r') as f:
        lines = f.readlines()
        for c, line in enumerate(lines):
            tokens = line.strip().split()
            label = np.array([int(i) for i in tokens[1:]], np.int32)
            full_path = os.path.join(img_dir, tokens[0])
            if os.path.exists(full_path):
                queries[0].append(full_path)
                queries[1].append(label)
            sys.stdout.write("processed: %d\r" % c)
            sys.stdout.flush()

    print("reading database data...")
    with open('data/{}/database.txt'.format(dataset), 'r') as f:
        lines = f.readlines()
        for c, line in enumerate(lines):
            tokens = line.strip().split()
            label = np.array([int(i) for i in tokens[1:]], np.int32)
            full_path = os.path.join(img_dir, tokens[0])
            if os.path.exists(full_path):
                database[0].append(full_path)
                database[1].append(label)
            else:
                print("{} not found".format(full_path))
                exit()
            sys.stdout.write("processed: %d\r" % c)
            sys.stdout.flush()

    if include_train:
        print("reading train data...")
        with open('data/{}/train_{}.txt'.format(dataset, gamma), 'r') as f:
            lines = f.readlines()
            for c, line in enumerate(lines):
                tokens = line.strip().split()
                label = np.array([float(i) for i in tokens[1:]], np.float32)
                full_path = os.path.join(img_dir, tokens[0])
                if os.path.exists(full_path):
                    train[0].append(full_path)
                    train[1].append(label)
                else:
                    print("{} not found".format(full_path))
                    exit()
                sys.stdout.write("processed: %d\r" % c)
                sys.stdout.flush()
        trainX = ImageDataset(train[0])
        trainY = np.array(train[1], dtype=np.float32)
    else:
        trainX = None
        trainY = None

    output = {'queries/img': ImageDataset(queries[0]),
              'queries/label': np.array(queries[1], dtype=np.int32),
              'train/img': trainX,
              'train/label': trainY,
              'database/img': ImageDataset(database[0]),
              'database/label': np.array(database[1], dtype=np.int32),
              }

    return output

def nuswide(gamma=0):
    return get_data('nuswide', gamma)

def nuswide_21():
    return get_data('nuswide_21')

def nuswide_21_full(gamma=0):
    return get_data('nuswide_21_full')

def coco(gamma=0):
    return get_data('coco', gamma)

def cifar10():
    return get_data('cifar10')

def holidays():
    return get_data('INRIA_holidays', include_train=False)

# def cifar10(num_query=1000, num_train=5000):
#     tr, te = datasets.get_cifar10()
#
#     q_classes = [0] * 10
#     t_classes = [0] * 10
#     d_classes = [0] * 10
#
#     mixed = ConcatenatedDataset(tr, te)
#
#     q_per_class = int(num_query / 10)
#     t_per_class = int(num_train / 10)
#     d_per_class = int((len(mixed) - (num_query + num_train)) / 10)
#
#     queries = [[], []]
#     database = [[], []]
#     train = [[], []]
#
#     N = len(mixed)
#     c = 1
#     one_hot_classes = np.eye(10, dtype=np.int32)
#     for img, label in mixed:
#         candidates = []
#         if q_classes[label] < q_per_class:
#             candidates.append((queries, q_classes))
#
#         if t_classes[label] < t_per_class:
#             candidates.append((train, t_classes))
#
#         if d_classes[label] < d_per_class:
#             candidates.append((database, d_classes))
#
#         chosen = random.choice(candidates)
#         chosen[0][0].append(img)
#         chosen[0][1].append(one_hot_classes[label])
#         chosen[1][label] += 1
#
#         sys.stdout.write("processed: %d/%d\r" % (c, N))
#         sys.stdout.flush()
#         c += 1
#
#     output = {'queries/img': np.array(queries[0], dtype=np.float32),
#               'queries/label': np.array(queries[1], dtype=np.int32),
#               'train/img': np.array(train[0], dtype=np.float32),
#               'train/label': np.array(train[1], dtype=np.int32),
#               'database/img': np.array(database[0], dtype=np.float32),
#               'database/label': np.array(database[1], dtype=np.int32),
#               }
#
#     return output

if __name__ == "__main__":
    coco()