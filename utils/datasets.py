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
        with open('data/{}/train.txt'.format(dataset), 'r') as f:
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

def coco(gamma=0):
    return get_data('coco', gamma)

def cifar10():
    return get_data('cifar10')

def holidays():
    return get_data('INRIA_holidays', include_train=False)

if __name__ == "__main__":
    coco()