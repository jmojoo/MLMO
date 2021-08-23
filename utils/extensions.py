import chainer
import math
from chainer import training
from chainer.iterators import _statemachine
import sys


class UpdateBatchSize(training.Extension):

    def __init__(self, iter, threshold=128):
        self.iter = iter
        self.threshold = threshold

    def __call__(self, trainer):
        log_report = trainer.get_extension('LogReport')
        stats = log_report.log[-1]
        n_triplets = stats['main/nt']

        if n_triplets < self.threshold:
            print("increasing batch size...")
            self.iter.batch_size += 10


class UpdateTrainData(training.Extension):

    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iter, model, train_img, train_embed, train_codes, train_labels,
                 update_triplets=True):
        self.iter = iter
        self.model = model
        self.train_img = train_img
        self.train_embed = train_embed
        self.train_codes = train_codes
        self.train_labels = train_labels
        self.update_triplets = update_triplets

    def __call__(self, trainer):
        # orig = chainer.global_config.train
        # chainer.global_config.train = False

        sys.stdout.write("updating embeddings...\r")
        with chainer.using_config('train', True):
            self.model.update_embedding(self.train_img, self.train_embed)

        sys.stdout.write("updating codes and centers...\r")
        self.model.update_codes_batch((self.train_embed, self.train_codes), 500)

        if self.update_triplets:
            sys.stdout.write("updating triplets...\r")
            sys.stdout.flush()
            self.iter.dataset.update_triplets(self.train_embed, self.train_labels)

        num_batches = int(math.ceil(len(self.iter.dataset)/self.iter.batch_size))

        report = {'total_iter': num_batches}
        observation = {}
        with chainer.reporter.report_scope(observation):
            trainer.reporter.report(report, self.model)
        # target.C.array = center_backup
        return observation


class UpdateTrainData2(training.Extension):

    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iter, model, train_img, train_embed, train_codes, train_labels,
                 triplets, similarities, update_triplets=True):
        self.iter = iter
        self.model = model
        self.train_img = train_img
        self.train_embed = train_embed
        self.train_codes = train_codes
        self.train_labels = train_labels
        self.triplets = triplets
        self.similarities = similarities
        self.update_triplets = update_triplets

    def __call__(self, trainer):
        # orig = chainer.global_config.train
        # chainer.global_config.train = False

        sys.stdout.write("updating embeddings...\r")
        with chainer.using_config('train', True):
            self.model.update_embedding(self.train_img, self.train_embed)

        sys.stdout.write("updating codes and centers...\r")
        self.model.update_codes_batch((self.train_embed, self.train_codes), 500)

        if self.update_triplets:
            sys.stdout.write("updating triplets...\r")
            sys.stdout.flush()
            self.iter.dataset.update_triplets(self.train_embed, self.train_labels,
                                              self.triplets, self.similarities)

        num_batches = int(math.ceil(len(self.iter.dataset)/self.iter.batch_size))

        report = {'total_iter': num_batches}
        observation = {}
        self.iter._state = _statemachine.IteratorState(
            0, self.iter.epoch, self.iter.is_new_epoch, self.iter._state.order)
        with chainer.reporter.report_scope(observation):
            trainer.reporter.report(report, self.model)
        # target.C.array = center_backup
        return observation


class LearningRateStep(training.Extension):

    def __init__(self, steps, scales):
        self.steps = steps
        self.scales = scales

    def initialize(self, trainer):
        iter = trainer.updater.iteration
        cumscale = 1
        for sp in self.steps:
            if sp < iter:
                idx = self.steps.index(sp)
                cumscale *= self.scales[idx]
        optimizer = trainer.updater.get_optimizer('main')
        for param in optimizer.target.params():
            param.update_rule.hyperparam.lr *= cumscale

    def __call__(self, trainer):
        # for name, param in trainer.updater.get_optimizer('main').target.namedparams():
        #     print(param.update_rule.hyperparam.lr, name)
        # exit()
        iter = trainer.updater.iteration
        if iter in self.steps:
            idx = self.steps.index(iter)
            scale = self.scales[idx]
            optimizer = trainer.updater.get_optimizer('main')
            for param in optimizer.target.params():
                param.update_rule.hyperparam.lr *= scale