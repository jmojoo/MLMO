from chainer.iterators.order_samplers import OrderSampler
import sys
import numpy


class TripletOrderSampler(OrderSampler):

    def __init__(self, dataset, embeddings, labels, triplets, similarities):
        self.dataset = dataset
        self.triplets = triplets
        self.similarities = similarities
        self.embeddings = embeddings
        self.labels = labels

    def __call__(self, current_order, current_position):
        sys.stdout.write("updating triplets...\r")
        sys.stdout.flush()
        self.dataset.update_triplets(self.embeddings, self.labels,
                                     self.triplets, self.similarities)
        return None