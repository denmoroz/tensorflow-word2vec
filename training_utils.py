# -*- coding: UTF-8 -*-

from __future__ import division, unicode_literals

import random


def sliding_window(seq, window_size):
    for i in xrange(len(seq) - window_size + 1):
        yield seq[i:i + window_size]


def batches_generator(text, window_size, vocabulary,
                      batch_size, bootstrap_last):
    batch_data = []

    for tokens in sliding_window(text, 2 * window_size + 1):
        central_word = tokens[window_size]
        context_words = tokens[:window_size] + tokens[window_size + 1:]

        word_idx = vocabulary.get_or_unknown(central_word)

        contexts_idx = [
            vocabulary.get_or_unknown(word)
            for word in context_words
        ]

        examples = (word_idx, contexts_idx)
        batch_data.append(examples)

        if len(batch_data) == batch_size:
            yield batch_data
            batch_data = []

    if batch_data and bootstrap_last:
        diff = batch_size - len(batch_data)

        for _ in xrange(diff):
            element = random.choice(batch_data)
            batch_data.append(element)

        yield batch_data
