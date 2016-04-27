# -*- coding: UTF-8 -*-

from __future__ import division, unicode_literals

import math
import numpy as np

import numba


def chunks_iter(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def truncated_poisson(mean_value, min_value, max_value=None):
    while True:
        value = np.random.poisson(lam=mean_value)

        if value >= min_value:
            if max_value is None:
                break
            elif value <= max_value:
                break

    return value


def build_sampling_distribution(vocab, alpha, chunks_num):
    chunk_size = int(math.ceil(len(vocab) / chunks_num))

    # константа нормализации для распределения
    total_count_pow = 0
    for word in vocab.words:
        word_count = vocab.frequency(word)
        total_count_pow += math.pow(word_count, alpha)

    pairs = []
    for word in vocab.words:
        word_idx = vocab[word]
        word_count = vocab.frequency(word)
        # вероятность сэмплирования слова по Mikolov'у
        word_proba = math.pow(word_count, alpha) / total_count_pow

        pairs.append((word_idx, word_proba))

    # слова, отсортированные по убыванию вероятности
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

    # делим на заданное количество чанков
    # и подсчитываем вероятность каждого из чанков
    chunk_indices, chunk_proba = [], []
    for chunk_data in chunks_iter(pairs, chunk_size):
        indices, probabilities = zip(*chunk_data)

        chunk_indices.append(indices)
        chunk_proba.append(np.sum(probabilities))

    return np.array(chunk_indices), np.array(chunk_proba)


@numba.autojit
def sample_words(size, chunks_indices, accumulated_proba):
    result = np.zeros(shape=size, dtype=np.int64)

    random_sample = np.random.random_sample(size)
    chunks_samples = np.digitize(random_sample, accumulated_proba)

    for i in xrange(size):
        chunk_idx = chunks_samples[i]

        chunk_words = chunks_indices[chunk_idx]
        word_idx = np.random.randint(0, len(chunk_words) - 1)

        result[i] = chunk_words[word_idx]

    return result

# Особенности HierarchicalSampler:
# 1. при chunks_num = |V| => превращается в сэмплер Word2Vec: p(w) ^ 0.75 / Z
# 2. при chunks_num = 1 => превращается в равномерный сэмплер по словарю
# 3. при промежуточных значениях сочетает работу обоих сэмплеров.
#    при этом даунсэмплинг самых популярных слов, наверное, не нужен т.к.
#    внутри каждого чанка слов сэмплирование происходит равномерно.


class HierarchicalSampler(object):

    def __init__(self, vocab, alpha, chunks_num):
        self.chunks_indices, chunks_proba = build_sampling_distribution(
            vocab, alpha, chunks_num
        )

        self._accumulated_proba = np.add.accumulate(chunks_proba)

    def __call__(self, samples):
        return sample_words(
            size=samples,
            chunks_indices=self.chunks_indices,
            accumulated_proba=self._accumulated_proba
        )


class UniformSampler(object):

    def __init__(self, vocabulary_size):
        self.vocabulary_size = vocabulary_size

    def __call__(self, samples):
        return np.random.randint(
            low=0, high=self.vocabulary_size - 1, size=samples
        )
