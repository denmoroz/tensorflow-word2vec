# -*- coding: UTF-8 -*-

from __future__ import division, unicode_literals

from collections import Iterable
import tensorflow as tf

import codecs


def _int64_feature(value):
    if not isinstance(value, Iterable):
        value = [value]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_example(word_idx, context_idx, noise_idx):
    example = tf.train.Example(features=tf.train.Features(
            feature={
                'word_idx': _int64_feature(word_idx),
                'context_idx': _int64_feature(context_idx),
                'noise_idx': _int64_feature(noise_idx)
            }
        )
    )

    return example


def create_examples_batch(window_size, noise_idx, words_idx, contexts_idx):
    words_flatten = words_idx.reshape(-1)
    contexts_flatten = contexts_idx.reshape(-1)

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                # ширина окна - понадобится для восстановления
                'window_size': _int64_feature(window_size),
                # одинаковые для всех в минибатче
                'noise': _int64_feature(noise_idx),
                # reshaped индексы слов
                'words': _int64_feature(words_flatten),
                # reshaped индексы контекстов
                'contexts': _int64_feature(contexts_flatten),
            }
        )
    )

    return example


def save_word2vec_format(vocab_file, in_vectors_file, out_vectors_file,
                         vocabulary, input_vectors, output_vectors,
                         binary=False):

    with codecs.open(vocab_file, 'w', encoding='utf-8') as v_f:
        # итерирование происходит в порядке убывания частот слов
        for word in vocabulary.words:
            v_f.write("{} {}\n".format(word, vocabulary.frequency(word)))

    with codecs.open(in_vectors_file, 'w', encoding='utf-8') as in_f:
        with codecs.open(out_vectors_file, 'w', encoding='utf-8') as out_f:
            in_f.write("%s %s\n" % input_vectors.shape)
            out_f.write("%s %s\n" % output_vectors.shape)

            for word in vocabulary.words:
                row_index = vocabulary[word]

                input_row = input_vectors[row_index]
                output_row = output_vectors[row_index]

                if binary:
                    in_f.write(word + b" " + input_row.tostring())
                    out_f.write(word + b" " + output_row.tostring())
                else:
                    row_string = ' '.join("%f" % val for val in input_row)
                    in_f.write("%s %s\n" % (word, row_string))

                    row_string = ' '.join("%f" % val for val in output_row)
                    out_f.write("%s %s\n" % (word, row_string))
