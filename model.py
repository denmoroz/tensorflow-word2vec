# -*- coding: UTF-8 -*-

from __future__ import division, unicode_literals

import math
import tensorflow as tf


class Word2Vec(object):

    def __init__(self, optimizer, embedding_size, vocabulary_size,
                 batch_size, negative_samples):
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.batch_size = batch_size
        self.negative_samples = negative_samples

        self._init_graph()
        self._init_training(optimizer)
        self._init_monitoring()

    def _init_graph(self):
        self.model_graph = tf.Graph()

        with self.model_graph.as_default():
            # индексы входных слов
            self.input_words = tf.placeholder(
                dtype=tf.int32, shape=[self.batch_size],
                name='input_idx'
            )

            input_words_reshaped = tf.reshape(
                self.input_words,
                shape=[self.batch_size, 1]
            )

            # индексы выходных слов-контекстов
            # второе измерение неизвестно т.к. ширина окна - случайная величина
            self.real_contexts = tf.placeholder(
                dtype=tf.int32, shape=[self.batch_size, None],
                name='output_idx'
            )

            self.contexts_size = tf.size(self.real_contexts)

            # индексы шумовых слов-контекстов
            self.sampled_contexts = tf.placeholder(
                dtype=tf.int32, shape=[self.negative_samples],
                name='sampled_idx'
            )

            # шумовые индексы одни и те же для всего батча
            # тайлим их на размер батча
            sampled_contexts_tiled = tf.tile(
                tf.expand_dims(self.sampled_contexts, dim=0),
                multiples=[self.batch_size, 1]
            )

            init_width = 0.5 / self.embedding_size

            # матрица входных векторов слов
            self.input_matrix = tf.Variable(
                initial_value=tf.random_uniform(
                    shape=[self.vocabulary_size, self.embedding_size],
                    minval=-init_width, maxval=init_width
                ),
                name='input_vectors'
            )

            # матрица выходных векторов слов
            self.output_matrix = tf.Variable(
                initial_value=tf.truncated_normal(
                    shape=[self.vocabulary_size, self.embedding_size],
                    stddev=1.0 / math.sqrt(self.embedding_size)
                ),
                name='output_vectors'
            )

            # выбираем соответствующие вектора из матриц
            in_vectors = tf.nn.embedding_lookup(
                self.input_matrix, input_words_reshaped
            )

            real_out_vectors = tf.nn.embedding_lookup(
                self.output_matrix, self.real_contexts
            )

            sampled_out_vectors = tf.nn.embedding_lookup(
                self.output_matrix, sampled_contexts_tiled
            )

            # транспонируем входные вектора
            transposed_in_vectors = tf.transpose(
                in_vectors, perm=[0, 2, 1]
            )

            # батчем считаем скалярные произведения
            # между входными векторами и всеми выходными
            dot_products_1 = tf.reshape(
                tf.batch_matmul(
                    real_out_vectors,
                    transposed_in_vectors
                ), [-1]
            )

            dot_products_2 = tf.reshape(
                tf.batch_matmul(
                    sampled_out_vectors,
                    transposed_in_vectors
                ), [-1]
            )

            # Подсчитываем вероятности
            self.positives_proba = tf.nn.sigmoid(dot_products_1)
            self.negatives_proba = tf.nn.sigmoid(-1.0 * dot_products_2)

            proba_values = tf.concat(
                values=[self.positives_proba, self.negatives_proba],
                concat_dim=0, name='proba_values'
            )

            # избегаем проблем с log(0.0)
            clipped_proba = tf.clip_by_value(proba_values, 1e-10, 1.0)
            self.loss = -1.0 * tf.reduce_sum(tf.log(clipped_proba))

    def _init_training(self, optimizer):
        with self.model_graph.as_default():
            # счётчик обработанных батчей
            self.batches_processed = tf.Variable(
                initial_value=0, trainable=False
            )

            increment_batches = self.batches_processed.assign_add(1)

            # аккумулятор для среднего значения потерь
            self.average_loss = tf.Variable(
                initial_value=0.0, trainable=False
            )

            # рекуррентный пересчёт среднего значения функции потерь
            updated_loss = tf.truediv(
                tf.add(
                    self.average_loss * tf.to_float(self.batches_processed),
                    self.loss
                ),
                tf.to_float(self.batches_processed) + 1.0
            )
            update_average_loss = self.average_loss.assign(updated_loss)

            opt_op = optimizer.minimize(self.loss)

            # группируем операции оптимизации и обновления счётчиков в одну
            with tf.control_dependencies([opt_op]):
                self.train_op = tf.group(
                    update_average_loss, increment_batches
                )

    def _init_monitoring(self):
        with self.model_graph.as_default():
            self.context_size_summary = tf.histogram_summary(
                'Mini-batch contexts words count', self.contexts_size
            )

            self.positives_proba_summary = tf.histogram_summary(
                'P(D=1 | w,c_pos)', self.positives_proba
            )

            self.negatives_proba_summary = tf.histogram_summary(
                'P(D=0 | w,c_neg)', self.negatives_proba
            )

            self.loss_summary = tf.scalar_summary(
                'Mini-batch loss', self.loss
            )

            self.average_loss_summary = tf.scalar_summary(
                'Average training loss', self.average_loss
            )

            self.summary_op = tf.merge_all_summaries()
