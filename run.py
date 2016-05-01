# -*- coding: UTF-8 -*-

from __future__ import division, unicode_literals

import time

import sys
from os import path
import Queue

import yaml
import logging
import argparse

import tensorflow as tf

from gensim.models.word2vec import LineSentence

from model import Word2Vec
from training import TrainingThread, ConsoleMonitoringThread
from vocabulary import Vocabulary

from training_utils import batches_generator
from sampling_utils import truncated_poisson, HierarchicalSampler
from io_utils import save_word2vec_format


logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser(
        description='''
            Word2Vec skip-gram modeling with negative sampling.
            Crafted with love by Dzianis Dus (dzianisdus@gmail.com).
        '''
    )

    parser.add_argument(
        'corpus', help='Path to the preprocessed corpus file'
    )

    parser.add_argument(
        'output', help='Path to directory to save vocabulary and vectors data'
    )

    parser.add_argument(
        '--config', default='config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--interval', type=int, default=5,
        help='Console monitoring interval in seconds'
    )

    parser.add_argument(
        '--binary', action='store_true',
        help='For saving vectors in binary format'
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    arguments = parse_args()

    logger.info('Loading config')
    with open(arguments.config) as config_file:
        config = yaml.load(config_file)

    logger.info('Initializing input stream')
    input_stream = LineSentence(
        arguments.corpus,
        max_sentence_length=config['sliding_window']['change_every_words']
    )

    min_word_freq = config['vocabulary']['min_freq']
    logger.info('Building vocabulary with min_freq={}'.format(min_word_freq))
    vocab = Vocabulary.from_documents(input_stream, min_word_freq)

    vocabulary_size = len(vocab)
    logger.info('Vocabulary size: {}'.format(vocabulary_size))

    logger.info('Building negative sampling distribution')
    negative_sampler = HierarchicalSampler(
        vocab=vocab,
        alpha=config['negative_sampling']['alpha'],
        chunks_num=config['negative_sampling']['vocab_chunks_num']
    )

    logger.info('Building model computation graph')
    optimizer = tf.train.AdagradOptimizer(
        learning_rate=config['training_params']['initial_learning_rate']
    )

    negative_samples_num = config['sliding_window']['max_size'] * \
        config['negative_sampling']['samples_num']

    word2vec = Word2Vec(
        optimizer=optimizer,
        embedding_size=config['embeddings']['size'],
        vocabulary_size=vocabulary_size,
        batch_size=config['training_params']['batch_size'],
        negative_samples=negative_samples_num
    )

    tf_threads_num = config['training_process']['tensorflow_threads']
    session_cfg = tf.ConfigProto(
        inter_op_parallelism_threads=tf_threads_num,
        intra_op_parallelism_threads=tf_threads_num,
        use_per_session_threads=True
    )

    with tf.Session(graph=word2vec.model_graph, config=session_cfg) as session:
        workers_num = config['training_process']['workers_num']
        queue_size = workers_num * config['training_process']['queue_factor']
        task_queue = Queue.Queue(maxsize=queue_size)

        logger.info('Initializing model params')
        session.run(tf.initialize_all_variables())

        logger.info('Initializing workers')
        if config['training_process']['tensorboard_monitoring']:
            summary_writer = tf.train.SummaryWriter(
                logdir=config['training_process']['tensorboard_logdir'],
                graph=session.graph,
                max_queue=16
            )
        else:
            summary_writer = None

        workers = []
        for _ in xrange(workers_num):
            worker = TrainingThread(
                batch_size=config['training_params']['batch_size'],
                model=word2vec,
                session=session,
                task_queue=task_queue,
                summary_writer=summary_writer
            )

            workers.append(worker)
            worker.start()

        logging.info('Initializing console monitoring')
        monitoring_thread = ConsoleMonitoringThread(
            logger=logger, check_interval=arguments.interval,
            model=word2vec, session=session
        )
        monitoring_thread.start()

        logging.info('Starting training')
        batch_num = 0
        training_started = time.time()

        for epoch_num in xrange(config['training_params']['epochs']):
            for text in input_stream:

                window_size = truncated_poisson(
                    min_value=config['sliding_window']['min_size'],
                    mean_value=config['sliding_window']['avg_size'],
                    max_value=config['sliding_window']['max_size']
                )

                training_batches = batches_generator(
                    text, window_size, vocab,
                    config['training_params']['batch_size'],
                    config['training_process']['bootstrap_last_batch']
                )

                for mini_batch in training_batches:
                    noise_idx = negative_sampler(negative_samples_num)

                    task_queue.put((batch_num, mini_batch, noise_idx))
                    batch_num += 1

        for worker in workers:
            task_queue.put(None)

        for worker in workers:
            worker.join()

        if config['training_process']['tensorboard_monitoring']:
            summary_writer.flush()
            summary_writer.close()

        monitoring_thread.stop_monitoring()

        logging.info('Saving model')
        W_matrix, U_matrix = session.run(
            [word2vec.input_matrix, word2vec.output_matrix]
        )

        save_word2vec_format(
            path.join(arguments.output, 'vocab.txt'),
            path.join(arguments.output, 'input_vectors.txt'),
            path.join(arguments.output, 'output_vectors.txt'),
            vocab,
            W_matrix,
            U_matrix,
            binary=arguments.binary
        )

        logging.info(
            'Done in {} seconds'.format(time.time() - training_started)
        )
