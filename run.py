# -*- coding: UTF-8 -*-

from __future__ import division, unicode_literals
import random
import Queue

import tensorflow as tf

from gensim.models.word2vec import Text8Corpus
# from gensim.models.word2vec import LineSentence

from model import Word2Vec
from training import TrainingThread
from vocabulary import Vocabulary

from sampling_utils import truncated_poisson
from sampling_utils import HierarchicalSampler
from io_utils import save_word2vec_format


with_monitoring = False
workers_num = 10
queue_factor = 4

# Ширина контекста - усечённая пуассоновская случайная величина
min_context_size = 1
mean_context_size = 4
max_context_size = 8

epochs = 3
batch_size = 16
initial_learning_rate = 0.1
noise_samples = 10
embedding_size = 300


# Документы длинной более 500 слов будут нарезаться на части по 500 слов:
# 1. каждые 500 слов случайным образом меняется ширина окна
# 2. каждые batch_size окон генерируются новые негативные сэмплы
#    их max_context_size * noise_samples штук
#    они остаются фиксированными для текущего батча
# 3. если последний батч неполный, то он или
#    отбрасывается или бутстрэпится до полного

LOG_DIR = 'logs'
CORPUS_FILE = '../test_corpus/text8'
input_stream = Text8Corpus(
    fname=CORPUS_FILE,
    max_sentence_length=500
)

# CORPUS_FILE = '../test_corpus/text8_without_stop'
# input_stream = LineSentence(CORPUS_FILE)


def sliding_window(seq, window_size):
    for i in xrange(len(seq) - window_size + 1):
        yield seq[i:i + window_size]


def training_batches_iterator(text, window_size, vocabulary,
                              batch_size, bootstrap_last=False):
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

    # если флаг bootstrap_last установлен в:
    # 1. False: неполные батчи отбрасываются
    # 2. True: бутстреппингом добираются до полного батча
    if batch_data and bootstrap_last:
        diff = batch_size - len(batch_data)

        for _ in xrange(diff):
            element = random.choice(batch_data)
            batch_data.append(element)

        yield batch_data


print 'Building vocabulary'
vocab = Vocabulary.from_documents(input_stream, min_freq=3)

vocabulary_size = len(vocab)
print 'Vocabulary size: {}'.format(vocabulary_size)

print 'Building sampling distribution'
negative_sampler = HierarchicalSampler(
    vocab=vocab, alpha=0.75, chunks_num=1000
)

print 'Building model'
word2vec = Word2Vec(
    #optimizer=tf.train.RMSPropOptimizer(learning_rate=initial_learning_rate),
    optimizer=tf.train.AdagradOptimizer(learning_rate=initial_learning_rate),
    embedding_size=embedding_size,
    vocabulary_size=vocabulary_size,
    batch_size=batch_size,
    negative_samples=max_context_size * noise_samples
)

session_config = tf.ConfigProto(
    inter_op_parallelism_threads=5,
    intra_op_parallelism_threads=5,
    use_per_session_threads=True
)


with tf.Session(graph=word2vec.model_graph, config=session_config) as session:
    workers = []
    task_queue = Queue.Queue(maxsize=workers_num * queue_factor)

    print 'Initializing params'
    session.run(tf.initialize_all_variables())

    print 'Initializing workers'
    if with_monitoring:
        summary_writer = tf.train.SummaryWriter(
            logdir=LOG_DIR, graph=session.graph, max_queue=16
        )
    else:
        summary_writer = None

    for _ in xrange(workers_num):
        worker = TrainingThread(
            batch_size=batch_size,
            model=word2vec,
            session=session,
            task_queue=task_queue,
            summary_writer=summary_writer
        )

        workers.append(worker)
        worker.start()

    print 'Training model'
    for epoch_num in xrange(1, epochs + 1):
        for text_num, text in enumerate(input_stream, 1):

            window_size = truncated_poisson(
                mean_value=mean_context_size,
                min_value=min_context_size,
                max_value=max_context_size
            )

            training_batches = training_batches_iterator(
                text, window_size, vocab, batch_size,
                bootstrap_last=True
            )

            for mini_batch in training_batches:
                noise_idx = negative_sampler(max_context_size * noise_samples)
                task_queue.put((epoch_num, text_num, mini_batch, noise_idx))

    for worker in workers:
        task_queue.put(None)

    for worker in workers:
        worker.join()

    if with_monitoring:
        summary_writer.flush()
        summary_writer.close()

    print 'Saving model'
    W_matrix, U_matrix = session.run(
        [word2vec.input_matrix, word2vec.output_matrix]
    )

    save_word2vec_format(
        'vocab.txt',
        'input_vectors.txt',
        'output_vectors.txt',
        vocab,
        W_matrix,
        U_matrix
    )

    print 'Done!'
