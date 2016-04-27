# -*- coding: UTF-8 -*-

from __future__ import division, unicode_literals
import time

import threading


batches_processed = 0
total_loss = 0
prev_time = None


class TrainingThread(threading.Thread):
    INFO_EVERY_N = 1000

    def __init__(self, batch_size, model, session,
                 task_queue, summary_writer=None):
        super(TrainingThread, self).__init__()
        self.daemon = True

        self.batch_size = batch_size
        self.model = model
        self.session = session

        self.task_queue = task_queue
        self.summary_writer = summary_writer

    def run(self):
        global batches_processed, total_loss
        global prev_time

        if prev_time is None:
            prev_time = time.time()

        while True:
            input_data = self.task_queue.get()

            if input_data is None:
                break

            epoch_num, text_num, mini_batch, noise = input_data
            words, contexts = zip(*mini_batch)

            feed_dict = {
                self.model.input_words: words,
                self.model.real_contexts: contexts,
                self.model.sampled_contexts: noise
            }

            _, loss_value = self.session.run(
                (
                    self.model.train_op,
                    self.model.loss
                ),
                feed_dict
            )

            batches_processed += 1
            total_loss += loss_value

            if batches_processed % self.INFO_EVERY_N == 0:
                if self.summary_writer is not None:
                    current_summary = self.session.run(
                        self.model.summary_op, feed_dict
                    )

                    self.summary_writer.add_summary(
                        summary=current_summary,
                        global_step=batches_processed
                    )

                time_delta = time.time() - prev_time

                print 'Epoch {}: {} texts / {} batches processed'.format(
                    epoch_num, text_num, batches_processed
                )

                print 'Total time: {}. Average batch time: {}'.format(
                    time_delta, time_delta / self.INFO_EVERY_N
                )

                print 'Average training loss: {}'.format(
                    total_loss / (self.batch_size * batches_processed)
                )

                prev_time = time.time()
