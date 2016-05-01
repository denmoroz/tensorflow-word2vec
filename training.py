# -*- coding: UTF-8 -*-

from __future__ import division, unicode_literals

import time
import threading


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
        while True:
            input_data = self.task_queue.get()

            if input_data is None:
                break

            batch_num, batch_data, noise = input_data
            words, contexts = zip(*batch_data)

            feed_dict = {
                self.model.input_words: words,
                self.model.real_contexts: contexts,
                self.model.sampled_contexts: noise
            }

            self.session.run(self.model.train_op, feed_dict)

            if batch_num % self.INFO_EVERY_N == 0:
                if self.summary_writer is not None:
                    current_summary = self.session.run(
                        self.model.summary_op, feed_dict
                    )

                    self.summary_writer.add_summary(
                        summary=current_summary,
                        global_step=batch_num
                    )


class ConsoleMonitoringThread(threading.Thread):

    def __init__(self, logger, check_interval, model, session):
        super(ConsoleMonitoringThread, self).__init__()
        self.daemon = True

        self.logger = logger
        self.check_interval = check_interval

        self.model = model
        self.session = session

        self.working = True

    def stop_monitoring(self):
        self.working = False

    def run(self):
        prev_batches_num = 0

        while self.working:
            batches_num, avg_loss = self.session.run(
                [self.model.batches_processed, self.model.average_loss]
            )

            self.logger.info(
                'Total batches processed: {}; Average batch loss: {}'.format(
                    batches_num, avg_loss
                )
            )

            self.logger.info(
                'Batches per second: {}'.format(
                    (batches_num - prev_batches_num) / self.check_interval
                )
            )
            prev_batches_num = batches_num

            time.sleep(self.check_interval)
