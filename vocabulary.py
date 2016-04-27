# -*- coding: UTF-8 -*-

from __future__ import division, unicode_literals
from collections import Counter

from cached_property import cached_property


class Vocabulary(object):
    UNKNOWN_TOKEN = '<UNKNOWN>'

    @staticmethod
    def from_documents(documents, min_freq=1):
        vocab = Vocabulary()

        # строим счётчик
        word_counter = Counter()
        for text in documents:
            word_counter.update(text)

        for text in documents:
            for word in text:
                # редкие слова заменяем на UNKNOWN_TOKEN
                if word_counter[word] >= min_freq:
                    vocab.add_or_update(word)
                else:
                    vocab.add_or_update(Vocabulary.UNKNOWN_TOKEN)

        return vocab

    def __init__(self):
        self._vocab = {}
        self._reverse_vocab = {}
        self._counts = {}

    def _add_word(self, word):
        current_idx = len(self._vocab)

        self._vocab[word] = current_idx
        self._reverse_vocab[current_idx] = word
        self._counts[current_idx] = 1

        return current_idx

    def _update_word(self, word):
        word_idx = self._vocab[word]
        self._counts[word_idx] += 1

    def add_or_update(self, word):
        if word not in self:
            self._add_word(word)
        else:
            self._update_word(word)

    def __len__(self):
        return len(self._vocab)

    def __getitem__(self, word):
        return self._vocab[word]

    def __contains__(self, word):
        return word in self._vocab

    @cached_property
    def words(self):
        words_counts = [
            (word, self.frequency(word))
            for word in self._vocab.keys()
        ]

        sorted_words = sorted(words_counts, key=lambda x: x[1], reverse=True)

        return [word for word, freq in sorted_words]

    def get(self, word):
        return self._vocab.get(word)

    def get_or_unknown(self, word):
        return self._vocab.get(
            word, self._vocab[Vocabulary.UNKNOWN_TOKEN]
        )

    def reverse_get(self, word_index):
        return self._reverse_vocab.get(word_index)

    def frequency(self, word):
        return self._counts.get(self.get(word), 0)
