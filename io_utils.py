# -*- coding: UTF-8 -*-

from __future__ import division, unicode_literals

import codecs


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
