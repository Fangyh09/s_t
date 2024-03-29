import numpy as np
import os
import tensorflow as tf


from .data_utils import minibatches, pad_sequences, get_chunks
from .general_utils import Progbar
from .base_model import BaseModel


class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}


    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")

        # self.max_length_word = tf.placeholder(dtype=tf.int32, shape=[],
        #                 name="max_length_word")
        # print(type(self.max_length_word))


    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            # all words has same length of vector

            #==> word_ids:
            # [[525, 1504, 8139, 6984, 4217, 5867, 4217, 5299, 677, 3865, 4856, 319, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #   0, 0, 0, 0, 0],
            #  [2491, 4856, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #  [4856, 7006, 2474, 5131, 319, 0, 0, 0, 0, 0, 0,
            #   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

            #==> sequence_lengths:
            # [12, 2, 5, 2, 12, 14, 5, 2, 2, 29, 4, 10, 10, 2, 18, 2, 2, 6, 8, 9]
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            #==> char_ids:
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0,
            # 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0,
            # 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0,
            #==> word_lengths:
            # [[7, 9, 7, 9, 1, 10, 1, 3, 3, 4, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [7, 8, 0, 0, 0
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                nlevels=2)
            # self.max_length_word = max_length_word
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths
            # feed[self.max_length_word] = max_length_word

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        # sequence_lengths: [12, 2, 5, 2, 12, 14, 5, 2, 2, 29, 4, 10, 10, 2, 18, 2, 2, 6, 8, 9]
        return feed, sequence_lengths


    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars", initializer=tf.orthogonal_initializer()):
            if self.config.use_chars:
                if self.config.use_cnn:
                    # get char embeddings matrix
                    _char_embeddings = tf.get_variable(
                            name="_char_embeddings",
                            dtype=tf.float32,
                            shape=[self.config.nchars, self.config.dim_char])
                    char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                            self.char_ids, name="char_embeddings")

                    # put the time dimension on axis=1
                    s = tf.shape(char_embeddings)
                    char_embeddings = tf.reshape(char_embeddings,
                            shape=[s[0]*s[1], s[-2], self.config.dim_char])
                    word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                    # # bi lstm on chars
                    # cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                    #         state_is_tuple=True)
                    # cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                    #         state_is_tuple=True)
                    # # cell_fw = tf.contrib.rnn.GRUCell(self.config.hidden_size_char)
                    # # cell_bw = tf.contrib.rnn.GRUCell(self.config.hidden_size_char)
                    #
                    # _output = tf.nn.bidirectional_dynamic_rnn(
                    #         cell_fw, cell_bw, char_embeddings,
                    #         sequence_length=word_lengths, dtype=tf.float32)
                    #
                    # # read and concat output
                    # _, ((_, output_fw), (_, output_bw)) = _output
                    # output = tf.concat([output_fw, output_bw], axis=-1)
                    #
                    # # shape = (batch size, max sentence length, char hidden size)
                    # output = tf.reshape(output,
                    #         shape=[s[0], s[1], 2*self.config.hidden_size_char])
                    char_embeddings_expanded = tf.expand_dims(char_embeddings, -1)
                    pooled_outputs = []
                    filter_sizes = [3, 4, 5]
                    num_filters = 128
                    dropout_keep_prob = 0.5
                    for i, filter_size in enumerate(filter_sizes):
                        with tf.name_scope("conv-maxpool-%s" % filter_size):
                            # Convolution Layer
                            filter_shape = [filter_size, self.config.dim_char, 1, num_filters]
                            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                            conv = tf.nn.conv2d(
                                char_embeddings_expanded,
                                W,
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="conv")
                            # Apply nonlinearity

                            # # hh = h
                            hh = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                            hh = tf.squeeze(hh, [2])
                            pooled = tf.reduce_max(hh, axis=[1])


                            #
                            # # pooled = tf.layers.max_pooling1d(hh, tf.shape(hh)[1], tf.shape(hh)[1])
                            # print("shape pooled", pooled)

                            # pooled = tf.squeeze(pooled, [1])
                            # Maxpooling over the outputs
                            # pooled = tf.nn.max_pool(
                            #     h,
                            #     ksize=[1, self.max_length_word[0] - filter_size + 1, 1, 1],
                            #     strides=[1, 1, 1, 1],
                            #     padding='VALID',
                            #     name="pool")
                            pooled_outputs.append(pooled)

                    # print("ok!!!!")
                    # tf.Print("pooled_outputs", pooled_outputs)

                    # num_filters_total = num_filters * len(filter_sizes)

                    h_pool = tf.concat(pooled_outputs, 1)


                    h_pool = tf.reshape(h_pool, shape=[s[0], s[1], 128 * 3])
                    # print("h_pool", h_pool)



                    # h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
                    h_drop = tf.nn.dropout(h_pool, dropout_keep_prob)


                    # print(word_embeddings)

                    # todo uncomment it
                    word_embeddings = tf.concat([word_embeddings, h_drop], axis=-1)
                else:
                    # get char embeddings matrix
                    _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                    char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                             self.char_ids, name="char_embeddings")

                    # put the time dimension on axis=1
                    s = tf.shape(char_embeddings)
                    char_embeddings = tf.reshape(char_embeddings,
                                                 shape=[s[0] * s[1], s[-2], self.config.dim_char])
                    word_lengths = tf.reshape(self.word_lengths, shape=[s[0] * s[1]])

                    # bi lstm on chars
                    cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                                                      state_is_tuple=True)
                    cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                                                      state_is_tuple=True)
                    _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float32)

                    # read and concat output
                    _, ((_, output_fw), (_, output_bw)) = _output
                    output = tf.concat([output_fw, output_bw], axis=-1)

                    # shape = (batch size, max sentence length, char hidden size)
                    output = tf.reshape(output,
                                        shape=[s[0], s[1], 2 * self.config.hidden_size_char])
                    word_embeddings = tf.concat([word_embeddings, output], axis=-1)
                # print("word_embeddings", word_embeddings)

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)


    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm", initializer=tf.orthogonal_initializer()):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        # with tf.variable_scope("bi-lstm", initializer=tf.orthogonal_initializer()):
        #     cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
        #     cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
        #     (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
        #         cell_fw, cell_bw, output,
        #         sequence_length=self.sequence_lengths, dtype=tf.float32)
        #     output = tf.concat([output_fw, output_bw], axis=-1)
        #     output = tf.nn.dropout(output, self.dropout)


        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])


    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                    tf.int32)


    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)


    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init
#       path = "results/test/model.weights"
#self.restore_session(path)


    def predict_batch(self, words, withprob=False):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        # prepare data
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            viterbi_score_all = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]
                viterbi_score_all += [viterbi_score]
            if withprob:
                return viterbi_sequences, viterbi_score_all, sequence_lengths
            else:
                return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths


    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout)

            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)
            # if i % 100 == 0:
            #     self.file_writer.flush()

        metrics = self.run_evaluate(dev)

        summary = tf.Summary()
        summary.value.add(tag="acc", simple_value=metrics["acc"])
        summary.value.add(tag="f1", simple_value=metrics["f1"])
        self.file_epoch_writer.add_summary(summary, epoch)
        self.file_epoch_writer.flush()
        # summary = self.sess.run([self.merged])
        # self.file_epoch_wr1iter.add_summary(summary, epoch)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]


    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        #
        #
        # tf.summary.scalar("acc", tf.convert_to_tensor(acc, np.float32))
        # tf.summary.scalar("f1", tf.convert_to_tensor(f1, np.float32))
        return {"acc": 100*acc, "f1": 100*f1}


    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        # convert w into number
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds

    def tmp(self, test, outfile="result.txt"):
        fout = open(outfile, "w+")
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, _ = self.predict_batch(words)
            for sent in list(labels_pred):
                for wordidx in list(sent):
                    tag = self.idx_to_tag[wordidx]
                    fout.write(tag + "\n")
                fout.write("\n")



