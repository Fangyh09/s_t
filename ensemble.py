# import tensorflow as tf
# from model.data_utils import pad_sequences, get_chunks
# from model.data_utils import CoNLLDataset, minibatches
# from config import Config
# import h5py
# import numpy as np
# from util.get_elmo import build_conf, configs
#
# def get_feed_dict(words, elmo_embedding, labels=None, lr=None,
#                   dropout=None):
#     """Given some data, pad it and build a feed dictionary
#
#     Args:
#         words: list of sentences. A sentence is a list of ids of a list of
#             words. A word is a list of ids
#         labels: list of ids
#         lr: (float) learning rate
#         dropout: (float) keep prob
#
#     Returns:
#         dict {placeholder: value}
#
#     """
#     config = Config()
#     # perform padding of the given data
#     if config.use_chars:
#         #yes
#         char_ids, word_ids = zip(*words)
#         word_ids, sequence_lengths = pad_sequences(word_ids, 0)
#         char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
#                                                nlevels=2)
#     else:
#         word_ids, sequence_lengths = pad_sequences(words, 0)
#
#     feed = {
#         "word_ids:0": word_ids,
#         "sequence_lengths:0": sequence_lengths,
#         "lr_1:0": elmo_embedding
#     }
#
#     if config.use_chars:
#         feed["char_ids:0"] = char_ids
#         feed["word_lengths:0"] = word_lengths
#
#     if labels is not None:
#         labels, _ = pad_sequences(labels, 0)
#         feed["labels:0"] = labels
#
#     if lr is not None:
#         feed["lr:0"] = lr
#
#     if dropout is not None:
#         feed["dropout:0"] = dropout
#
#     return feed, sequence_lengths
#
#
#
#
# class EnsembleGraph:
#     """  Importing and running isolated TF graph """
#     def __init__(self, config, models):
#         # Create local graph and use it in the session
#         self.graph = tf.Graph()
#         self.config = config
#         self.models = models
#         self.num_models = len(models)
#         # config = tf.ConfigProto(log_device_placement=False)
#         # config.gpu_options.allow_growth = True
#         # self.sess = tf.Session(graph=self.graph, config=config)
#         # with self.graph.as_default():
#         #     # Import saved model from location 'loc' into local graph
#         #     saver = tf.train.import_meta_graph(loc + '.meta',
#         #                                        clear_devices=True)
#         #     saver.restore(self.sess, loc)
#         #     # There are TWO options how to get activation operation:
#         #       # FROM SAVED COLLECTION:
#         #     self.logits = self.graph.get_operation_by_name('proj/Reshape_1').outputs[0]
#         #     # self.activation = tf.get_collection('activation')[0]
#         #       # BY NAME:
#         #     # self.activation = self.graph.get_operation_by_name('activation_opt').outputs[0]
#
#
#     def get_logits(self, config, words, embeddings, labels=None, lr=None,
#                   dropout=None):
#         elmo_embedding = embeddings[str(i)][:].tolist()
#
#         config = build_conf(config)
#
#         fd_out, _ = get_feed_dict(words, elmo_embedding, dropout=1.0)
#         result_logits = []
#         for model in self.models:
#             res = model.run(fd_out)
#             result_logits.append(res[0])
#         result_logits = np.array(result_logits)
#
#         feed = {}
#         feed[self.logits] = result_logits
#
#         if labels is not None:
#             labels, _ = pad_sequences(labels, 0)
#             feed[self.labels] = labels
#
#         if config.use_chars:
#             # yes
#             char_ids, word_ids = zip(*words)
#             word_ids, sequence_lengths = pad_sequences(word_ids, 0)
#             char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
#                                                    nlevels=2)
#         else:
#             word_ids, sequence_lengths = pad_sequences(words, 0)
#
#         feed[self.sequence_lengths] = sequence_lengths
#         if lr is not None:
#             feed[self.lr] = lr
#         return feed, sequence_lengths
#
#
#     def add_placeholders(self):
#         with self.graph.as_default():
#             self.inputlogits = tf.placeholder(dtype=tf.float32,
#                                          shape=[self.num_models,
#                                                 self.config.batch_size, None,
#                                                 self.config.ntags],
#                                          name="inputlogits")
#
#             self.labels = tf.placeholder(tf.int32, shape=[None, None],
#                                          name="labels")
#
#             self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
#                                                    name="sequence_lengths")
#
#             self.lr = tf.placeholder(dtype=tf.float32, shape=[],
#                                      name="lr")
#
#     def add_loss_op(self):
#         with self.graph.as_default():
#             with tf.variable_scope("ensemble"):
#                 W = tf.get_variable("W", dtype=tf.float32,
#                                     shape=[1, self.num_models])
#                 normal_W = tf.nn.softmax(W)
#                 #[3, 10, 354, 10] -> [1, 10, 354, 10]
#                 flat_logits = tf.reshape(self.logits, [self.num_models, -1])
#                 print("ok?????????")
#                 s = tf.shape(self.logits)
#                 weighted_logits = tf.matmul(normal_W, flat_logits)
#                 weighted_logits = tf.reshape(weighted_logits,
#                                              [s[1], -1,
#                                               self.config.ntags])
#                 # weighted_logits = self.logits[0]
#
#                 log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
#                     weighted_logits, self.labels, self.sequence_lengths)
#
#                 self.trans_params = trans_params
#                 self.loss = tf.reduce_mean(-log_likelihood)
#                 tf.summary.scalar("loss", self.loss)
#                 # todo
#                 self.add_train_op("adam", 1e-3, self.loss,
#                                   5)
#
#     def initialize_session(self):
#         config = tf.ConfigProto(log_device_placement=False)
#         config.gpu_options.allow_growth = True
#         self.sess = tf.Session(graph=self.graph, config=config)
#
#         self.sess.run(tf.global_variables_initializer())
#         self.saver = tf.train.Saver()
#
#     def build(self):
#         self.add_placeholders()
#         self.add_loss_op()
#         self.initialize_session()
#
#
#     def add_summary(self):
#         """Defines variables for Tensorboard
#
#         Args:
#             dir_output: (string) where the results are written
#
#         """
#         self.merged = tf.summary.merge_all()
#
#         from datetime import datetime
#         now = datetime.now()
#         # logdir = "-" + now.strftime("%Y%m%d-%H%M%S") + "/"
#         logdir = ""
#         self.file_writer = tf.summary.FileWriter(
#             self.config.dir_output + "train" + logdir,
#             self.sess.graph)
#         self.file_epoch_writer = tf.summary.FileWriter(
#             self.config.dir_output + "test" + logdir)
#
#
#     def run_epoch(self, train, dev, epoch, train_embeddings, dev_embeddings):
#         """Performs one complete pass over the train set and evaluate on dev
#
#         Args:
#             train: dataset that yields tuple of sentences, tags
#             dev: dataset
#             epoch: (int) index of the current epoch
#
#         Returns:
#             f1: (python float), score to select model on, higher is better
#
#         """
#         # progbar stuff for logging
#         batch_size = self.config.batch_size
#         # hack it
#         # nbatches = (len(train) + batch_size - 1) // batch_size
#         nbatches = (11421 + batch_size - 1) // batch_size
#         # nbatches = (11421 + batch_size - 1) // batch_size
#
#         prog = Progbar(target=nbatches)
#
#         # iterate over dataset
#         for i, (words, labels) in enumerate(minibatches(train,
#                                                              batch_size)):
#
#             elmo_embedding = embeddings[str(i)][:].tolist()
#             default_config = Config()
#             fd, _ = self.get_logits(default_config, words, elmo_embedding,
#                                     labels,
#                                     self.config.lr,
#                                     self.config.dropout)
#
#
#             # print("hahaha, I got it!!!!!!!!!")
#             # elmo_embedding = train_embeddings[str(i)][:].tolist()
#             _, train_loss, summary = self.sess.run(
#                     [self.train_op, self.loss, self.merged], feed_dict=fd)
#
#             prog.update(i + 1, [("train loss", train_loss)])
#
#             # tensorboard
#             if i % 10 == 0:
#                 self.file_writer.add_summary(summary, epoch*nbatches + i)
#             # if i % 100 == 0:
#             #     self.file_writer.flush()
#
#         metrics = self.run_evaluate(dev, dev_embeddings)
#
#         summary = tf.Summary()
#         summary.value.add(tag="acc", simple_value=metrics["acc"])
#         summary.value.add(tag="f1", simple_value=metrics["f1"])
#         self.file_epoch_writer.add_summary(summary, epoch)
#         self.file_epoch_writer.flush()
#         # summary = self.sess.run([self.merged])
#         # self.file_epoch_wr1iter.add_summary(summary, epoch)
#         msg = " - ".join(["{} {:04.2f}".format(k, v)
#                 for k, v in metrics.items()])
#         self.logger.info(msg)
#
#         return metrics["f1"]
#
#
#     def add_train_op(self, lr_method, lr, loss, clip=-1):
#         """Defines self.train_op that performs an update on a batch
#
#         Args:
#             lr_method: (string) sgd method, for example "adam"
#             lr: (tf.placeholder) tf.float32, learning rate
#             loss: (tensor) tf.float32 loss to minimize
#             clip: (python float) clipping of gradient. If < 0, no clipping
#
#         """
#         _lr_m = lr_method.lower() # lower to make sure
#
#         with tf.variable_scope("train_step"):
#             if _lr_m == 'adam': # sgd method
#                 optimizer = tf.train.AdamOptimizer(lr)
#             elif _lr_m == 'adagrad':
#                 optimizer = tf.train.AdagradOptimizer(lr)
#             elif _lr_m == 'sgd':
#                 optimizer = tf.train.GradientDescentOptimizer(lr)
#             elif _lr_m == 'rmsprop':
#                 optimizer = tf.train.RMSPropOptimizer(lr)
#             else:
#                 raise NotImplementedError("Unknown method {}".format(_lr_m))
#
#             if clip > 0: # gradient clipping if clip is positive
#                 grads, vs     = zip(*optimizer.compute_gradients(loss))
#                 grads, gnorm  = tf.clip_by_global_norm(grads, clip)
#                 self.train_op = optimizer.apply_gradients(zip(grads, vs))
#             else:
#                 self.train_op = optimizer.minimize(loss)
#
#
#     def train(self, train, dev, train_embeddings, dev_embeddings,
#               reporter=False):
#         """Performs training with early stopping and lr exponential decay
#
#         Args:
#             train: dataset that yields tuple of (sentences, tags)
#             dev: dataset
#
#         """
#         best_score = 0
#         nepoch_no_imprv = 0 # for early stopping
#         self.add_summary() # tensorboard
#         decay_nums = 0
#         for epoch in range(self.config.nepochs):
#             self.logger.info("Epoch {:} out of {:}".format(epoch + 1,
#                         self.config.nepochs))
#
#             score = self.run_epoch(train, dev, epoch, train_embeddings, dev_embeddings)
#
#             if self.config.decay_mode == "normal":
#                 self.config.lr *= self.config.lr_decay # decay learning rate
#             elif self.config.decay_mode == "4normal":
#                 if epoch % 4 == 0:
#                     self.config.lr *= self.config.lr_decay
#
#             if reporter is not False:
#                 reporter(timesteps_total=epoch, mean_accuracy=score)
#
#             # early stopping and saving best parameters
#             if score >= best_score:
#                 nepoch_no_imprv = 0
#                 self.save_session(epoch=epoch)
#                 best_score = score
#                 self.logger.info("- new best score!")
#             else:
#                 nepoch_no_imprv += 1
#                 # if decay_nums == 0:
#                 #     self.config.lr = self.config.lr / (1 + self.config.lr_decay * decay_nums)
#                 # else:
#                 #     self.config.lr = self.config.lr * (1 + self.config.lr_decay * (decay_nums - 1)) / (
#                 #                 1 + self.config.lr_decay * decay_nums)
#                 if self.config.decay_mode == "greedy":
#                     self.config.lr = self.config.lr * self.config.lr_decay
#                 # print("===> lr decay=", self.config.lr)
#                 if self.config.decay_mode == "greedy-half":
#                     self.config.lr /= 2.0
#                 # decay_nums += 1
#
#                 if nepoch_no_imprv >= self.config.nepoch_no_imprv:
#                     self.logger.info("- early stopping {} epochs without "\
#                             "improvement".format(nepoch_no_imprv))
#                     break
#
#             # if not self.config.use_cnn and score < 70:
#             #     return best_score
#             #
#             # if self.config.use_cnn and score < 10:
#             #     return best_score
#
#         return best_score
#
#     def run_evaluate(self, test, dev_embeddings):
#         """Evaluates performance on test set
#
#         Args:
#             test: dataset that yields tuple of (sentences, tags)
#
#         Returns:
#             metrics: (dict) metrics["acc"] = 98.4, ...
#
#         """
#         accs = []
#         correct_preds, total_correct, total_preds = 0., 0., 0.
#         for idx, (words, labels) in enumerate(minibatches(test,
#                                                           self.config.batch_size)):
#             elmo_embedding = dev_embeddings[str(idx)][:]
#             labels_pred, sequence_lengths = self.predict_batch(words,
#                                                                elmo_embedding)
#
#             for lab, lab_pred, length in zip(labels, labels_pred,
#                                              sequence_lengths):
#                 lab = lab[:length]
#                 lab_pred = lab_pred[:length]
#                 accs += [a == b for (a, b) in zip(lab, lab_pred)]
#
#                 lab_chunks = set(get_chunks(lab, self.config.vocab_tags))
#                 lab_pred_chunks = set(get_chunks(lab_pred,
#                                                  self.config.vocab_tags))
#
#                 correct_preds += len(lab_chunks & lab_pred_chunks)
#                 total_preds += len(lab_pred_chunks)
#                 total_correct += len(lab_chunks)
#
#         p = correct_preds / total_preds if correct_preds > 0 else 0
#         r = correct_preds / total_correct if correct_preds > 0 else 0
#         f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
#         acc = np.mean(accs)
#
#         #
#         #
#         # tf.summary.scalar("acc", tf.convert_to_tensor(acc, np.float32))
#         # tf.summary.scalar("f1", tf.convert_to_tensor(f1, np.float32))
#         return {"acc": 100 * acc, "f1": 100 * f1}
#
#     def predict_batch(self, words, elmo_embedding, withprob=False):
#         """
#         Args:
#             words: list of sentences
#
#         Returns:
#             labels_pred: list of labels for each sentence
#             sequence_length
#
#         """
#         # prepare data
#         fd, sequence_lengths = self.get_logits(words, elmo_embedding,
#                                                   dropout=1.0)
#
#         if self.config.use_crf:
#             # get tag scores and transition params of CRF
#             viterbi_sequences = []
#             viterbi_score_all = []
#             logits, trans_params = self.sess.run(
#                     [self.logits, self.trans_params], feed_dict=fd)
#
#             # iterate over the sentences because no batching in vitervi_decode
#             for logit, sequence_length in zip(logits, sequence_lengths):
#                 logit = logit[:sequence_length] # keep only the valid steps
#                 viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
#                         logit, trans_params)
#                 viterbi_sequences += [viterbi_seq]
#                 viterbi_score_all += [viterbi_score]
#             if withprob:
#                 return viterbi_sequences, viterbi_score_all, sequence_lengths
#             else:
#                 return viterbi_sequences, sequence_lengths
#
#         else:
#             labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)
#
#             return labels_pred, sequence_lengths
#
# def main():
#     loc1 = "/SSD1/yinghong/tmp/s_t_elmo/rayresults/elmo-offline" \
#                   "-config9/model.weights/elmo-model2018-06-23-02-07"
#     loc2 = "/SSD1/yinghong/tmp/s_t_elmo/rayresults/elmo-offline" \
#                   "-config13/model.weights/elmo-model2018-06-23-02-08"
#     config = tf.ConfigProto(log_device_placement=False)
#     config.gpu_options.allow_growth = True
#     sess = tf.Session(config=config)
#     saver1 = tf.train.import_meta_graph(loc1 + '.meta',
#                                        clear_devices=True,
#                                         import_scope="red1")
#     saver1.restore(sess, loc1)
#     saver2 = tf.train.import_meta_graph(loc1 + '.meta',
#                                        clear_devices=True, import_scope="red2")
#     saver2.restore(sess, loc2)
#
#
#
#
#
# if __name__ == "__main__":
#     # main()
#     # mode = "dev"
#
#     train_embeddings = h5py.File("data/elmo_" + "train" + ".embedding.h5", 'r')
#     dev_embeddings = h5py.File("data/elmo_" + "dev" + ".embedding.h5", 'r')
#
#     default_config = Config()
#     train = CoNLLDataset(default_config.filename_train,
#                         default_config.processing_word,
#                         default_config.processing_tag, default_config.max_iter,
#                         test=True)
#     dev = CoNLLDataset(default_config.filename_dev,
#                        default_config.processing_word,
#                        default_config.processing_tag, default_config.max_iter)
#     # if mode == "dev":
#     #     dataset = dev
#     #     save_path = "presubmit"
#     # elif mode == "test":
#     #     dataset = test
#     #     save_path = "submit"
#
#     modelname_9 = "/SSD1/yinghong/tmp/s_t_elmo/rayresults/elmo-offline" \
#                   "-config9/model.weights/elmo-model2018-06-23-02-07"
#
#     modelname_13 = "/SSD1/yinghong/tmp/s_t_elmo/rayresults/elmo-offline" \
#            "-config13/model.weights/elmo-model2018-06-23-02-08"
#
#     model_names = [modelname_9, modelname_13]
#     models = []
#     all_config = [build_conf(configs["config9"]), build_conf(configs[
#                                                                  "config9"])]
#
#     for name in model_names:
#         model = ImportGraph(name)
#         models.append(model)
#
#
#     # for i, (words, labels) in enumerate(minibatches(dataset,
#     #                                                 minibatch_size=10)):
#     #     # print("hahaha, I got it!!!!!!!!!")
#     #     elmo_embedding = embeddings[str(i)][:].tolist()
#     #
#     #     print(type(elmo_embedding))
#     #     fd, _ = get_feed_dict(words, elmo_embedding, dropout=1.0)
#     #     result_logits = []
#     #     for model in models:
#     #        res = model.run(fd)
#     #        result_logits.append(res[0])
#     #     result_logits = np.array(result_logits)
#     #     print(result_logits.shape)
#     #
#     #     print("done!!!!")
#     print("ok?????????")
#
#     model = EnsembleGraph(build_conf(configs["config-ensem-1"]), models)
#     print("ok?????????")
#     model.build()
#     print("ok!!!!!!!!")
#     model.train(train, dev, train_embeddings, dev_embeddings)
#     # config = Config()
#     # graph = EnsembleGraph(config=config)
#
