import os

import tensorflow as tf
from tensorflow.python import debug as tf_debug

TF_DEBUG = False

class BaseModel(object):
    """Generic class for general methods that are not specific to NER"""

    def __init__(self, config):
        """Defines self.config and self.logger

        Args:
            config: (Config instance) class with hyper parameters,
                vocab and embeddings

        """
        self.config = config
        self.logger = config.logger
        self.sess   = None
        self.saver  = None


    def reinitialize_weights(self, scope_name):
        """Reinitializes the weights of a given layer"""
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)


    def add_train_op(self, lr_method, lr, loss, clip=-1):
        """Defines self.train_op that performs an update on a batch

        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient. If < 0, no clipping

        """
        _lr_m = lr_method.lower() # lower to make sure

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam': # sgd method
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0: # gradient clipping if clip is positive
                grads, vs     = zip(*optimizer.compute_gradients(loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)


    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        self.logger.info("Initializing tf session")
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
        # from tensorflow.python import debug as tf_debug
        if TF_DEBUG:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()


    def restore_session(self, dir_model):
        """Reload weights into session

        Args:
            sess: tf.Session()
            dir_model: dir with weights

        """
        self.logger.info("Reloading the latest trained model...")
        # meta = "/home/yinghong/project/tmp/s_t_rollback/ray_results/06-19/01" \
        #        "-HasCNN/try85.63/results/tmptmptest/bz=10-training-bieo-nocnn/model.weights/final-model2018-06-20-21-08.meta" \
        #        ""
        # self.saver = tf.train.import_meta_graph(meta)
        # self.saver.restore(self.sess,
        #                    "/home/yinghong/project/tmp/s_t_rollback/ray_results/06-19/01-HasCNN/try85.63/results/tmptmptest/bz=10-training-bieo-nocnn/model.weights/"
        #                    "final-model2018-06-20-21-08")

        self.saver.restore(self.sess, tf.train.latest_checkpoint(dir_model))


    def save_session(self, epoch=""):
        """Saves session = weights"""
        import datetime
        date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model+date_str)
        # if epoch != "":
        #     self.saver.save(self.sess, self.config.dir_model,
        #                     global_step=epoch)
        # else:
        self.saver.save(self.sess, self.config.dir_model+date_str)

    def close_session(self):
        """Closes the session"""
        self.sess.close()


    def add_summary(self):
        """Defines variables for Tensorboard

        Args:
            dir_output: (string) where the results are written

        """
        self.merged = tf.summary.merge_all()

        from datetime import datetime
        now = datetime.now()
        # logdir = "-" + now.strftime("%Y%m%d-%H%M%S") + "/"
        logdir = ""
        self.file_writer = tf.summary.FileWriter(self.config.dir_output + "train" + logdir,
                                                 self.sess.graph)
        self.file_epoch_writer = tf.summary.FileWriter(self.config.dir_output + "test" + logdir)


    def train(self, train, dev, reporter=False):
        """Performs training with early stopping and lr exponential decay

        Args:
            train: dataset that yields tuple of (sentences, tags)
            dev: dataset

        """
        best_score = 0
        nepoch_no_imprv = 0 # for early stopping
        self.add_summary() # tensorboard
        decay_nums = 0
        for epoch in range(self.config.nepochs):
            self.logger.info("Epoch {:} out of {:}".format(epoch + 1,
                        self.config.nepochs))

            score = self.run_epoch(train, dev, epoch)

            if self.config.decay_mode == "normal":
                self.config.lr *= self.config.lr_decay # decay learning rate
            elif self.config.decay_mode == "4normal":
                if epoch % 4 == 0:
                    self.config.lr *= self.config.lr_decay

            if reporter is not False:
                reporter(timesteps_total=epoch, mean_accuracy=score)

            # early stopping and saving best parameters
            if score >= best_score:
                nepoch_no_imprv = 0
                self.save_session(epoch=epoch)
                best_score = score
                self.logger.info("- new best score!")
            else:
                nepoch_no_imprv += 1
                # if decay_nums == 0:
                #     self.config.lr = self.config.lr / (1 + self.config.lr_decay * decay_nums)
                # else:
                #     self.config.lr = self.config.lr * (1 + self.config.lr_decay * (decay_nums - 1)) / (
                #                 1 + self.config.lr_decay * decay_nums)
                if self.config.decay_mode == "greedy":
                    self.config.lr = self.config.lr * self.config.lr_decay
                # print("===> lr decay=", self.config.lr)
                if self.config.decay_mode == "greedy-half":
                    self.config.lr /= 2.0
                # decay_nums += 1

                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    self.logger.info("- early stopping {} epochs without "\
                            "improvement".format(nepoch_no_imprv))
                    break

            # if not self.config.use_cnn and score < 70:
            #     return best_score
            #
            # if self.config.use_cnn and score < 10:
            #     return best_score

        return best_score


    def evaluate(self, test):
        """Evaluate model on test set

        Args:
            test: instance of class Dataset

        """
        self.logger.info("Testing model over test set")
        metrics = self.run_evaluate(test)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)
