from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from config import Config
import tensorflow as tf
import ray
import ray.tune as tune
import os
from tensorflow.python import debug as tf_debug


PRETRAIN_MODE = True

def main2():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
# model.restore_session("results/crf/model.weights/") # optional, restore weights
#model.reinitialize_weights("proj")

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    # train model
    model.train(train, dev)


def main():

    default_config = Config()
    dev = CoNLLDataset(default_config.filename_dev, default_config.processing_word,
                       default_config.processing_tag, default_config.max_iter)
    train = CoNLLDataset(default_config.filename_train, default_config.processing_word,
                         default_config.processing_tag, default_config.max_iter)

    pretrain_path = "/home/yinghong/project/tmp/s_t/ray_results/go1-old/exp-go3/train_f" \
                    "unc_72_00-use_reg=False,13-input_keep_prob=1,15-output_keep_prob=1," \
                    "17-lstm_layers=1,19-clip=5,25-lr=0.003_2018-06-13_05-15-32e44db7z5"
    # @ray.remote(num_gpus=1)
    def train_func(_config, reporter):
        # tf.reset_default_graph()
        config = Config()
        # for (key, val) in _config.items():
        #     # config[key] = val
        #     setattr(config, key[3:], val)
        # config["dir_output"] = ""
        setattr(config, "dir_output", "")
        setattr(config, "nepochs", 50)
        setattr(config, "batch_size", 50)

        if PRETRAIN_MODE:
            config_path = os.path.join(pretrain_path, "params.json")
            with open(config_path) as fin:
                content = fin.read().replace('\n', '')
                import json
                j = json.loads(content)
                for (key, val) in j.items():
                    setattr(config, key, val)


        model = NERModel(config)
        model.build()
        if PRETRAIN_MODE:
            model.restore_session(os.path.join(pretrain_path, "results/tmptmptest/bz=10-training-"
                                                          "bieo-nocnn/model.weights/"))
        model.train(train, dev, reporter)

    # ray.init(redis_address="192.168.1.201:20198")
    ray.init(num_cpus=1, num_gpus=1)

    tune.register_trainable("train_func", train_func)

    # with tf.variable_scope("train_step"):
    #     if _lr_m == 'adam':  # sgd method
    #         optimizer = tf.train.AdamOptimizer(lr)
    #     elif _lr_m == 'adagrad':
    #         optimizer = tf.train.AdagradOptimizer(lr)
    #     elif _lr_m == 'sgd':
    #         optimizer = tf.train.GradientDescentOptimizer(lr)
    #     elif _lr_m == 'rmsprop':
    #         optimizer = tf.train.RMSPropOpt

    tune.run_experiments({
        # "exp-go1": {
        #     "run": "train_func",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "/home/cewu/thirdHDD/yinghong/ray_results/go1",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     # "num_gpus": 1,
        #
        #     "config": {
        #         # "use_reg": tune.grid_search([False, True]),
        #         "hidden_size_lstm": 300,
        #         "hidden_size_char": tune.grid_search([100, 50, 30]),
        #         "dim_char": tune.grid_search([100, 50, 30]),
        #         # "filter_sizes": tune.grid_search([[3], [3,4], [3,4,5]]),
        #         # "use_cnn": tune.grid_search([True, False]),
        #         "input_keep_prob": 1,
        #         "output_keep_prob": 1,
        #         # "lstm_layers": tune.grid_search([1, 2, 5]),
        #         "clip": tune.grid_search([0, 5]),
        #         "lr_method": "adam",
        #         "lr_decay": 0.95,
        #         "lr": 0.005,
        #
        #         # "resources": {"cpu": 1, "gpu": 1}
        #         # "momentum": tune.grid_search([0.1, 0.2]),
        #     }
        # },
        #####################
        # run on sjtu02
        #####################
        # "exp-go2-epoch40": {
        #     "run": "train_func",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/go1",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     # "num_gpus": 1,
        #
        #     "config": {
        #         # "00-use_reg": tune.grid_search([False, True]),
        #         "03-hidden_size_lstm": 300,
        #         # # "05-hidden_size_char": tune.grid_search([100, 50, 30]),
        #         "05-hidden_size_char": 100,
        #         # # "07-dim_char": tune.grid_search([100, 50, 30]),
        #         "07-dim_char": 100,
        #         "09-filter_sizes": tune.grid_search([[3], [3,4], [3,4,5]]),
        #         "11-use_cnn": True,
        #         "13-input_keep_prob": tune.grid_search([1, 0.5]),
        #         "15-output_keep_prob": tune.grid_search([1, 0.5]),
        #         "17-lstm_layers": tune.grid_search([1, 2, 5]),
        #         # "19-clip": tune.grid_search([0, 5]),
        #         "21-lr_method": "adam",
        #         "23-lr_decay": 0.9,
        #         "25-lr": 0.001,
        #         # "25-lr": tune.grid_search([0.001, 0.005]),
        #
        #         # "resources": {"cpu": 1, "gpu": 1}
        #         # "momentum": tune.grid_search([0.1, 0.2]),
        #     }
        # },
        "exp-pretrain-epoch50": {
            "run": "train_func",
            "stop": {"mean_accuracy": 99},
            "local_dir": "./ray_results/debug",
            "trial_resources": {'cpu': 0, 'gpu': 1},
            # "num_gpus": 1,

            "config": {
                # "00-use_reg": tune.grid_search([False, True]),
                # "03-hidden_size_lstm": 300,
                # # "05-hidden_size_char": tune.grid_search([100, 50, 30]),
                # "05-hidden_size_char": 100,
                # # "07-dim_char": tune.grid_search([100, 50, 30]),
                # "07-dim_char": 100,
                # "09-filter_sizes": tune.grid_search([[3,4], [3,4,5]]),
                # "11-use_cnn": True,
                # "13-input_keep_prob": 1,
                # "15-output_keep_prob": 1,
                # "17-lstm_layers": 2,
                # "19-clip": 5,
                # "21-lr_method": "adam",
                # "23-lr_decay": 0.9,
                "25-lr": 0.001,
                # "27-decay_mode": tune.grid_search(["normal", "greedy", "greedy-half"])
                # "25-lr": tune.grid_search([0.001, 0.005]),

                # "resources": {"cpu": 1, "gpu": 1}
                # "momentum": tune.grid_search([0.1, 0.2]),
            }
        },
        # "exp-debug-epoch50": {
        #     "run": "train_func",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/debug",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     # "num_gpus": 1,
        #
        #     "config": {
        #         # "00-use_reg": tune.grid_search([False, True]),
        #         # "03-hidden_size_lstm": 300,
        #         # # "05-hidden_size_char": tune.grid_search([100, 50, 30]),
        #         # "05-hidden_size_char": 100,
        #         # # "07-dim_char": tune.grid_search([100, 50, 30]),
        #         # "07-dim_char": 100,
        #         # "09-filter_sizes": tune.grid_search([[3,4], [3,4,5]]),
        #         # "11-use_cnn": True,
        #         # "13-input_keep_prob": 1,
        #         # "15-output_keep_prob": 1,
        #         # "17-lstm_layers": 2,
        #         # "19-clip": 5,
        #         # "21-lr_method": "adam",
        #         # "23-lr_decay": 0.9,
        #         "25-lr": 0.001,
        #         # "27-decay_mode": tune.grid_search(["normal", "greedy", "greedy-half"])
        #         # "25-lr": tune.grid_search([0.001, 0.005]),
        #
        #         # "resources": {"cpu": 1, "gpu": 1}
        #         # "momentum": tune.grid_search([0.1, 0.2]),
        #     }
        # },
        # "exp-go2-epoch50-03": {
        #     "run": "train_func",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/go1",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     # "num_gpus": 1,
        #
        #     "config": {
        #         # "00-use_reg": tune.grid_search([False, True]),
        #         # "03-hidden_size_lstm": 300,
        #         # # "05-hidden_size_char": tune.grid_search([100, 50, 30]),
        #         # "05-hidden_size_char": 100,
        #         # # "07-dim_char": tune.grid_search([100, 50, 30]),
        #         # "07-dim_char": 100,
        #         "09-filter_sizes": tune.grid_search([[3, 4], [3, 4, 5]]),
        #         "11-use_cnn": True,
        #         "13-input_keep_prob": 1,
        #         "15-output_keep_prob": 1,
        #         "17-lstm_layers": 2,
        #         "19-clip": 5,
        #         "21-lr_method": "sgd",
        #         "23-lr_decay": 0.9,
        #         "25-lr": 0.1,
        #         "27-decay_mode": "greedy-half"
        #         # "25-lr": tune.grid_search([0.001, 0.005]),
        #
        #         # "resources": {"cpu": 1, "gpu": 1}
        #         # "momentum": tune.grid_search([0.1, 0.2]),
        #     }
        # },
        # "exp-go2-epoch50-02": {
        #     "run": "train_func",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/go1",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     # "num_gpus": 1,
        #
        #     "config": {
        #         # "00-use_reg": tune.grid_search([False, True]),
        #         # "03-hidden_size_lstm": 300,
        #         # # "05-hidden_size_char": tune.grid_search([100, 50, 30]),
        #         # "05-hidden_size_char": 100,
        #         # # "07-dim_char": tune.grid_search([100, 50, 30]),
        #         # "07-dim_char": 100,
        #         "09-filter_sizes": tune.grid_search([[3, 4], [3, 4, 5]]),
        #         "11-use_cnn": True,
        #         "13-input_keep_prob": 1,
        #         "15-output_keep_prob": 1,
        #         "17-lstm_layers": 2,
        #         "19-clip": 5,
        #         "21-lr_method": "adam",
        #         "23-lr_decay": 0.9,
        #         "25-lr": 0.001,
        #         "27-decay_mode": tune.grid_search(["normal", "greedy", "greedy-half"]),
        #         "29-use_crf": False
        #         # "25-lr": tune.grid_search([0.001, 0.005]),
        #
        #         # "resources": {"cpu": 1, "gpu": 1}
        #         # "momentum": tune.grid_search([0.1, 0.2]),
        #     }
        # },
        # "exp-go2-epoch50-01": {
        #     "run": "train_func",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/go1",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     # "num_gpus": 1,
        #
        #     "config": {
        #         # "00-use_reg": tune.grid_search([False, True]),
        #         # "03-hidden_size_lstm": 300,
        #         # # "05-hidden_size_char": tune.grid_search([100, 50, 30]),
        #         # "05-hidden_size_char": 100,
        #         # # "07-dim_char": tune.grid_search([100, 50, 30]),
        #         # "07-dim_char": 100,
        #         "09-filter_sizes": tune.grid_search([[3, 4], [3, 4, 5]]),
        #         "11-use_cnn": True,
        #         "13-input_keep_prob": 1,
        #         "15-output_keep_prob": 1,
        #         "17-lstm_layers": 2,
        #         "19-clip": 5,
        #         "21-lr_method": "sgd",
        #         "23-lr_decay": 0.9,
        #         "25-lr": 0.1,
        #         "27-decay_mode": "greedy-half",
        #         "29-use_crf": False
        #         # "25-lr": tune.grid_search([0.001, 0.005]),
        #
        #         # "resources": {"cpu": 1, "gpu": 1}
        #         # "momentum": tune.grid_search([0.1, 0.2]),
        #     }
        # },

    })



if __name__ == "__main__":
    main2()
