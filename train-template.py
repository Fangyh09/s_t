from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from config import Config
from tensorflow.python import debug as tf_debug
import tensorflow as tf
import ray
import ray.tune as tune


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

    # @ray.remote(num_gpus=1)
    def train_func(_config, reporter):
        # tf.reset_default_graph()
        config = Config()
        for (key, val) in _config.items():
            # config[key] = val
            setattr(config, key[3:], val)
        # config["dir_output"] = ""
        setattr(config, "dir_output", "")
        setattr(config, "nepochs", 50)

        model = NERModel(config)
        model.build()
        model.train(train, dev, reporter)

    # ray.init(redis_address="192.168.1.201:20198")
    ray.init(num_cpus=1, num_gpus=2)

    tune.register_trainable("train_func_final", train_func)

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
        # "exp-go2-epoch50-04": {
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
        #         "09-filter_sizes": tune.grid_search([[3,4], [3,4,5]]),
        #         "11-use_cnn": True,
        #         "13-input_keep_prob": 1,
        #         "15-output_keep_prob": 1,
        #         "17-lstm_layers": 2,
        #         "19-clip": 5,
        #         "21-lr_method": "adam",
        #         "23-lr_decay": 0.9,
        #         "25-lr": 0.001,
        #         "27-decay_mode": tune.grid_search(["normal", "greedy", "greedy-half"])
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
        # "exp-go3": {
        #     "run": "train_func",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/go1",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     # "num_gpus": 1,
        #
        #     "config": {
        #         "00-use_reg": tune.grid_search([False, True]),
        #         "03-hidden_size_lstm": 300,
        #         # # "05-hidden_size_char": tune.grid_search([100, 50, 30]),
        #         "05-hidden_size_char": 100,
        #         # # "07-dim_char": tune.grid_search([100, 50, 30]),
        #         "07-dim_char": 100,
        #         # "09-filter_sizes": tune.grid_search([[3], [3, 4], [3, 4, 5]]),
        #         # "11-use_cnn": tune.grid_search([True, False]),
        #         "13-input_keep_prob": tune.grid_search([1, 0.5]),
        #         "15-output_keep_prob": tune.grid_search([1, 0.5]),
        #         "17-lstm_layers": tune.grid_search([1, 2, 5]),
        #         "19-clip": tune.grid_search([0, 5]),
        #         "21-lr_method": "adam",
        #         "23-lr_decay": 0.9,
        #         "25-lr": tune.grid_search([0.001, 0.003]),
        #         # "25-lr": tune.grid_search([0.001, 0.005]),
        #
        #         # "resources": {"cpu": 1, "gpu": 1}
        #         # "momentum": tune.grid_search([0.1, 0.2]),
        #     }
        # },
        # "exp2": {
        #     "run": "train_func",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "/home/cewu/thirdHDD/yinghong/ray_results/choose_bz-clip",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     # "num_gpus": 1,
        #     "config": {
        #         "batch_size": tune.grid_search([10, 15, 20]),
        #         "clip": 5
        #         # "resources": {"cpu": 1, "gpu": 1}
        #         # "momentum": tune.grid_search([0.1, 0.2]),
        #     }
        # },
        # "exp1": {
        #     "run": "train_func",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "/home/cewu/thirdHDD/yinghong/ray_results/choose_bz",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     # "num_gpus": 1,
        #     "config": {
        #         "batch_size": tune.grid_search([10, 15, 20]),
        #         # "resources": {"cpu": 1, "gpu": 1}
        #         # "momentum": tune.grid_search([0.1, 0.2]),
        #     }
        # }
        ###################
        # run on sjtu01
        ###################
        # "exp-go3-epoch50-1": {
        #     "run": "train_func",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/go1",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     # "num_gpus": 1,
        #
        #     "config": {
        #         "13-input_keep_prob": 1,
        #         "15-output_keep_prob": 1,
        #         "17-lstm_layers": 2,
        #         "19-clip": tune.grid_search([5,0]),
        #         "21-lr_method": "adam",
        #         "23-lr_decay": 0.9,
        #         "25-lr": 0.001,
        #         "27-decay_mode": tune.grid_search(["normal", "greedy", "greedy-half"])
        #         # "25-lr": tune.grid_search([0.001, 0.005]),
        #
        #         # "resources": {"cpu": 1, "gpu": 1}
        #         # "momentum": tune.grid_search([0.1, 0.2]),
        #     }
        # },
        # "exp-go3-epoch50-2": {
        #     "run": "train_func",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/go1",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     # "num_gpus": 1,
        #
        #     "config": {
        #         "13-input_keep_prob": 1,
        #         "15-output_keep_prob": 1,
        #         "17-lstm_layers": 2,
        #         "19-clip": tune.grid_search([5,0]),
        #         "21-lr_method": "sgd",
        #         "23-lr_decay": 0.9,
        #         "25-lr": 0.1,
        #         "27-decay_mode": tune.grid_search(["greedy-half"])
        #
        #         # "25-lr": tune.grid_search([0.001, 0.005]),
        #
        #         # "resources": {"cpu": 1, "gpu": 1}
        #         # "momentum": tune.grid_search([0.1, 0.2]),
        #     }
        # },
        # "exp-go3-epoch50-3": {
        #     "run": "train_func",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/go1",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     # "num_gpus": 1,
        #
        #     "config": {
        #         "13-input_keep_prob": 1,
        #         "15-output_keep_prob": 1,
        #         "17-lstm_layers": 1,
        #         "19-clip": 5,
        #         "21-lr_method": "adam",
        #         "23-lr_decay": 0.9,
        #         "25-lr": 0.003,
        #         "27-decay_mode": tune.grid_search(["normal", "greedy", "greedy-half"])
        #         # "25-lr": tune.grid_search([0.001, 0.005]),
        #
        #         # "resources": {"cpu": 1, "gpu": 1}
        #         # "momentum": tune.grid_search([0.1, 0.2]),
        #     }
        # },
        # "exp-go3-epoch50-4": {
        #     "run": "train_func",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/go1",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     # "num_gpus": 1,
        #
        #     "config": {
        #         "13-input_keep_prob": 1,
        #         "15-output_keep_prob": 1,
        #         "17-lstm_layers": 1,
        #         "19-clip": 5,
        #         "21-lr_method": "sgd",
        #         "23-lr_decay": 0.9,
        #         "25-lr": 0.1,
        #         "27-decay_mode": tune.grid_search(["greedy-half"])
        #         # "25-lr": tune.grid_search([0.001, 0.005]),
        #
        #         # "resources": {"cpu": 1, "gpu": 1}
        #         # "momentum": tune.grid_search([0.1, 0.2]),
        #     }
        # },
        # "exp-go3-epoch50-5": {
        #     "run": "train_func",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/go1",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     # "num_gpus": 1,
        #
        #     "config": {
        #         "13-input_keep_prob": 1,
        #         "15-output_keep_prob": 1,
        #         "17-lstm_layers": 2,
        #         "19-clip": tune.grid_search([0, 5]),
        #         "21-lr_method": "adam",
        #         "23-lr_decay": 0.9,
        #         "25-lr": 0.001,
        #         "27-decay_mode": tune.grid_search(["normal"]),
        #         "29-use_crf": False
        #         # "25-lr": tune.grid_search([0.001, 0.005]),
        #
        #         # "resources": {"cpu": 1, "gpu": 1}
        #         # "momentum": tune.grid_search([0.1, 0.2]),
        #     }
        # },
        # "exp-go3-epoch50-6": {
        #     "run": "train_func",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/go1",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     # "num_gpus": 1,
        #
        #     "config": {
        #         "13-input_keep_prob": 1,
        #         "15-output_keep_prob": 1,
        #         "17-lstm_layers": 1,
        #         "19-clip": 5,
        #         "21-lr_method": "adam",
        #         "23-lr_decay": 0.9,
        #         "25-lr": 0.003,
        #         "27-decay_mode": tune.grid_search(["normal"]),
        #         "29-use_crf": False
        #         # "25-lr": tune.grid_search([0.001, 0.005]),
        #
        #         # "resources": {"cpu": 1, "gpu": 1}
        #         # "momentum": tune.grid_search([0.1, 0.2]),
        #     }
        # },
        #
        # "exp-final-epoch30": {
        #     "run": "train_func_final",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/06-17",
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
        #         "17-lstm_layers": 2,
        #         "19-clip": 5,
        #         "21-lr_method": "adam",
        #         "23-lr_decay": 0.9,
        #         "25-lr": 0.001,
        #         "27-decay_mode": "none",
        #         # "25-lr": tune.grid_search([0.001, 0.005]),
        #
        #         # "resources": {"cpu": 1, "gpu": 1}
        #         # "momentum": tune.grid_search([0.1, 0.2]),
        #     }
        # },
        "exp-final-epoch30-sgd": {
            "run": "train_func_final",
            "stop": {"mean_accuracy": 99},
            "local_dir": "./ray_results/06-17",
            "trial_resources": {'cpu': 0, 'gpu': 1},
            # "num_gpus": 1,

            "config": {
                # "00-use_reg": tune.grid_search([False, True]),
                # "03-hidden_size_lstm": 300,
                # # "05-hidden_size_char": tune.grid_search([100, 50, 30]),
                # "05-hidden_size_char": 100,
                # # "07-dim_char": tune.grid_search([100, 50, 30]),
                # "07-dim_char": 100,
                "17-lstm_layers": 2,
                "19-clip": 5,
                "21-lr_method": "sgd",
                "23-lr_decay": 0.9,
                "25-lr": 0.015,
                "27-decay_mode": "none",
                # "25-lr": tune.grid_search([0.001, 0.005]),

                # "resources": {"cpu": 1, "gpu": 1}
                # "momentum": tune.grid_search([0.1, 0.2]),
            }
        },
        # "exp-final-epoch30-sgd": {
        #     "run": "train_func",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/final",
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
        #         "17-lstm_layers": 2,
        #         "19-clip": 5,
        #         "21-lr_method": "sgd",
        #         "23-lr_decay": 0.9,
        #         "25-lr": 0.015,
        #         "27-decay_mode": "4normal",
        #         # "25-lr": tune.grid_search([0.001, 0.005]),
        #
        #         # "resources": {"cpu": 1, "gpu": 1}
        #         # "momentum": tune.grid_search([0.1, 0.2]),
        #     }
        # },
    })



if __name__ == "__main__":
    main()
