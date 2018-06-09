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
            setattr(config, key, val)
        # config["dir_output"] = ""
        setattr(config, "dir_output", "")
        model = NERModel(config)
        model.build()
        model.train(train, dev, reporter)

    ray.init(num_gpus=1,num_cpus=2)

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
        "exp3": {
            "run": "train_func",
            "stop": {"mean_accuracy": 99},
            "local_dir": "/home/cewu/thirdHDD/yinghong/ray_results/oneshot",
            "trial_resources": {'cpu': 0, 'gpu': 1},
            # "num_gpus": 1,

            "config": {
                "batch_size": tune.grid_search([10, 15, 20]),
                "nepochs": 1,
                "use_reg": tune.grid_search([True, False]),
                "hidden_size_lstm": tune.grid_search([300, 400, 500]),
                "hidden_size_char": tune.grid_search([30, 50, 100]),
                "dim_char": tune.grid_search([30, 50, 100]),
                "filter_sizes": tune.grid_search([[3,4,5], [3,4], [3,4,5]]),
                "use_cnn": tune.grid_search([True, False]),
                "input_keep_prob": tune.grid_search([0.5, 1]),
                "output_keep_prob": tune.grid_search([0.5, 1]),
                "lstm_layers": tune.grid_search([1, 2, 5]),
                "clip": tune.grid_search([0, 5, 10]),
                "lr_method": tune.grid_search(["sgd", "adam", "adagrad", "rmsprop"]),
                "lr_decay": tune.grid_search([0.95, 0.9, 0.85, 0.8]),
                "lr": tune.grid_search([0.015, 0.005, 0.001, 0.005]),

                # "resources": {"cpu": 1, "gpu": 1}
                # "momentum": tune.grid_search([0.1, 0.2]),
            }
        },
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
    })



if __name__ == "__main__":
    main()
