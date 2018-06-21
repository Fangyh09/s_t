import ray
import ray.tune as tune
from allennlp.commands.elmo import ElmoEmbedder

from config import Config
from model.data_utils import CoNLLDataset
from model.ner_model import NERModel


# 4. get elmo


# usage
# "use_reg": tune.grid_search([False, True]),
# "hidden_size_lstm": 300,
# "hidden_size_char": 100,
# "dim_char": 100,
# "lr_method": "adam",
# "lr": 0.001,
# "lr_decay": 0.9,
# "lstm_layers": 2,
# "clip": tune.grid_search([0, 5]),
# "decay_mode": "normal",
# "lr": tune.grid_search([0.001, 0.005]),


def main():

    def train_func(_config, reporter):
        # tf.reset_default_graph()
        config = Config()
        for (key, val) in _config.items():
            # config[key] = val
            setattr(config, key, val)
        setattr(config, "dir_output", "")
        setattr(config, "nepochs", 100)

        model = NERModel(config)
        model.build()
        dev = CoNLLDataset(config.filename_dev,
                           config.processing_word,
                           config.processing_tag,
                           config.max_iter)
        train = CoNLLDataset(config.filename_train,
                             config.processing_word,
                             config.processing_tag,
                             config.max_iter)
        model.train(train, dev, reporter)

    # ray.init(redis_address="192.168.1.201:20198")
    ray.init(num_cpus=1, num_gpus=2)

    tune.register_trainable("finaltrain100iter", train_func)

    tune.run_experiments({
        # "02-NoCNN": {
        #     "run": "finaltrain100iter",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/06-19",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     "config": {
        #         "lstm_layers": 2,
        #         "clip": tune.grid_search([5, 0]),
        #         "lr_decay": tune.grid_search([0.9, 0.95])
        #
        #     }
        # },
        # "01-HasCNN": {
        #     "run": "finaltrain100iter",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/06-19",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     "config": {
        #         "lstm_layers": 2,
        #         # "clip": tune.grid_search([0, 5]),
        #         "filter_sizes": tune.grid_search([[3,4], [3,4,5]]),
        #     }
        # },
        # "02-CV-NoCNN": {
        #     "run": "finaltrain100iter",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/06-19",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     "config": {
        #         "lstm_layers": 2,
        #         "clip": tune.grid_search([5, 0]),
        #         # "lr_decay": tune.grid_search([0.9, 0.95]),
        #         "cv": True, # by hand
        #         # "reverse": tune.grid_search([True, False])
        #     }
        # },

        # "ReverseOrNot": {
        #     "run": "trainreverse",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/06-19",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     "config": {
        #         "lstm_layers": 2,
        #         "lr_decay": 0.9,
        #         "clip": 5,
        #         "reverse": tune.grid_search([False, True])
        #     }
        # },
        # "RandomTest": {
        #     "run": "randomtest",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/randomtest",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     "config": {
        #         "lstm_layers": 2,
        #         "lr_decay": 0.9,
        #         "clip": 5,
        #         "reverse": tune.grid_search([False, True])
        #     }
        # },
        # "RealHasCNN": {
        #     "run": "finaltrain100iter",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/06-19",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     "config": {
        #         "lstm_layers": 2,
        #         "use_cnn": True,
        #         "clip": tune.grid_search([0, 5]),
        #         "filter_sizes": tune.grid_search([[3, 4], [3, 4, 5]]),
        #     }
        # },

        "RealHasCNN-try2": {
            "run": "finaltrain100iter",
            "stop": {"mean_accuracy": 99},
            "local_dir": "./ray_results/06-19",
            "trial_resources": {'cpu': 0, 'gpu': 1},
            "config": {
                "lstm_layers": 2,
                "use_cnn": True,
                "lr_method": "adam",
                "filter_sizes": tune.grid_search([[3, 4], [3, 4, 5]]),
            }
        },
    })


def main2():
    # create instance of config

    options_file = "Elmo/data/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "Elmo/data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    print("loadding elmo")
    elmo = ElmoEmbedder(options_file, weight_file, cuda_device=0)
    print("finish loading")

    config = Config()

    # build model
    model = NERModel(config)
    model.build()
# model.restore_session("results/crf/model.weights/") # optional, restore weights
#model.reinitialize_weights("proj")

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, elmo, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, elmo, config.processing_word,
                         config.processing_tag, config.max_iter)

    # train model
    model.train(train, dev)


if __name__ == "__main__":
    main2()
