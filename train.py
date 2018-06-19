from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from config import Config
from tensorflow.python import debug as tf_debug
import tensorflow as tf
import ray
import ray.tune as tune

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
    default_config = Config()
    dev = CoNLLDataset(default_config.filename_dev, default_config.processing_word,
                       default_config.processing_tag, default_config.max_iter)
    train = CoNLLDataset(default_config.filename_train, default_config.processing_word,
                         default_config.processing_tag, default_config.max_iter)

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
        model.train(train, dev, reporter)

    # ray.init(redis_address="192.168.1.201:20198")
    ray.init(num_cpus=1, num_gpus=2)

    tune.register_trainable("train100iter", train_func)

    tune.run_experiments({
        "02-NoCNN": {
            "run": "train100iter",
            "stop": {"mean_accuracy": 99},
            "local_dir": "./ray_results/06-19",
            "trial_resources": {'cpu': 0, 'gpu': 1},
            "config": {
                "lstm_layers": 2,
                "clip": tune.grid_search([5, 0]),
                "lr_decay": tune.grid_search([0.9, 0.95])

            }
        },
        "01-HasCNN": {
            "run": "train100iter",
            "stop": {"mean_accuracy": 99},
            "local_dir": "./ray_results/06-19",
            "trial_resources": {'cpu': 0, 'gpu': 1},
            "config": {
                "lstm_layers": 2,
                # "clip": tune.grid_search([0, 5]),
                "filter_sizes": tune.grid_search([[3,4], [3,4,5]]),
            }
        },

    })



if __name__ == "__main__":
    main()
