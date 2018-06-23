# import pudb; pu.db
# from allennlp.commands.elmo import ElmoEmbedder
from config import Config
from model.data_utils import CoNLLDataset
from util.get_elmo import get_pre_embedding, build_conf, configs
import ray
import ray.tune as tune

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

# def get_pre_embedding(mode):
#     import h5py
#     if mode == "dev":
#         f = h5py.File('data/elmo_dev.embedding.h5','r')
#         return f
#         # dev_emb = {}
#         # for key in f.keys():
#         #     dev_emb[key] = f[key][:]
#         # return dev_emb
#     elif mode == "train":
#         f = h5py.File('data/elmo_train.embedding.h5', 'r')
#         return f
#         # train_emb = {}
#         # for key in f.keys():
#         #     train_emb[key] = f[key][:]
#         # return train_emb
#     else:
#         raise ValueError("Not fond mode", mode)


def main():
    #
    # options_file = "Elmo/data/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    # weight_file = "Elmo/data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    #
    # print("loadding elmo")
    # elmo = ElmoEmbedder(options_file, weight_file, cuda_device=0)
    # print("finish loading")
    #
    # print("ok??????????????????????")
    # elmo_embedding = elmo.embed_sentence(["I", "Love", "You"])
    # print("ok!!!!!!!!!!!!!!!!!!!!")
    from config import Config
    from model.data_utils import CoNLLDataset
    from model.ner_model import NERModel
    import ray
    import ray.tune as tune

    train_embeddings = get_pre_embedding('train')
    dev_embeddings = get_pre_embedding('dev')

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
        # MODE = {
        #     'dev': dev,
        #     'train': train
        # }
        #
        # mode = 'train'

        model.train(train, dev, train_embeddings, dev_embeddings)

    # ray.init(redis_address="192.168.1.201:20198")
    import os
    ray.init(num_cpus=1, num_gpus=1)

    tune.register_trainable("elmo_offline_train", train_func)

    tune.run_experiments({
        "ElmoOffline-NoCNN": {
            "run": "elmo_offline_train",
            "stop": {"mean_accuracy": 99},
            "local_dir": "./rayresults/elmo_offline_train",
            "trial_resources": {'cpu': 0, 'gpu': 1},
            "config": {
                "lstm_layers": 2,
                "reverse": False,
                "lr_decay": tune.grid_search([0.9, 0.95]),
                "clip": tune.grid_search([0, 5]),
                # "filter_sizes": tune.grid_search([[3,4], [3,4,5]]),
            }
        },
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

        # "RealHasCNN-try2": {
        #     "run": "finaltrain100iter",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/06-19",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     "config": {
        #         "lstm_layers": 2,
        #         "use_cnn": True,
        #         "lr_method": "adam",
        #         "filter_sizes": tune.grid_search([[3, 4], [3, 4, 5]]),
        #     }
        # },

        # "01-HasCNN": {
        #     "run": "elmo_train",
        #     "stop": {"mean_accuracy": 99},
        #     "local_dir": "./ray_results/elmo_train",
        #     "trial_resources": {'cpu': 0, 'gpu': 1},
        #     "config": {
        #         "lstm_layers": 2,
        #         # "clip": tune.grid_search([0, 5]),
        #         # "filter_sizes": tune.grid_search([[3,4], [3,4,5]]),
        #     }
        # },
    })




def justmain():
    # todo
    config_name = "config18"
    config = build_conf(configs[config_name])
    print(config_name)
    from model.ner_model import NERModel
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
    # MODE = {
    #     'dev': dev,
    #     'train': train
    # }
    #
    # mode = 'train'
    train_embeddings = get_pre_embedding('train')
    dev_embeddings = get_pre_embedding('dev')
    model.train(train, dev, train_embeddings, dev_embeddings)


if __name__ == "__main__":
    justmain()