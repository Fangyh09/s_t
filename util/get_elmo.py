from config import Config


def get_pre_embedding(mode):
    import h5py
    if mode == "dev":
        f = h5py.File('data/elmo_dev.embedding.h5','r')
        return f
        # dev_emb = {}
        # for key in f.keys():
        #     dev_emb[key] = f[key][:]
        # return dev_emb
    elif mode == "train":
        f = h5py.File('data/elmo_train.embedding.h5', 'r')
        return f
        # train_emb = {}
        # for key in f.keys():
        #     train_emb[key] = f[key][:]
        # return train_emb
    else:
        raise ValueError("Not fond mode", mode)




def build_conf(in_conf):
    config = Config()
    for key, val in in_conf.items():
        print(key, val)
        # config[key] = val
        setattr(config, key, val)
    setattr(config, "dir_model",
            getattr(config, "dir_output")+ \
                                 "model.weights/elmo-model")
    setattr(config, "path_log", getattr(config, "dir_output") + "log.txt")
    return config




configs = {
    "config1": {
    "nepochs": 100,
    "lstm_layers": 2,
    "reverse": False,
    "lr_decay": 0.9,
    "clip": 0,
    "dir_output": "rayresults/elmo-offline-config1/"
    },
    # todo
    "config2": {
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.95,
        "clip": 0,
        # todo
        "dir_output": "rayresults/elmo-offline-config2/"
    },
    # todo
    "config3": {
        "nepochs": 100,
        "lstm_layers": 3,
        "reverse": False,
        "lr_decay": 0.9,
        "clip": 0,
        # todo
        "dir_output": "rayresults/elmo-offline-config3/"
    },
    "config4": {
        "nepochs": 100,
        "lstm_layers": 2,
        # todo hard set
        "reverse": True,
        "lr_decay": 0.9,
        "clip": 0,
        "dir_output": "rayresults/elmo-offline-config4/"
    },
    # todo
    "config5": {
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": True,
        "lr_decay": 0.95,
        "clip": 0,
        # todo
        "dir_output": "rayresults/elmo-offline-config5/"
    },
    # todo
    "config6": {
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.9,
        "clip": 5,
        # todo
        "dir_output": "rayresults/elmo-offline-config6/"
    },
    "config7": {
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.92,
        "clip": 0,
        # todo
        "dir_output": "rayresults/elmo-offline-config7/"
    },
    "config8": {
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.9,
        "clip": 0,
        "elmo_drop": True,
        # todo
        "dir_output": "rayresults/elmo-offline-config8/"
    },
    "config9": {
        # fail
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.95,
        "clip": 0,
        "elmo_drop": True,
        # todo
        "dir_output": "rayresults/elmo-offline-config9/"
    },
    "config10": {
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.9,
        "clip": 0,
        "elmo_drop_drop": True,
        # todo
        "dir_output": "rayresults/elmo-offline-config10/"
    },
    "config11": {
        # fail
        # set cell dropout
        "cell_dropout": 0.5,
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.9,
        "clip": 0,
        # todo
        "dir_output": "rayresults/elmo-offline-config11/"
    },
    "config12": {
        # set cell dropout
        "cell_dropout": 0.5,
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.9,
        "clip": 0,
        # todo
        "dir_output": "rayresults/elmo-offline-config12/"
    },
    "config13": {
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.9,
        "clip": 5,
        "elmo_drop": True,
        # todo
        "dir_output": "rayresults/elmo-offline-config13/"
    },
    "config14": {
        #remove char_embeddings dropout
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.9,
        "clip": 5,
        "elmo_drop": True,
        # todo
        "dir_output": "rayresults/elmo-offline-config14/"
    },

}

