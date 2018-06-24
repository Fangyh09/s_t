from config import Config
import os

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
    elif mode == "test":
        f = h5py.File('data/elmo_test.embedding.h5', 'r')
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
        "dir_output": "rayresults/elmo-offline-config6/",
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
        "dir_output": "rayresults/elmo-offline-config9/",
        # "pretrain_path": os.path.join(prefix,
        # "elmo-offline-config9/model.weights/elmo-model2018-06-23-02-07")

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

    "config15": {
        # simple cnn
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.9,
        "clip": 5,
        "elmo_drop": True,
        "use_cnn": True,
        # todo
        "dir_output": "rayresults/elmo-offline-config15/"
    },

    "config16": {
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.95,
        "clip": 0,
        "elmo_drop": True,
        # todo
        "dir_output": "rayresults/elmo-offline-config16/"
    },

    "config17": {
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.95,
        "clip": 5,
        "elmo_drop": True,
        # todo
        "dir_output": "rayresults/elmo-offline-config17/"
    },

    "config18": {
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.95,
        "clip": 0,
        "elmo_2drop": True,
        # todo
        "dir_output": "rayresults/elmo-offline-config18/"
    },
    "config19": {
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.95,
        "clip": 0,
        # todo
        "dir_output": "rayresults/elmo-offline-config19/"
    },
    "config20": {
        # fail
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.95,
        "clip": 0,
        "elmo_drop": True,
        # todo
        "dir_output": "rayresults/elmo-offline-config20/",
        # "pretrain_path": os.path.join(prefix,
        # "elmo-offline-config9/model.weights/elmo-model2018-06-23-02-07")

    },
    "config-ensem-1": {
        # 3 models
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.9,
        "lr": 0.01,
        "clip": 0,
        "elmo_2drop": True,
        # todo
        "dir_output": "rayresults/config-ensem-1/"
    },
    "config-ensem-2": {
        # 5 models
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.9,
        "lr": 0.01,
        "clip": 0,
        "elmo_2drop": True,
        # todo
        "dir_output": "rayresults/config-ensem-2/"
    },
    "config-ensem-3": {
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.9,
        "lr": 0.01,
        "clip": 0,
        "elmo_2drop": True,
        # todo
        "dir_output": "rayresults/config-ensem-3/"
    },
    "config-ensem-4": {
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.8,
        "lr": 0.01,
        "clip": 0,
        "elmo_2drop": True,
        # todo
        "dir_output": "rayresults/config-ensem-4/"
    },
    "config-ensem-5": {
        "nepochs": 100,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.9,
        "lr": 0.01,
        "clip": 0,
        "elmo_2drop": True,
        # todo
        "dir_output": "rayresults/config-ensem-5/"
    },
    "config-ensem-6": {
        "nepochs": 100,
        "use_tag_weight": True,
        "lstm_layers": 2,
        "reverse": False,
        "lr_decay": 0.95,
        "clip": 0,
        "elmo_drop": True,
        # todo
        "dir_output": "rayresults/config-ensem-6",
        # "pretrain_path": os.path.join(prefix,
        # "elmo-offline-config9/model.weights/elmo-model2018-06-23-02-07")

    },
    # "config19": {
    #     "nepochs": 100,
    #     "lstm_layers": 2,
    #     "reverse": False,
    #     "lr_decay": 0.95,
    #     "clip": 0,
    #     # todo
    #     "dir_output": "rayresults/elmo-offline-config19/"
    # },

}


from model.data_utils import pad_sequences
def get_outnet_feed_dict(words, elmo_embedding, labels=None, lr=None,
                         dropout=None):
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
    # config = Config()
    # perform padding of the given data
    # config = {}
    use_chars = True
    if use_chars:
        # yes
        char_ids, word_ids = zip(*words)
        word_ids, sequence_lengths = pad_sequences(word_ids, 0)
        char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                                               nlevels=2)
    else:
        word_ids, sequence_lengths = pad_sequences(words, 0)

    feed = {
        "word_ids:0": word_ids,
        "sequence_lengths:0": sequence_lengths,
        "lr_1:0": elmo_embedding
    }

    if use_chars:
        feed["char_ids:0"] = char_ids
        feed["word_lengths:0"] = word_lengths

    if labels is not None:
        labels, _ = pad_sequences(labels, 0)
        feed["labels:0"] = labels

    if lr is not None:
        feed["lr:0"] = lr

    if dropout is not None:
        feed["dropout:0"] = dropout

    return feed, sequence_lengths
