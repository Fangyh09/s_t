from model.data_utils import minibatches
import tensorflow as tf
import operator
from util.get_elmo import  get_pre_embedding
from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from util.get_elmo import get_pre_embedding, build_conf, configs
import os

def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned



def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.strip().split(" ")

        if words_raw == ["exit"]:
            break

        preds = model.predict(words_raw)
        to_print = align_data({"input": words_raw, "output": preds})

        for key, seq in to_print.items():
            model.logger.info(seq)

#
# def main2():
#     # create instance of config
#     config = Config()
#
#     # build model
#     model = NERModel(config)
#     model.build()
#     model.restore_session(config.dir_model)
#
#     # create dataset
#     test  = CoNLLDataset(config.filename_test, config.processing_word,
#                          config.processing_tag, config.max_iter)
#
#     # evaluate and interact
#     model.evaluate(test)
    interactive_shell(model)

def main():
    import os

    from config import Config
    from model.data_utils import CoNLLDataset
    from model.ner_model import NERModel
    # create instance of config
    config = Config()
    prefix = "/home/yinghong/project/tmp/s_t/ray_results"
    # pretrain_path = "/home/yinghong/project/tmp/s_t/ray_results/final/exp-final-epoch30" \
    #                 "/train_func_0_2018-06-16_01-24-13vmtghosb"
    # pretrain_path = \
        # os.path.join(prefix,"06-17/exp-final-epoch30/train_func_fi"
        #                     "nal_0_2018-06-17_11-41-242ciyu4yq")
    # pretrain_path = "/home/yinghong/project/tmp/s_t/ray_results/go1-old/exp" \
                    # "-go3/normal3"
    # pretrain_path = "/home/yinghong/project/tmp/s_t/ray_results/final/exp-final-epo" \
    #                 "ch30/train_func_final_0_2018-06-16_10-38-30qfc8b21c"

    # config_path = os.path.join(pretrain_path, "params.json")
    # with open(config_path) as fin:
    #     content = fin.read().replace('\n', '')
    #     import json
    #     j = json.loads(content)
    #     for (key, val) in j.items():
    #         setattr(config, key, val)
    # setattr(config, "lstm_layers", 2)
    # setattr(config, "clip", 5)
    # build model
    setattr(config, "lstm_layers", 2)

    setattr(config, "nepochs", 100)

    import datetime
    date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    dir_output = "results/finalrun/" + "main_layer2/" + date_str + "/"
    setattr(config, "dir_output", dir_output)
    setattr(config, "dir_model", dir_output + "model.weights/finalmodel")
    setattr(config, "path_log", dir_output + "log.txt")

    model = NERModel(config)
    model.build()

    model.restore_session(
        "/home/yinghong/project/tmp/s_t_rollback/ray_results/06-19/01-HasCNN/try3")

    # create dataset
    # test  = CoNLLDataset(config.filename_test, config.processing_word,
    #                      config.processing_tag, config.max_iter)
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)


    # evaluate and interact
    model.tmp(dev, outfile="result-dev-goo.txt")
    # interactive_shell(model)



def extract_data(fname):
    # fname = "dev.eval"
    fout_name = fname + ".trim-test"
    # !!! dont not use readlines later
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    fout = open(fout_name, "w+")
    for x in content:
        # rm '/n'
        ls = x.split(' ')
        fout.write(ls[0] + "\n")

def pretrain():
    import os

    from config import Config
    from model.data_utils import CoNLLDataset
    from model.ner_model import NERModel
    config = Config()
    # reverse,
    pretrain_path = "/home/yinghong/project/tmp/s_t_rollback/ray_results/06" \
                    "-19/01-HasCNN/try5"
    # reverse,
    # pretrain_path = "/home/yinghong/project/tmp/s_t_rollback/ray_results/06-19/best-HasCNN/try4"
    # reverse = True
    # cv = False

    config_path = os.path.join(pretrain_path, "params.json")
    with open(config_path) as fin:
        content = fin.read().replace('\n', '')
        import json
        j = json.loads(content)
        for (key, val) in j.items():
            setattr(config, key, val)
    model = NERModel(config)
    model.build()

    model.restore_session(
        os.path.join(pretrain_path, "results/tmptmptest/bz=10-training-"
                                    "bieo-nocnn/model.weights/"))

    # create dataset
    test = CoNLLDataset(config.filename_test, config.processing_word,
                        config.processing_tag, config.max_iter, test=True)
    dev = CoNLLDataset(config.filename_dev, config.processing_word,
                       config.processing_tag, config.max_iter)

    # evaluate and interact
    model.tmp(dev, outfile="result-test-google85.63.txt")


def elmo_pretrain():
    from allennlp.commands.elmo import ElmoEmbedder
    options_file = "Elmo/data/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "Elmo/data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    print("loadding elmo")
    elmo = ElmoEmbedder(options_file, weight_file, cuda_device=0)
    print("finish loading")

    print("ok??????????????????????")
    elmo_embedding = elmo.embed_sentence(["I", "Love", "You"])
    print("ok!!!!!!!!!!!!!!!!!!!!")

    from config import Config
    from model.data_utils import CoNLLDataset
    from model.ner_model import NERModel
    import ray
    import ray.tune as tune

    config = Config()
    # reverse,
    # pretrain_path = "/home/yinghong/project/tmp/s_t_rollback/ray_results/06" \
    #                 "-19/01-HasCNN/try5"
    # reverse,
    # pretrain_path = "/home/yinghong/project/tmp/s_t_rollback/ray_results/06-19/best-HasCNN/try4"
    # reverse = True
    # cv = False
    pretrain_path = "/SSD1/yinghong/tmp/s_t_elmo/ra" \
                    "yresults/elmo/tmptmptest/bz=10" \
                    "-training-bieo-nocnn/model.weights/elmo-model2018-06-22-08-15"

    # config_path = os.path.join(pretrain_path, "params.json")
    # with open(config_path) as fin:
    #     content = fin.read().replace('\n', '')
    #     import json
    #     j = json.loads(content)
    #     for (key, val) in j.items():
    #         setattr(config, key, val)
    setattr(config, "clip", 5)
    setattr(config, "lstm_layers", 2)
    model = NERModel(config)
    model.build()

    model.restore_session(pretrain_path)

    # create dataset
    test = CoNLLDataset(config.filename_test, config.processing_word,
                        config.processing_tag, config.max_iter, test=True)
    dev = CoNLLDataset(config.filename_dev, config.processing_word,
                       config.processing_tag, config.max_iter)

    # evaluate and interact
    model.tmp(test, elmo, outfile="result-test-google85.63.txt")


def static_elmo_pretrain(config_name, dataset, embeddings, outfile):
    config_class = build_conf(configs[config_name])

    model = NERModel(config_class)
    model.build()
    prefix = os.path.join("/SSD1/yinghong/tmp/s_t_elmo/",configs[
        config_name]["dir_output"])
    pretrain_path = os.path.join(prefix, "model.weights")
    # model.restore_session(configs[config_name]["pretrain_path"])
    model.restore_session(pretrain_path)
    # evaluate and interact
    model.tmp(dataset, embeddings, outfile=outfile)


def list2str(list):
    """
    :param list: [[1,2], [3]]
    :return:
    """
    return '_'.join('_'.join(str(x)) for x in list)



def ensemble(models, dataset, elmo_embeddings, outfile="result.txt", batch_size
= 10):
    fout = open(outfile, "w+")
    for idx, (words, labels)  in enumerate(minibatches(dataset, \
                                                      batch_size)):
        elmo_embedding = elmo_embeddings[str(idx)][:].tolist()

        tags_counter = {}
        orig_labels = {}
        scores = {}
        for model in models:
            cur_labels_pred, cur_prob_pred, _ = model.predict_batch(words,
                                                       elmo_embedding,
                                                       withprob=True)
            pred_str = list2str(cur_labels_pred)
            tags_counter[pred_str] = tags_counter.get(pred_str, 0) + 1
            orig_labels[pred_str] = cur_labels_pred
            scores[pred_str] += cur_prob_pred

        # labels_pred = orig_labels[max(tags_counter.iteritems(),
        #                     key=operator.itemgetter(1))[0]]

        labels_pred = orig_labels[max(tags_counter.items(),
                                      key=lambda x: (x[1], scores[x[0]]))]

        for sent in list(labels_pred):
            for wordidx in list(sent):
                tag = models[0].idx_to_tag[wordidx]
                # tbd add for bieo
                if tag[0] == 'E':
                    tag = "O"
                fout.write(tag + "\n")
            fout.write("\n")
        # index_i = 0
        # for sent in list(labels_pred):
        #     cur_prob_sent = list(prob_pred)[index_i]
        #     # index_j = 0
        #     for wordidx in list(sent):
        #         # cur_prob_word = list(cur_prob_sent)[index_j]
        #         tag = models[0].idx_to_tag[wordidx]
        #         # tbd add for bieo
        #         if tag[0] == 'E':
        #             tag = "O"
        #         fout.write(tag + "\n")
        #     fout.write("Prob is :" + str(cur_prob_sent) +
        #                "\n")
        #     # index_j += 1
        #     fout.write("\n")
        #     index_i += 1




if __name__ == "__main__":
    # elmo_pretrain()
    # from config import Config
    # default_config = Config()
    # dev = CoNLLDataset(default_config.filename_dev,
    #                    default_config.processing_word,
    #                    default_config.processing_tag,
    #                    default_config.max_iter)
    # train = CoNLLDataset(default_config.filename_train,
    #                      default_config.processing_word,
    #                      default_config.processing_tag,
    #                      default_config.max_iter)
    #
    # train_embeddings = get_pre_embedding('train')
    # dev_embeddings = get_pre_embedding('dev')
    #
    # config = build_conf(configs["config13"])
    # model = NERModel(config)
    # model.build()
    #
    # ensemble(models, dev, dev_embeddings, outfile="result-test-google85.63.txt")
    # creat trim
    # dev_name = "dev.eval"
    # extract_data(dev_name)
    mode = "dev"


    from config import Config
    default_config = Config()
    test = CoNLLDataset(default_config.filename_test, default_config.processing_word,
                        default_config.processing_tag, default_config.max_iter, test=True)
    dev = CoNLLDataset(default_config.filename_dev, default_config.processing_word,
                       default_config.processing_tag, default_config.max_iter)

    # train_embeddings = get_pre_embedding('train')
    # embeddings = get_pre_embedding(mode)
    import h5py
    embeddings = h5py.File("data/elmo_" + mode + ".embedding.h5", 'r')
    if mode == "dev":
        dataset = dev
        save_path = "presubmit"
    elif mode == "test":
        dataset = test
        save_path = "submit"

    # config_name = "config13"
    idx_set = [1,2,3,7,8,9,10,11,12,13,14,16,17,18]
    for idx in idx_set:
        config_name = "config" + str(idx)
        config = build_conf(configs[config_name])
        tf.reset_default_graph()
        static_elmo_pretrain(config_name, dataset, embeddings,
                             outfile= os.path.join(save_path,
                                              config_name+"-" + mode + ".predict"))













