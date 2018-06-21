import os

from config import Config
from model.data_utils import CoNLLDataset
from model.ner_model import NERModel


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

if __name__ == "__main__":
    pretrain()

    # creat trim
    # dev_name = "dev.eval"
    # extract_data(dev_name)















