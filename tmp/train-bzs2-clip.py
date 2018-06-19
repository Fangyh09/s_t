from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from config import Config
from tensorflow.python import debug as tf_debug
import tensorflow as tf

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
    # create instance of config

    best_score = -10
    best_conf = Config()
    char_ratio = [1, 1.5, 2]
    dim_char_poss = [30, 50, 100]
    word_ratio = [1, 1.2, 1.5, 2]
    lrs = [0.005, 0.001]
    use_regs = [True, False]
    lstm_layers = [1, 2]
    use_grus = [True, False]
    batch_sizes = [10, 12, 14, 16, 18, 20]
    # dim_word_pass
    fout = open("best_nocnn_config.txt", "a+")


    # attrs = vars(config)
    # {'kids': 0, 'name': 'Dog', 'color': 'Spotted', 'age': 10, 'legs': 2, 'smell': 'Alot'}
    # now dump this in some way or another
    # print ', '.join("%s: %s" % item for item in attrs.items())
    # config.dump()
    # print(config.__dict__)

    # build model
    for batch_size in batch_sizes:
        # for c_dim  in dim_char_poss:
        #     for w_ratio in word_ratio:
        #         for lr in lrs:
        #             for use_reg in use_regs:
        #                 for lstm_layer in lstm_layers:
        #                     for use_gru in use_grus:
        tf.reset_default_graph()

        config = Config()
        config["batch_size"] = batch_size
        # config["clip"] = 5
        config["dir_output"] = "results/tmptmptest/" + "trylr-bz=" + str(batch_size) + "/"
        # config['use_cnn'] = True
        # config['use_gru'] = use_gru
        # config['use_reg'] = use_reg
        # config['nepochs'] = 30
        # config['lr'] = lr
        # config['lstm_layers'] = lstm_layer
        # config['dim_char'] = c_dim
        # config['hidden_size_char'] = int(c_dim * c_ratio)
        # config['hidden_size_lstm'] = int(300 * w_ratio)
        # hidden_size_char = 100 # lstm on chars
        # hidden_size_lstm = 300 # lstm on word embeddings
        # dim_char = 100
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
        score = model.train(train, dev)
        if score >= best_score:
            best_score = score
            best_conf = config
            print("===> new best_score %f", best_score)
            # config.dump()
            # print("==|> current best_score %f" % best_score)
            fout.write("\n==|> score %f\n" % score)
            # best_conf.dump(fout=fout)
        else:
            print("==|> score %f" % score)
            # config.dump()

        # if score >= 75:
        #     fout.write("\n==|> score %f\n" % score)
        #     config.dump(fout=fout)


    print("current best_score %f", best_score)
    # best_conf.dump()

if __name__ == "__main__":
    main()
